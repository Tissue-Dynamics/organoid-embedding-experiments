#!/usr/bin/env python3
"""
Analyze time series data to detect potential untagged media change events.
Compare variance-based detection with recorded events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# Setup paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "step3_validation"

# Import from step3
import sys
sys.path.append(str(project_root / "data"))
from step3_event_detection import MediaChangeDetector

def load_all_plates():
    """Load all plates from Step 1 results."""
    step1_path = project_root / "results" / "data" / "step1_quality_assessment_all_plates.parquet"
    if not step1_path.exists():
        raise FileNotFoundError(f"Step 1 data not found at {step1_path}")
    
    step1_data = pd.read_parquet(step1_path)
    return step1_data['plate_id'].unique()

def load_event_data():
    """Load all event data."""
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    
    if not event_path.exists():
        print("Warning: Event data not found.")
        return pd.DataFrame()
    
    events_df = pd.read_parquet(event_path)
    events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    # Filter for media changes
    media_changes = events_df[events_df['title'] == 'Medium Change'].copy()
    
    return events_df, media_changes

def load_plate_time_series(plate_id):
    """Load time series data for a plate."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    query = f"""
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND is_excluded = false
    ORDER BY well_id, timestamp
    """
    
    data = conn.execute(query).fetchdf()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate elapsed hours
    if len(data) > 0:
        plate_start = data['timestamp'].min()
        data['elapsed_hours'] = (data['timestamp'] - plate_start).dt.total_seconds() / 3600
    
    conn.close()
    return data

def detect_variance_spikes(time_series_data, detector):
    """Detect spikes using variance method for all wells in a plate."""
    wells = time_series_data['well_id'].unique()
    all_detected = []
    
    for well_id in wells:
        well_data = time_series_data[time_series_data['well_id'] == well_id].copy()
        
        if len(well_data) < 50:  # Skip if too few points
            continue
            
        # Detect spikes
        detected_changes = detector.detect_media_changes_from_variance(well_data)
        
        for change in detected_changes:
            change['well_id'] = well_id
            all_detected.append(change)
    
    return all_detected

def analyze_untagged_events():
    """Main analysis to find untagged events."""
    print("Loading data...")
    
    # Load event data
    all_events, media_changes = load_event_data()
    print(f"Total events in database: {len(all_events)}")
    print(f"Media change events: {len(media_changes)}")
    print(f"Plates with media changes: {media_changes['plate_id'].nunique()}")
    
    # Load all plates
    all_plates = load_all_plates()
    print(f"\nTotal plates in dataset: {len(all_plates)}")
    
    # Initialize detector with sensitive settings
    detector = MediaChangeDetector(variance_threshold=1.5, peak_prominence=0.3)
    
    # Storage for results
    results = []
    
    # Sample analysis on subset of plates
    sample_size = min(10, len(all_plates))
    sample_plates = np.random.choice(all_plates, sample_size, replace=False)
    
    print(f"\nAnalyzing {sample_size} sample plates...")
    
    for i, plate_id in enumerate(sample_plates):
        print(f"\rProcessing plate {i+1}/{sample_size}...", end='', flush=True)
        
        # Load time series
        try:
            ts_data = load_plate_time_series(plate_id)
            if len(ts_data) == 0:
                continue
        except Exception as e:
            print(f"\nError loading plate {plate_id}: {e}")
            continue
        
        # Get recorded events for this plate
        plate_events = media_changes[media_changes['plate_id'] == plate_id]
        
        # Convert event times to elapsed hours
        recorded_times = []
        if len(plate_events) > 0 and len(ts_data) > 0:
            plate_start = ts_data['timestamp'].min()
            for _, event in plate_events.iterrows():
                elapsed = (event['occurred_at'] - plate_start).total_seconds() / 3600
                if elapsed >= 0:  # Only include events after plate start
                    recorded_times.append(elapsed)
        
        # Detect spikes using variance method
        detected_spikes = detect_variance_spikes(ts_data, detector)
        detected_times = [s['detected_time_hours'] for s in detected_spikes]
        
        # Match detected with recorded (within 6 hour window)
        matched_recorded = set()
        matched_detected = set()
        unmatched_detected = []
        
        for det_time in detected_times:
            matched = False
            for rec_time in recorded_times:
                if abs(det_time - rec_time) < 6:  # 6 hour tolerance
                    matched_recorded.add(rec_time)
                    matched_detected.add(det_time)
                    matched = True
                    break
            
            if not matched:
                unmatched_detected.append(det_time)
        
        unmatched_recorded = [t for t in recorded_times if t not in matched_recorded]
        
        # Store results
        results.append({
            'plate_id': plate_id,
            'n_wells': ts_data['well_id'].nunique(),
            'duration_hours': ts_data['elapsed_hours'].max() if len(ts_data) > 0 else 0,
            'recorded_events': len(recorded_times),
            'detected_spikes': len(detected_times),
            'matched_events': len(matched_detected),
            'untagged_spikes': len(unmatched_detected),
            'missed_recorded': len(unmatched_recorded),
            'untagged_spike_times': unmatched_detected,
            'missed_recorded_times': unmatched_recorded
        })
    
    print("\n\nAnalysis complete!")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Plates analyzed: {len(results_df)}")
    print(f"Total recorded events: {results_df['recorded_events'].sum()}")
    print(f"Total detected spikes: {results_df['detected_spikes'].sum()}")
    print(f"Matched events: {results_df['matched_events'].sum()}")
    print(f"Potential untagged events: {results_df['untagged_spikes'].sum()}")
    print(f"Recorded but not detected: {results_df['missed_recorded'].sum()}")
    
    # Detection rates
    total_recorded = results_df['recorded_events'].sum()
    total_detected = results_df['detected_spikes'].sum()
    total_matched = results_df['matched_events'].sum()
    
    if total_recorded > 0:
        print(f"\nDetection rate of recorded events: {total_matched/total_recorded*100:.1f}%")
    if total_detected > 0:
        print(f"Untagged spike rate: {results_df['untagged_spikes'].sum()/total_detected*100:.1f}%")
    
    # Per-plate statistics
    print("\n=== PER-PLATE STATISTICS ===")
    print("Plates with untagged spikes:", len(results_df[results_df['untagged_spikes'] > 0]))
    print("Average untagged spikes per plate:", results_df['untagged_spikes'].mean())
    print("Max untagged spikes in a plate:", results_df['untagged_spikes'].max())
    
    return results_df

def create_visualization(results_df):
    """Create visualization of tagged vs untagged events."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analysis of Untagged Media Change Events', fontsize=16)
    
    # 1. Overview bar chart
    ax = axes[0, 0]
    summary_data = pd.DataFrame({
        'Recorded Events': [results_df['recorded_events'].sum()],
        'Detected Spikes': [results_df['detected_spikes'].sum()],
        'Matched': [results_df['matched_events'].sum()],
        'Untagged': [results_df['untagged_spikes'].sum()],
        'Missed': [results_df['missed_recorded'].sum()]
    })
    summary_data.plot(kind='bar', ax=ax, rot=0)
    ax.set_ylabel('Count')
    ax.set_title('Event Detection Summary')
    ax.legend(loc='upper right')
    
    # 2. Per-plate comparison
    ax = axes[0, 1]
    plates_with_events = results_df[results_df['detected_spikes'] > 0].copy()
    if len(plates_with_events) > 0:
        x = range(len(plates_with_events))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], plates_with_events['recorded_events'], 
               width, label='Recorded', alpha=0.8)
        ax.bar([i + width/2 for i in x], plates_with_events['detected_spikes'], 
               width, label='Detected', alpha=0.8)
        
        ax.set_xlabel('Plate Index')
        ax.set_ylabel('Number of Events')
        ax.set_title('Recorded vs Detected Events by Plate')
        ax.legend()
        ax.set_xticks(range(len(plates_with_events)))
    
    # 3. Untagged events distribution
    ax = axes[1, 0]
    untagged_counts = results_df['untagged_spikes'].value_counts().sort_index()
    ax.bar(untagged_counts.index, untagged_counts.values)
    ax.set_xlabel('Number of Untagged Spikes')
    ax.set_ylabel('Number of Plates')
    ax.set_title('Distribution of Untagged Spikes per Plate')
    
    # 4. Detection efficiency
    ax = axes[1, 1]
    # Calculate detection efficiency per plate
    efficiency_data = []
    for _, row in results_df.iterrows():
        if row['recorded_events'] > 0:
            efficiency = row['matched_events'] / row['recorded_events'] * 100
            efficiency_data.append(efficiency)
    
    if efficiency_data:
        ax.hist(efficiency_data, bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(efficiency_data), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(efficiency_data):.1f}%')
        ax.set_xlabel('Detection Efficiency (%)')
        ax.set_ylabel('Number of Plates')
        ax.set_title('Detection Efficiency of Recorded Events')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'untagged_events_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed timeline visualization for a sample plate
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Find a plate with both tagged and untagged events
    candidate_plates = results_df[
        (results_df['untagged_spikes'] > 0) & 
        (results_df['recorded_events'] > 0)
    ].sort_values('untagged_spikes', ascending=False)
    
    if len(candidate_plates) > 0:
        sample_plate = candidate_plates.iloc[0]
        
        # Load actual time series for visualization
        ts_data = load_plate_time_series(sample_plate['plate_id'])
        
        if len(ts_data) > 0:
            # Plot time series for first few wells
            wells = ts_data['well_id'].unique()[:3]
            colors = plt.cm.tab10(np.linspace(0, 1, len(wells)))
            
            for i, well_id in enumerate(wells):
                well_data = ts_data[ts_data['well_id'] == well_id]
                ax.plot(well_data['elapsed_hours'], well_data['o2_percent'], 
                       alpha=0.6, label=f'Well {well_id.split("_")[-1]}', color=colors[i])
            
            # Add recorded events
            events_df, media_changes = load_event_data()
            plate_events = media_changes[media_changes['plate_id'] == sample_plate['plate_id']]
            
            if len(plate_events) > 0:
                plate_start = ts_data['timestamp'].min()
                for _, event in plate_events.iterrows():
                    event_time = (event['occurred_at'] - plate_start).total_seconds() / 3600
                    if 0 <= event_time <= ts_data['elapsed_hours'].max():
                        ax.axvline(event_time, color='green', linestyle='-', 
                                  alpha=0.7, linewidth=2, label='Recorded' if _ == 0 else '')
            
            # Add untagged spikes
            for i, spike_time in enumerate(sample_plate['untagged_spike_times']):
                ax.axvline(spike_time, color='red', linestyle='--', 
                          alpha=0.7, linewidth=2, label='Untagged' if i == 0 else '')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('O₂ (%)')
            ax.set_title(f'Example Plate with Untagged Events - {sample_plate["untagged_spikes"]} untagged, {sample_plate["recorded_events"]} recorded')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'untagged_events_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run the untagged events analysis."""
    print("=== Detecting Untagged Media Change Events ===\n")
    
    # Run analysis
    results_df = analyze_untagged_events()
    
    # Save results
    results_path = data_dir / "untagged_events_analysis.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualization(results_df)
    
    print(f"\nVisualizations saved to: {fig_dir}")
    print("  - untagged_events_analysis.png")
    print("  - untagged_events_timeline.png")
    
    # Print detailed breakdown for plates with many untagged events
    print("\n=== PLATES WITH MOST UNTAGGED EVENTS ===")
    top_untagged = results_df.nlargest(5, 'untagged_spikes')
    for _, plate in top_untagged.iterrows():
        if plate['untagged_spikes'] > 0:
            print(f"\nPlate {plate['plate_id']}:")
            print(f"  Duration: {plate['duration_hours']:.1f} hours")
            print(f"  Recorded events: {plate['recorded_events']}")
            print(f"  Detected spikes: {plate['detected_spikes']}")
            print(f"  Untagged spikes: {plate['untagged_spikes']}")
            print(f"  Untagged times: {[f'{t:.1f}h' for t in plate['untagged_spike_times'][:5]]}")
            if len(plate['untagged_spike_times']) > 5:
                print(f"    ... and {len(plate['untagged_spike_times'])-5} more")

if __name__ == "__main__":
    main()