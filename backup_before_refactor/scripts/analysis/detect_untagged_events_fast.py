#!/usr/bin/env python3
"""
Fast analysis to detect potential untagged media change events.
Samples wells to speed up processing.
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

def load_event_summary():
    """Get summary of recorded events."""
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    
    if not event_path.exists():
        return None, None
    
    events_df = pd.read_parquet(event_path)
    events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    # Get media change summary
    media_changes = events_df[events_df['title'] == 'Medium Change'].copy()
    
    # Count events per plate
    events_per_plate = media_changes.groupby('plate_id').size().reset_index(name='n_recorded_events')
    
    return media_changes, events_per_plate

def quick_spike_detection(plate_id, n_wells_sample=5):
    """Quick spike detection on sample of wells."""
    database_url = os.getenv('DATABASE_URL')
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # First get well count
    count_query = f"""
    SELECT COUNT(DISTINCT well_number) as n_wells
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND is_excluded = false
    """
    
    well_count = conn.execute(count_query).fetchone()[0]
    
    # Get sample of wells
    sample_query = f"""
    WITH sampled_wells AS (
        SELECT DISTINCT well_number
        FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
        WHERE plate_id::text = '{plate_id}' 
        AND is_excluded = false
        ORDER BY RANDOM()
        LIMIT {n_wells_sample}
    )
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND well_number IN (SELECT well_number FROM sampled_wells)
    AND is_excluded = false
    ORDER BY well_id, timestamp
    """
    
    data = conn.execute(sample_query).fetchdf()
    conn.close()
    
    if len(data) == 0:
        return 0, 0, []
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    plate_start = data['timestamp'].min()
    data['elapsed_hours'] = (data['timestamp'] - plate_start).dt.total_seconds() / 3600
    
    # Initialize detector
    detector = MediaChangeDetector(variance_threshold=1.5, peak_prominence=0.3)
    
    # Detect spikes in sampled wells
    all_spike_times = []
    wells = data['well_id'].unique()
    
    for well_id in wells:
        well_data = data[data['well_id'] == well_id].copy()
        if len(well_data) > 50:
            spikes = detector.detect_media_changes_from_variance(well_data)
            all_spike_times.extend([s['detected_time_hours'] for s in spikes])
    
    # Cluster spike times (within 6 hours)
    if len(all_spike_times) == 0:
        return well_count, 0, []
    
    all_spike_times = sorted(all_spike_times)
    clustered_events = []
    current_cluster = [all_spike_times[0]]
    
    for spike_time in all_spike_times[1:]:
        if spike_time - current_cluster[-1] < 6:
            current_cluster.append(spike_time)
        else:
            clustered_events.append(np.mean(current_cluster))
            current_cluster = [spike_time]
    
    clustered_events.append(np.mean(current_cluster))
    
    # Scale up estimate based on sampling
    estimated_total_events = len(clustered_events) * (well_count / len(wells))
    
    return well_count, estimated_total_events, clustered_events

def analyze_all_plates_fast():
    """Fast analysis of all plates."""
    print("Loading event data...")
    media_changes, events_per_plate = load_event_summary()
    
    if events_per_plate is None:
        print("No event data found!")
        return None
    
    print(f"Found {len(media_changes)} recorded media change events")
    print(f"Across {len(events_per_plate)} plates")
    
    # Load all plates
    step1_path = project_root / "results" / "data" / "step1_quality_assessment_all_plates.parquet"
    step1_data = pd.read_parquet(step1_path)
    all_plates = step1_data['plate_id'].unique()
    
    print(f"\nAnalyzing {len(all_plates)} plates (sampling 5 wells per plate)...")
    
    results = []
    
    for i, plate_id in enumerate(all_plates):
        print(f"\rProcessing plate {i+1}/{len(all_plates)}...", end='', flush=True)
        
        # Get recorded events for this plate
        n_recorded = events_per_plate[events_per_plate['plate_id'] == plate_id]['n_recorded_events'].values
        n_recorded = n_recorded[0] if len(n_recorded) > 0 else 0
        
        # Quick spike detection
        try:
            n_wells, estimated_events, event_times = quick_spike_detection(plate_id)
            
            results.append({
                'plate_id': plate_id,
                'n_wells': n_wells,
                'n_recorded_events': n_recorded,
                'estimated_detected_events': estimated_events,
                'detected_event_times': event_times,
                'has_recorded': n_recorded > 0,
                'has_detected': estimated_events > 0
            })
        except Exception as e:
            print(f"\nError processing plate {plate_id}: {e}")
            continue
    
    print("\n\nAnalysis complete!")
    
    return pd.DataFrame(results)

def create_summary_visualization(results_df):
    """Create summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analysis of Recorded vs Detected Media Change Events', fontsize=16)
    
    # 1. Overall statistics
    ax = axes[0, 0]
    ax.axis('off')
    
    # Calculate statistics
    total_plates = len(results_df)
    plates_with_recorded = results_df['has_recorded'].sum()
    plates_with_detected = results_df['has_detected'].sum()
    plates_with_both = ((results_df['has_recorded']) & (results_df['has_detected'])).sum()
    plates_detected_only = ((~results_df['has_recorded']) & (results_df['has_detected'])).sum()
    plates_recorded_only = ((results_df['has_recorded']) & (~results_df['has_detected'])).sum()
    
    total_recorded = results_df['n_recorded_events'].sum()
    total_detected = results_df['estimated_detected_events'].sum()
    
    stats_text = f"""
    📊 EVENT DETECTION SUMMARY
    
    Total Plates Analyzed: {total_plates}
    
    PLATE STATISTICS:
    • Plates with recorded events: {plates_with_recorded} ({plates_with_recorded/total_plates*100:.1f}%)
    • Plates with detected spikes: {plates_with_detected} ({plates_with_detected/total_plates*100:.1f}%)
    • Plates with both: {plates_with_both}
    • Plates with detected only: {plates_detected_only} (potential untagged events)
    • Plates with recorded only: {plates_recorded_only} (missed by detection)
    
    EVENT COUNTS:
    • Total recorded events: {total_recorded}
    • Total detected events (estimated): {total_detected:.0f}
    • Potential untagged rate: {(total_detected-total_recorded)/total_detected*100:.1f}%
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 2. Venn diagram-like visualization
    ax = axes[0, 1]
    categories = ['Recorded\nOnly', 'Both', 'Detected\nOnly', 'Neither']
    counts = [
        plates_recorded_only,
        plates_with_both,
        plates_detected_only,
        total_plates - plates_recorded_only - plates_with_both - plates_detected_only
    ]
    colors = ['#ff9999', '#99ff99', '#9999ff', '#cccccc']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Plates')
    ax.set_title('Plate Classification by Event Detection')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # 3. Scatter plot of recorded vs detected
    ax = axes[1, 0]
    ax.scatter(results_df['n_recorded_events'], 
               results_df['estimated_detected_events'],
               alpha=0.6, s=50)
    
    # Add diagonal line
    max_events = max(results_df['n_recorded_events'].max(), 
                     results_df['estimated_detected_events'].max())
    ax.plot([0, max_events], [0, max_events], 'r--', alpha=0.5, label='Perfect match')
    
    ax.set_xlabel('Recorded Events')
    ax.set_ylabel('Detected Events (estimated)')
    ax.set_title('Recorded vs Detected Events per Plate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Histogram of detection differences
    ax = axes[1, 1]
    differences = results_df['estimated_detected_events'] - results_df['n_recorded_events']
    
    ax.hist(differences, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.axvline(differences.mean(), color='orange', linestyle='--', 
               label=f'Mean: {differences.mean():.1f}')
    
    ax.set_xlabel('Detected - Recorded Events')
    ax.set_ylabel('Number of Plates')
    ax.set_title('Distribution of Detection Differences')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'event_detection_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed view of plates with untagged events
    untagged_plates = results_df[
        (~results_df['has_recorded']) & (results_df['has_detected'])
    ].sort_values('estimated_detected_events', ascending=False)
    
    if len(untagged_plates) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Show top 10 plates with untagged events
        top_untagged = untagged_plates.head(10)
        
        x = range(len(top_untagged))
        ax.bar(x, top_untagged['estimated_detected_events'], 
               color='red', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Plate Index')
        ax.set_ylabel('Estimated Number of Events')
        ax.set_title(f'Top {len(top_untagged)} Plates with Untagged Events Only')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Plate {i+1}" for i in x], rotation=45)
        
        # Add well count as text
        for i, (idx, row) in enumerate(top_untagged.iterrows()):
            ax.text(i, row['estimated_detected_events'] + 0.5,
                    f"{row['n_wells']} wells",
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'untagged_events_plates.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run fast analysis of untagged events."""
    print("=== Fast Detection of Untagged Media Change Events ===\n")
    
    # Run analysis
    results_df = analyze_all_plates_fast()
    
    if results_df is None:
        return
    
    # Save results
    results_path = data_dir / "event_detection_analysis.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_summary_visualization(results_df)
    
    print(f"\nVisualizations saved to: {fig_dir}")
    print("  - event_detection_summary.png")
    print("  - untagged_events_plates.png (if applicable)")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    
    # Plates with potential untagged events
    untagged_only = results_df[
        (~results_df['has_recorded']) & (results_df['has_detected'])
    ]
    
    if len(untagged_only) > 0:
        print(f"\n🔴 {len(untagged_only)} plates have detected events but NO recorded events!")
        print("These likely have untagged media changes:")
        for _, plate in untagged_only.nlargest(5, 'estimated_detected_events').iterrows():
            print(f"  - Plate {plate['plate_id']}: ~{plate['estimated_detected_events']:.0f} events detected")

if __name__ == "__main__":
    main()