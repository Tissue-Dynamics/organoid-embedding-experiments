#!/usr/bin/env python3
"""
Find the actual media change spikes within ±10 hours of recorded event times.
This accounts for timing uncertainty in the event database.
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
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# Setup paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "step3_validation"

def load_plate_data(plate_id):
    """Load all time series data for a plate."""
    database_url = os.getenv('DATABASE_URL')
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
    conn.close()
    
    if len(data) == 0:
        return None
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    plate_start = data['timestamp'].min()
    data['elapsed_hours'] = (data['timestamp'] - plate_start).dt.total_seconds() / 3600
    
    return (data, plate_start)

def detect_all_spikes_in_well(well_data, min_height=10.0):
    """Detect all significant spikes in a well."""
    if len(well_data) < 20:
        return []
    
    o2_values = well_data['o2_percent'].values
    elapsed_hours = well_data['elapsed_hours'].values
    
    # Smooth data
    smoothed_o2 = uniform_filter1d(o2_values, size=3)
    
    # Find peaks
    peaks, properties = find_peaks(smoothed_o2, 
                                  height=min_height,  # Minimum height above baseline
                                  prominence=min_height * 0.8,  # Prominence
                                  distance=20)  # Min 20 points between peaks
    
    spikes = []
    for peak_idx in peaks:
        # Calculate baseline before peak
        baseline_start = max(0, peak_idx - 20)
        baseline_end = max(0, peak_idx - 5)
        baseline = np.median(o2_values[baseline_start:baseline_end])
        
        spike_height = o2_values[peak_idx] - baseline
        
        if spike_height >= min_height:
            spikes.append({
                'time': elapsed_hours[peak_idx],
                'height': spike_height,
                'peak_value': o2_values[peak_idx],
                'baseline': baseline
            })
    
    return spikes

def find_best_spike_match(recorded_time, all_spike_times, time_window=10.0):
    """Find the best matching spike within time window of recorded event."""
    best_match = None
    best_score = -1
    
    for spike_data in all_spike_times:
        time_diff = abs(spike_data['time'] - recorded_time)
        
        if time_diff <= time_window:
            # Score based on: number of wells affected and spike height
            score = spike_data['n_wells'] * spike_data['mean_height']
            
            if score > best_score:
                best_score = score
                best_match = spike_data
                best_match['time_diff'] = spike_data['time'] - recorded_time
    
    return best_match

def analyze_plate_events(plate_id):
    """Analyze events for a single plate."""
    print(f"\nAnalyzing plate {plate_id}...")
    
    # Load data
    result = load_plate_data(plate_id)
    if result is None:
        print(f"  No data found for plate {plate_id}")
        return None
    
    data, plate_start = result
    
    # Get recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    events_df = pd.read_parquet(event_path)
    events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    plate_events = events_df[
        (events_df['plate_id'] == plate_id) & 
        (events_df['title'] == 'Medium Change')
    ].copy()
    
    if len(plate_events) == 0:
        print("  No recorded events for this plate")
        return None
    
    # Convert recorded times to elapsed hours
    recorded_times = []
    for _, event in plate_events.iterrows():
        elapsed = (event['occurred_at'] - plate_start).total_seconds() / 3600
        if 0 <= elapsed <= data['elapsed_hours'].max():
            recorded_times.append(elapsed)
    
    print(f"  Found {len(recorded_times)} recorded events")
    
    # Detect all spikes in all wells
    wells = data['well_id'].unique()
    all_spikes_by_well = {}
    
    for well_id in wells:
        well_data = data[data['well_id'] == well_id]
        spikes = detect_all_spikes_in_well(well_data, min_height=10.0)
        all_spikes_by_well[well_id] = spikes
    
    # Find synchronized spikes
    all_spike_times = []
    for well_id, spikes in all_spikes_by_well.items():
        for spike in spikes:
            all_spike_times.append(spike['time'])
    
    if not all_spike_times:
        return None
    
    # Cluster spikes to find synchronized events
    all_spike_times = sorted(all_spike_times)
    synchronized_spikes = []
    
    i = 0
    while i < len(all_spike_times):
        current_time = all_spike_times[i]
        
        # Count wells with spikes near this time
        wells_affected = []
        spike_heights = []
        
        for well_id, spikes in all_spikes_by_well.items():
            for spike in spikes:
                if abs(spike['time'] - current_time) <= 2.0:  # 2 hour window
                    wells_affected.append(well_id)
                    spike_heights.append(spike['height'])
                    break
        
        if len(wells_affected) >= 10:  # At least 10 wells
            synchronized_spikes.append({
                'time': current_time,
                'n_wells': len(wells_affected),
                'well_fraction': len(wells_affected) / len(wells),
                'mean_height': np.mean(spike_heights),
                'std_height': np.std(spike_heights)
            })
            
            # Skip past this cluster
            while i < len(all_spike_times) and all_spike_times[i] - current_time < 2.0:
                i += 1
        else:
            i += 1
    
    print(f"  Found {len(synchronized_spikes)} synchronized spike events")
    
    # Match recorded events to actual spikes
    matches = []
    for rec_time in recorded_times:
        best_match = find_best_spike_match(rec_time, synchronized_spikes, time_window=10.0)
        
        matches.append({
            'recorded_time': rec_time,
            'matched_spike': best_match,
            'plate_id': plate_id
        })
    
    return {
        'plate_id': plate_id,
        'data': data,
        'recorded_times': recorded_times,
        'synchronized_spikes': synchronized_spikes,
        'matches': matches,
        'all_spikes_by_well': all_spikes_by_well
    }

def create_event_alignment_visualization(results):
    """Create visualization showing recorded vs actual event times."""
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4*len(results)))
    if len(results) == 1:
        axes = [axes]
    
    fig.suptitle('Recorded Events vs Actual Media Change Spikes (±10 hour window)', fontsize=16)
    
    for idx, plate_result in enumerate(results):
        ax = axes[idx]
        
        # Plot time series for a few sample wells
        data = plate_result['data']
        wells = sorted(data['well_id'].unique())
        sample_wells = wells[::max(1, len(wells)//10)][:10]  # Sample 10 wells
        
        for i, well_id in enumerate(sample_wells):
            well_data = data[data['well_id'] == well_id]
            ax.plot(well_data['elapsed_hours'], 
                   well_data['o2_percent'] + i*5,
                   'gray', alpha=0.3, linewidth=0.8)
        
        # Plot synchronized spikes
        for spike in plate_result['synchronized_spikes']:
            height = spike['n_wells'] / len(wells) * 100  # Convert to percentage
            ax.scatter(spike['time'], -10, 
                      s=height*5, alpha=0.6, color='green',
                      label='Detected spikes' if spike == plate_result['synchronized_spikes'][0] else '')
        
        # Plot recorded events and their matches
        for match in plate_result['matches']:
            rec_time = match['recorded_time']
            
            # Recorded event line
            ax.axvline(rec_time, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            
            # Show search window
            ax.axvspan(rec_time - 10, rec_time + 10, 
                      alpha=0.1, color='blue')
            
            # If matched, draw connection
            if match['matched_spike']:
                actual_time = match['matched_spike']['time']
                time_diff = match['matched_spike']['time_diff']
                
                # Draw arrow from recorded to actual
                ax.annotate('', xy=(actual_time, -5), 
                           xytext=(rec_time, -5),
                           arrowprops=dict(arrowstyle='->',
                                         color='red',
                                         lw=2))
                
                # Add time difference label
                ax.text((rec_time + actual_time) / 2, -7,
                       f'{time_diff:+.1f}h',
                       ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('O₂ (%) / Event indicators')
        ax.set_title(f'Plate {idx+1}: {plate_result["plate_id"]}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        if idx == 0:
            ax.scatter([], [], s=100, alpha=0.6, color='green', label='Detected spike clusters')
            ax.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Recorded event time')
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'event_timing_alignment.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics(all_results):
    """Create summary of timing differences and spike characteristics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Media Change Event Timing Analysis Summary', fontsize=16)
    
    # Collect all matches
    all_matches = []
    for result in all_results:
        for match in result['matches']:
            if match['matched_spike']:
                all_matches.append({
                    'time_diff': match['matched_spike']['time_diff'],
                    'n_wells': match['matched_spike']['n_wells'],
                    'mean_height': match['matched_spike']['mean_height'],
                    'well_fraction': match['matched_spike']['well_fraction']
                })
    
    if all_matches:
        match_df = pd.DataFrame(all_matches)
        
        # 1. Time difference distribution
        ax = axes[0, 0]
        ax.hist(match_df['time_diff'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect match')
        ax.axvline(match_df['time_diff'].mean(), color='orange', linestyle='-', 
                   linewidth=2, label=f'Mean: {match_df["time_diff"].mean():.1f}h')
        ax.set_xlabel('Actual - Recorded Time (hours)')
        ax.set_ylabel('Count')
        ax.set_title('Timing Difference Distribution')
        ax.legend()
        
        # 2. Spike characteristics of matched events
        ax = axes[0, 1]
        ax.scatter(match_df['well_fraction'] * 100, match_df['mean_height'], 
                  s=match_df['n_wells'], alpha=0.6)
        ax.set_xlabel('Well Coverage (%)')
        ax.set_ylabel('Mean Spike Height (% O₂)')
        ax.set_title('Characteristics of Matched Media Changes')
        ax.grid(True, alpha=0.3)
        
        # 3. Summary statistics
        ax = axes[1, 0]
        ax.axis('off')
        
        # Count unmatched events
        total_recorded = sum(len(r['recorded_times']) for r in all_results)
        total_matched = len(all_matches)
        total_detected = sum(len(r['synchronized_spikes']) for r in all_results)
        
        stats_text = f"""
        EVENT MATCHING SUMMARY
        
        Recorded events analyzed: {total_recorded}
        Successfully matched: {total_matched} ({total_matched/total_recorded*100:.0f}%)
        Unmatched recorded events: {total_recorded - total_matched}
        
        Total spike clusters detected: {total_detected}
        Potential untagged events: {total_detected - total_matched}
        
        TIMING DIFFERENCES (Actual - Recorded):
        Mean: {match_df['time_diff'].mean():.1f} hours
        Std: {match_df['time_diff'].std():.1f} hours
        Range: [{match_df['time_diff'].min():.1f}, {match_df['time_diff'].max():.1f}] hours
        
        MATCHED SPIKE CHARACTERISTICS:
        Mean height: {match_df['mean_height'].mean():.1f} ± {match_df['mean_height'].std():.1f} % O₂
        Mean well coverage: {match_df['well_fraction'].mean()*100:.0f}%
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 4. Optimal thresholds based on matched events
        ax = axes[1, 1]
        ax.axis('off')
        
        min_height = match_df['mean_height'].quantile(0.05)
        min_coverage = match_df['well_fraction'].quantile(0.05)
        
        threshold_text = f"""
        RECOMMENDED DETECTION THRESHOLDS
        (Based on successfully matched events)
        
        To capture 95% of true media changes:
        • Min spike height: {min_height:.1f} % O₂
        • Min well coverage: {min_coverage*100:.0f}%
        • Time window: ±10 hours
        
        To capture 80% of true media changes:
        • Min spike height: {match_df['mean_height'].quantile(0.20):.1f} % O₂
        • Min well coverage: {match_df['well_fraction'].quantile(0.20)*100:.0f}%
        
        Median characteristics:
        • Spike height: {match_df['mean_height'].median():.1f} % O₂
        • Well coverage: {match_df['well_fraction'].median()*100:.0f}%
        """
        
        ax.text(0.05, 0.95, threshold_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'event_timing_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run analysis accounting for ±10 hour timing uncertainty."""
    print("=== Media Change Event Analysis with Timing Uncertainty ===")
    print("Searching for actual spikes within ±10 hours of recorded events...\n")
    
    # Get plates with recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    events_df = pd.read_parquet(event_path)
    media_changes = events_df[events_df['title'] == 'Medium Change']
    plates_with_events = media_changes['plate_id'].unique()
    
    # Analyze each plate
    all_results = []
    for plate_id in plates_with_events[:3]:  # Analyze first 3 plates
        result = analyze_plate_events(plate_id)
        if result:
            all_results.append(result)
    
    if all_results:
        # Create visualizations
        create_event_alignment_visualization(all_results)
        create_summary_statistics(all_results)
        
        print(f"\nVisualizations saved to: {fig_dir}")
        print("  - event_timing_alignment.png")
        print("  - event_timing_summary.png")
        
        # Print summary
        print("\n=== SUMMARY ===")
        for result in all_results:
            print(f"\nPlate {result['plate_id']}:")
            matched = sum(1 for m in result['matches'] if m['matched_spike'])
            print(f"  Recorded events: {len(result['recorded_times'])}")
            print(f"  Matched to spikes: {matched}")
            print(f"  Total spike clusters: {len(result['synchronized_spikes'])}")
            
            # Show timing differences
            if matched > 0:
                diffs = [m['matched_spike']['time_diff'] 
                        for m in result['matches'] if m['matched_spike']]
                print(f"  Timing differences: {[f'{d:+.1f}h' for d in diffs]}")

if __name__ == "__main__":
    main()