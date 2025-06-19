#!/usr/bin/env python3
"""
Comprehensive spike detection analysis with multiple examples.
Focus on detecting ALL spikes including the initial drug dosing.
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

def load_plate_data_with_wells(plate_id, n_wells=10):
    """Load data for specific number of wells from a plate."""
    database_url = os.getenv('DATABASE_URL')
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # First get well list
    well_query = f"""
    SELECT DISTINCT well_number
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND is_excluded = false
    ORDER BY well_number
    LIMIT {n_wells}
    """
    
    wells = conn.execute(well_query).fetchdf()
    
    if len(wells) == 0:
        conn.close()
        return None
    
    well_list = "', '".join(wells['well_number'].astype(str))
    
    # Load data for selected wells
    query = f"""
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        well_number,
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND well_number IN ('{well_list}')
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
    
    return data, plate_start

def detect_all_spikes_comprehensive(well_data, min_height=5.0, min_prominence=3.0):
    """Comprehensive spike detection with lower thresholds."""
    if len(well_data) < 20:
        return []
    
    o2_values = well_data['o2_percent'].values
    elapsed_hours = well_data['elapsed_hours'].values
    
    # Light smoothing to reduce noise
    smoothed_o2 = uniform_filter1d(o2_values, size=3)
    
    # Find peaks with relaxed criteria
    peaks, properties = find_peaks(
        smoothed_o2,
        height=min_height,  # Lower threshold
        prominence=min_prominence,  # Lower prominence
        distance=10  # At least 10 points between peaks
    )
    
    spikes = []
    for i, peak_idx in enumerate(peaks):
        # Calculate baseline before peak
        baseline_start = max(0, peak_idx - 20)
        baseline_end = max(0, peak_idx - 5)
        
        if baseline_end > baseline_start:
            baseline = np.median(o2_values[baseline_start:baseline_end])
        else:
            baseline = o2_values[max(0, peak_idx - 1)]
        
        spike_height = o2_values[peak_idx] - baseline
        
        # Also check for sharp increases (rate of change)
        if peak_idx > 0:
            rate = (o2_values[peak_idx] - o2_values[peak_idx - 1]) / \
                   (elapsed_hours[peak_idx] - elapsed_hours[peak_idx - 1] + 0.001)
        else:
            rate = 0
        
        spikes.append({
            'time': elapsed_hours[peak_idx],
            'height': spike_height,
            'prominence': properties['prominences'][i],
            'peak_value': o2_values[peak_idx],
            'baseline': baseline,
            'rate': rate
        })
    
    # Sort by time
    spikes = sorted(spikes, key=lambda x: x['time'])
    
    return spikes

def analyze_multiple_plates(n_plates=5):
    """Analyze multiple plates to test detection robustness."""
    # Get plates from Step 1 data
    step1_df = pd.read_parquet(data_dir / "step1_quality_assessment_all_plates.parquet")
    plates = step1_df['plate_id'].unique()[:n_plates]
    
    # Get recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    events_df = None
    if event_path.exists():
        events_df = pd.read_parquet(event_path)
        events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    fig, axes = plt.subplots(n_plates, 2, figsize=(16, 4*n_plates))
    if n_plates == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Comprehensive Spike Detection Analysis - Multiple Plates', fontsize=16)
    
    all_spike_stats = []
    
    for plate_idx, plate_id in enumerate(plates):
        print(f"\nAnalyzing plate {plate_idx + 1}/{n_plates}: {plate_id}")
        
        # Load data
        result = load_plate_data_with_wells(plate_id, n_wells=20)
        if result is None:
            print(f"  No data found")
            continue
        
        data, plate_start = result
        wells = data['well_id'].unique()
        
        # Get recorded events for this plate
        recorded_events = []
        if events_df is not None:
            plate_events = events_df[
                (events_df['plate_id'] == plate_id) & 
                (events_df['title'] == 'Medium Change')
            ]
            for _, event in plate_events.iterrows():
                event_time = (event['occurred_at'] - plate_start).total_seconds() / 3600
                if 0 <= event_time <= data['elapsed_hours'].max():
                    recorded_events.append(event_time)
        
        # Detect spikes in all wells
        all_spikes = []
        spike_times_by_well = {}
        
        for well_id in wells:
            well_data = data[data['well_id'] == well_id]
            spikes = detect_all_spikes_comprehensive(well_data)
            spike_times_by_well[well_id] = spikes
            
            for spike in spikes:
                all_spikes.append({
                    'well_id': well_id,
                    'time': spike['time'],
                    'height': spike['height'],
                    'prominence': spike['prominence']
                })
        
        if not all_spikes:
            continue
        
        spike_df = pd.DataFrame(all_spikes)
        
        # Plot 1: Individual wells with spikes
        ax = axes[plate_idx, 0]
        
        # Sample 5 wells for detailed view
        sample_wells = wells[:5]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_wells)))
        
        for i, well_id in enumerate(sample_wells):
            well_data = data[data['well_id'] == well_id]
            well_spikes = spike_times_by_well[well_id]
            
            # Plot time series
            ax.plot(well_data['elapsed_hours'], 
                   well_data['o2_percent'] + i*10,
                   color=colors[i], alpha=0.6, linewidth=1)
            
            # Mark spikes
            for spike in well_spikes:
                ax.scatter(spike['time'], spike['peak_value'] + i*10,
                          color=colors[i], s=50, edgecolor='black', 
                          linewidth=1, zorder=5)
                
                # Annotate first few spikes
                if spike == well_spikes[0]:
                    ax.annotate(f"{spike['time']:.0f}h",
                               xy=(spike['time'], spike['peak_value'] + i*10),
                               xytext=(spike['time'] + 5, spike['peak_value'] + i*10 + 2),
                               fontsize=8)
        
        # Add recorded event lines
        for event_time in recorded_events:
            ax.axvline(event_time, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('O₂ (%) - Wells offset')
        ax.set_title(f'Plate {plate_idx + 1}: Individual Well Spikes')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Spike timing histogram
        ax = axes[plate_idx, 1]
        
        # Bin spike times
        time_bins = np.arange(0, spike_df['time'].max() + 10, 5)
        spike_counts, _ = np.histogram(spike_df['time'], bins=time_bins)
        
        # Plot histogram
        ax.bar(time_bins[:-1], spike_counts, width=5, align='edge', 
               alpha=0.7, edgecolor='black')
        
        # Mark recorded events
        for event_time in recorded_events:
            ax.axvline(event_time, color='red', linestyle='--', 
                      alpha=0.7, label='Recorded' if event_time == recorded_events[0] else '')
        
        # Find and mark synchronized spikes (>50% wells)
        threshold = len(wells) * 0.5
        for i, count in enumerate(spike_counts):
            if count > threshold:
                time_center = (time_bins[i] + time_bins[i+1]) / 2
                ax.axvline(time_center, color='green', linestyle='-', 
                          alpha=0.7, linewidth=2)
                ax.text(time_center, count + 1, f'{count}', 
                       ha='center', fontsize=8)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Number of Wells with Spike')
        ax.set_title(f'Spike Distribution (5h bins, n={len(wells)} wells)')
        ax.axhline(threshold, color='green', linestyle=':', alpha=0.5,
                  label=f'>50% wells ({threshold:.0f})')
        
        if recorded_events or i == 0:
            ax.legend()
        
        # Collect statistics
        # Find major spike clusters
        major_spikes = []
        for i in range(len(time_bins) - 1):
            if spike_counts[i] > threshold:
                time_window = (time_bins[i], time_bins[i+1])
                window_spikes = spike_df[
                    (spike_df['time'] >= time_window[0]) & 
                    (spike_df['time'] < time_window[1])
                ]
                major_spikes.append({
                    'time': window_spikes['time'].mean(),
                    'n_wells': len(window_spikes),
                    'mean_height': window_spikes['height'].mean()
                })
        
        all_spike_stats.append({
            'plate_id': plate_id,
            'n_wells': len(wells),
            'n_recorded_events': len(recorded_events),
            'total_spikes': len(spike_df),
            'major_spike_clusters': major_spikes,
            'first_spike_time': spike_df['time'].min() if len(spike_df) > 0 else None
        })
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'comprehensive_spike_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary figure
    create_detection_summary(all_spike_stats)
    
    return all_spike_stats

def create_detection_summary(spike_stats):
    """Create summary of detection across all plates."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Extract first spike times
    first_spikes = []
    major_spike_times = []
    
    for stat in spike_stats:
        if stat['first_spike_time'] is not None:
            first_spikes.append(stat['first_spike_time'])
        
        for spike in stat['major_spike_clusters']:
            major_spike_times.append(spike['time'])
    
    # Plot distribution of spike times
    all_times = sorted(major_spike_times)
    
    # Cluster similar times
    clustered_times = []
    if all_times:
        current_cluster = [all_times[0]]
        
        for time in all_times[1:]:
            if time - current_cluster[-1] < 10:  # Within 10 hours
                current_cluster.append(time)
            else:
                clustered_times.append({
                    'mean_time': np.mean(current_cluster),
                    'count': len(current_cluster),
                    'std': np.std(current_cluster) if len(current_cluster) > 1 else 0
                })
                current_cluster = [time]
        
        # Don't forget last cluster
        clustered_times.append({
            'mean_time': np.mean(current_cluster),
            'count': len(current_cluster),
            'std': np.std(current_cluster) if len(current_cluster) > 1 else 0
        })
    
    # Plot clustered spike times
    times = [c['mean_time'] for c in clustered_times]
    counts = [c['count'] for c in clustered_times]
    stds = [c['std'] for c in clustered_times]
    
    bars = ax.bar(range(len(times)), times, yerr=stds, capsize=5, 
                   color='skyblue', edgecolor='black', linewidth=1)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'n={count}', ha='center', va='bottom')
    
    # Highlight common event times
    ax.axhline(3, color='red', linestyle=':', alpha=0.5, label='Initial dosing (~3h)')
    ax.axhline(90, color='orange', linestyle=':', alpha=0.5, label='First media change (~90h)')
    ax.axhline(160, color='green', linestyle=':', alpha=0.5, label='Second media change (~160h)')
    ax.axhline(260, color='blue', linestyle=':', alpha=0.5, label='Third media change (~260h)')
    
    ax.set_xlabel('Event Number')
    ax.set_ylabel('Time Since Start (hours)')
    ax.set_title('Common Media Change Event Times Across Plates')
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f'Event {i+1}' for i in range(len(times))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary text
    summary_text = f"""
    Detection Summary:
    • Plates analyzed: {len(spike_stats)}
    • First spikes detected at: {np.mean(first_spikes):.1f} ± {np.std(first_spikes):.1f} hours
    • Major spike clusters found: {len(clustered_times)}
    • Common event times: {[f'{t["mean_time"]:.0f}h' for t in clustered_times[:5]]}
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'spike_detection_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run comprehensive spike detection analysis."""
    print("=== Comprehensive Spike Detection Analysis ===")
    print("Analyzing multiple plates with focus on ALL spikes including initial dosing...\n")
    
    spike_stats = analyze_multiple_plates(n_plates=5)
    
    print("\n=== SUMMARY ===")
    for stat in spike_stats:
        print(f"\nPlate: {stat['plate_id']}")
        print(f"  Wells analyzed: {stat['n_wells']}")
        print(f"  Total spikes detected: {stat['total_spikes']}")
        print(f"  Major spike clusters: {len(stat['major_spike_clusters'])}")
        if stat['first_spike_time'] is not None:
            print(f"  First spike at: {stat['first_spike_time']:.1f} hours")
        
        if stat['major_spike_clusters']:
            print("  Major events:")
            for i, event in enumerate(stat['major_spike_clusters']):
                print(f"    {i+1}. {event['time']:.1f}h: {event['n_wells']} wells, "
                      f"{event['mean_height']:.1f}% O₂")
    
    print(f"\nFigures saved to: {fig_dir}")
    print("  - comprehensive_spike_detection.png")
    print("  - spike_detection_summary.png")

if __name__ == "__main__":
    main()