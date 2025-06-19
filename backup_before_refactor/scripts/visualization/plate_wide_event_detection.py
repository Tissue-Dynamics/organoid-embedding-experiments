#!/usr/bin/env python3
"""
Analyze plate-wide spike patterns to identify true media change events.
Media changes should affect all wells within ~1 hour of each other.
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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

def load_plate_data_with_spikes(plate_id, detector=None):
    """Load all wells for a plate and detect spikes."""
    if detector is None:
        detector = MediaChangeDetector(variance_threshold=1.5, peak_prominence=0.3)
    
    database_url = os.getenv('DATABASE_URL')
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # Load all wells for this plate
    query = f"""
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        well_number,
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
        return None, None
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    plate_start = data['timestamp'].min()
    data['elapsed_hours'] = (data['timestamp'] - plate_start).dt.total_seconds() / 3600
    
    # Detect spikes for each well
    spike_times_by_well = {}
    wells = data['well_id'].unique()
    
    for well_id in wells:
        well_data = data[data['well_id'] == well_id].copy()
        
        if len(well_data) > 50:
            spikes = detector.detect_media_changes_from_variance(well_data)
            spike_times_by_well[well_id] = [s['detected_time_hours'] for s in spikes]
        else:
            spike_times_by_well[well_id] = []
    
    return data, spike_times_by_well

def cluster_spike_times(spike_times_by_well, time_window=1.0):
    """Cluster spike times across wells to find plate-wide events."""
    # Collect all spike times with well info
    all_spikes = []
    for well_id, times in spike_times_by_well.items():
        for t in times:
            all_spikes.append({'well_id': well_id, 'time': t})
    
    if len(all_spikes) == 0:
        return []
    
    spike_df = pd.DataFrame(all_spikes)
    spike_times = spike_df['time'].values
    
    # Cluster spike times
    if len(spike_times) > 1:
        # Use hierarchical clustering
        spike_times_2d = spike_times.reshape(-1, 1)
        linkage_matrix = linkage(spike_times_2d, method='single')
        clusters = fcluster(linkage_matrix, time_window, criterion='distance')
        spike_df['cluster'] = clusters
        
        # Find plate-wide events (clusters with spikes from multiple wells)
        cluster_stats = []
        for cluster_id in np.unique(clusters):
            cluster_data = spike_df[spike_df['cluster'] == cluster_id]
            n_wells = cluster_data['well_id'].nunique()
            
            if n_wells >= 3:  # At least 3 wells affected
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'mean_time': cluster_data['time'].mean(),
                    'std_time': cluster_data['time'].std(),
                    'n_wells': n_wells,
                    'well_coverage': n_wells / len(spike_times_by_well),
                    'wells': cluster_data['well_id'].unique().tolist()
                })
        
        return cluster_stats
    else:
        return []

def visualize_plate_wide_events(plate_id, n_sample_wells=20):
    """Create visualization of plate-wide spike detection."""
    print(f"\nAnalyzing plate {plate_id}...")
    
    # Load data and detect spikes
    detector = MediaChangeDetector(variance_threshold=1.5, peak_prominence=0.3)
    data, spike_times_by_well = load_plate_data_with_spikes(plate_id, detector)
    
    if data is None:
        print(f"No data found for plate {plate_id}")
        return None
    
    # Get recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    recorded_events = []
    if event_path.exists():
        events_df = pd.read_parquet(event_path)
        events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
        plate_events = events_df[
            (events_df['plate_id'] == plate_id) & 
            (events_df['title'] == 'Medium Change')
        ]
        
        if len(plate_events) > 0:
            plate_start = data['timestamp'].min()
            for _, event in plate_events.iterrows():
                event_time = (event['occurred_at'] - plate_start).total_seconds() / 3600
                if 0 <= event_time <= data['elapsed_hours'].max():
                    recorded_events.append(event_time)
    
    # Cluster spike times to find plate-wide events
    plate_wide_events = cluster_spike_times(spike_times_by_well, time_window=1.0)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    fig.suptitle(f'Plate-Wide Event Detection: {plate_id}', fontsize=16)
    
    # Sample wells for visualization
    wells = sorted(data['well_id'].unique())
    if len(wells) > n_sample_wells:
        # Sample evenly across the plate
        step = len(wells) // n_sample_wells
        sampled_wells = wells[::step][:n_sample_wells]
    else:
        sampled_wells = wells
    
    # Plot 1: Time series with spike markers
    for i, well_id in enumerate(sampled_wells):
        well_data = data[data['well_id'] == well_id]
        
        # Plot time series
        ax1.plot(well_data['elapsed_hours'], 
                well_data['o2_percent'] + i*5,  # Offset for visibility
                'b-', alpha=0.5, linewidth=0.8)
        
        # Mark detected spikes
        if well_id in spike_times_by_well:
            for spike_time in spike_times_by_well[well_id]:
                ax1.scatter(spike_time, 
                           well_data['o2_percent'].mean() + i*5,
                           color='red', s=30, alpha=0.7, zorder=5)
    
    # Add plate-wide event lines
    for event in plate_wide_events:
        ax1.axvline(event['mean_time'], color='green', linestyle='-', 
                   alpha=0.7, linewidth=2,
                   label=f"Detected ({event['n_wells']} wells)" if event == plate_wide_events[0] else '')
    
    # Add recorded event lines
    for i, event_time in enumerate(recorded_events):
        ax1.axvline(event_time, color='blue', linestyle='--', 
                   alpha=0.7, linewidth=2,
                   label='Recorded' if i == 0 else '')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('O₂ (%) - Wells stacked')
    ax1.set_title(f'Time Series for {len(sampled_wells)} Sample Wells (of {len(wells)} total)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spike timing heatmap
    # Create matrix of spike times
    max_time = data['elapsed_hours'].max()
    time_bins = np.arange(0, max_time + 1, 1)  # 1-hour bins
    spike_matrix = np.zeros((len(wells), len(time_bins)-1))
    
    for i, well_id in enumerate(wells):
        if well_id in spike_times_by_well:
            for spike_time in spike_times_by_well[well_id]:
                bin_idx = np.digitize(spike_time, time_bins) - 1
                if 0 <= bin_idx < len(time_bins)-1:
                    spike_matrix[i, bin_idx] = 1
    
    # Plot heatmap
    im = ax2.imshow(spike_matrix, aspect='auto', cmap='Reds', 
                    extent=[0, max_time, len(wells), 0])
    
    # Add plate-wide event lines
    for event in plate_wide_events:
        ax2.axvline(event['mean_time'], color='green', linestyle='-', 
                   alpha=0.7, linewidth=2)
    
    # Add recorded event lines
    for event_time in recorded_events:
        ax2.axvline(event_time, color='blue', linestyle='--', 
                   alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Well Index')
    ax2.set_title('Spike Detection Heatmap (red = spike detected)')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = fig_dir / f'plate_wide_events_{plate_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plate_wide_events, recorded_events

def analyze_multiple_plates():
    """Analyze multiple plates to compare detected vs recorded events."""
    # Load spike data to get plates
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    plates = spike_df['plate_id'].unique()
    
    all_results = []
    
    for plate_id in plates:
        plate_wide_events, recorded_events = visualize_plate_wide_events(plate_id)
        
        if plate_wide_events is not None:
            all_results.append({
                'plate_id': plate_id,
                'n_recorded': len(recorded_events),
                'n_detected_platewide': len(plate_wide_events),
                'detected_events': plate_wide_events,
                'recorded_events': recorded_events
            })
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Plate-Wide Media Change Event Analysis', fontsize=16)
    
    # Summary statistics
    ax = axes[0, 0]
    ax.axis('off')
    
    total_recorded = sum(r['n_recorded'] for r in all_results)
    total_detected = sum(r['n_detected_platewide'] for r in all_results)
    
    summary_text = f"""
    PLATE-WIDE EVENT DETECTION SUMMARY
    
    Plates Analyzed: {len(all_results)}
    
    Total Recorded Events: {total_recorded}
    Total Detected Plate-Wide Events: {total_detected}
    
    Detection Criteria:
    • Spikes in ≥3 wells within 1 hour
    • Variance threshold: 1.5x baseline
    • Peak prominence: 0.3x baseline
    
    Key Finding:
    • Detected {total_detected - total_recorded} potential untagged
      media changes affecting multiple wells
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Event comparison by plate
    ax = axes[0, 1]
    plate_names = [f"Plate {i+1}" for i in range(len(all_results))]
    x = np.arange(len(all_results))
    width = 0.35
    
    recorded_counts = [r['n_recorded'] for r in all_results]
    detected_counts = [r['n_detected_platewide'] for r in all_results]
    
    ax.bar(x - width/2, recorded_counts, width, label='Recorded', alpha=0.8)
    ax.bar(x + width/2, detected_counts, width, label='Detected', alpha=0.8)
    
    ax.set_ylabel('Number of Events')
    ax.set_title('Recorded vs Detected Events by Plate')
    ax.set_xticks(x)
    ax.set_xticklabels(plate_names)
    ax.legend()
    
    # Event timing comparison
    ax = axes[1, 0]
    for i, result in enumerate(all_results):
        y_offset = i * 0.5
        
        # Plot recorded events
        for event_time in result['recorded_events']:
            ax.scatter(event_time, y_offset, color='blue', marker='o', s=100)
        
        # Plot detected events
        for event in result['detected_events']:
            ax.scatter(event['mean_time'], y_offset + 0.2, 
                      color='green', marker='^', s=100,
                      alpha=event['well_coverage'])  # Alpha based on coverage
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Plate')
    ax.set_title('Event Timing Comparison')
    ax.set_yticks([i*0.5 + 0.1 for i in range(len(all_results))])
    ax.set_yticklabels(plate_names)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    ax.scatter([], [], color='blue', marker='o', s=100, label='Recorded')
    ax.scatter([], [], color='green', marker='^', s=100, label='Detected')
    ax.legend()
    
    # Well coverage statistics
    ax = axes[1, 1]
    all_coverages = []
    for result in all_results:
        for event in result['detected_events']:
            all_coverages.append(event['well_coverage'] * 100)
    
    if all_coverages:
        ax.hist(all_coverages, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_coverages), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_coverages):.0f}%')
        ax.set_xlabel('Well Coverage (%)')
        ax.set_ylabel('Number of Events')
        ax.set_title('Distribution of Well Coverage for Detected Events')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'plate_wide_event_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed results
    print("\n=== DETAILED RESULTS ===")
    for result in all_results:
        print(f"\nPlate {result['plate_id']}:")
        print(f"  Recorded events: {result['n_recorded']}")
        print(f"  Detected plate-wide events: {result['n_detected_platewide']}")
        
        if result['detected_events']:
            print("  Detected events:")
            for event in result['detected_events']:
                print(f"    - {event['mean_time']:.1f}h: {event['n_wells']} wells ({event['well_coverage']*100:.0f}% coverage)")

def main():
    """Run plate-wide event detection analysis."""
    print("=== Plate-Wide Media Change Event Detection ===")
    print("Looking for events that affect multiple wells within 1 hour...")
    
    analyze_multiple_plates()
    
    print(f"\nVisualizations saved to: {fig_dir}")
    print("  - plate_wide_events_*.png (individual plates)")
    print("  - plate_wide_event_summary.png (summary)")

if __name__ == "__main__":
    main()