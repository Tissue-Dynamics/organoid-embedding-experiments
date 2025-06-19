#!/usr/bin/env python3
"""
Spike detection analysis using only control wells.
Control wells should show clear media change effects without drug interference.
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

def load_control_wells_data(plate_id):
    """Load data specifically for control wells (DMSO, media-only, concentration=0)."""
    database_url = os.getenv('DATABASE_URL')
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # First, get well mapping to identify controls
    well_map_query = f"""
    SELECT 
        plate_id,
        well_number,
        drug,
        concentration
    FROM postgres_scan('{postgres_string}', 'public', 'well_map_data')
    WHERE plate_id::text = '{plate_id}'
    """
    
    well_map = conn.execute(well_map_query).fetchdf()
    
    if len(well_map) == 0:
        conn.close()
        return None
    
    # Identify control wells
    control_conditions = [
        well_map['drug'].str.lower().str.contains('dmso', na=False),
        well_map['drug'].str.lower().str.contains('control', na=False),
        well_map['drug'].str.lower().str.contains('media', na=False),
        well_map['concentration'] == 0,
        well_map['concentration'].isna()
    ]
    
    is_control = False
    for condition in control_conditions:
        is_control = is_control | condition
    
    control_wells = well_map[is_control]['well_number'].unique()
    
    if len(control_wells) == 0:
        print(f"  No control wells found for plate {plate_id}")
        conn.close()
        return None
    
    print(f"  Found {len(control_wells)} control wells")
    
    # Load time series data for control wells
    control_well_list = "', '".join(control_wells.astype(str))
    
    query = f"""
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        well_number,
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND well_number IN ('{control_well_list}')
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
    
    return data, plate_start, control_wells

def detect_spikes_in_controls(well_data, min_height=5.0):
    """Detect spikes in control wells with appropriate thresholds."""
    if len(well_data) < 20:
        return []
    
    o2_values = well_data['o2_percent'].values
    elapsed_hours = well_data['elapsed_hours'].values
    
    # Light smoothing
    smoothed_o2 = uniform_filter1d(o2_values, size=3)
    
    # Find peaks - controls should show clear media change spikes
    peaks, properties = find_peaks(
        smoothed_o2,
        height=np.percentile(smoothed_o2, 10) + min_height,  # Above baseline
        prominence=min_height,
        distance=15  # At least 15 points between peaks
    )
    
    spikes = []
    for i, peak_idx in enumerate(peaks):
        # Calculate baseline
        baseline_start = max(0, peak_idx - 30)
        baseline_end = max(0, peak_idx - 5)
        
        if baseline_end > baseline_start:
            baseline = np.median(o2_values[baseline_start:baseline_end])
        else:
            baseline = o2_values[max(0, peak_idx - 1)]
        
        spike_height = o2_values[peak_idx] - baseline
        
        # Calculate sharpness (rate of change)
        if peak_idx > 2:
            pre_values = o2_values[max(0, peak_idx-3):peak_idx]
            pre_times = elapsed_hours[max(0, peak_idx-3):peak_idx]
            if len(pre_values) > 1:
                rate = (o2_values[peak_idx] - pre_values[0]) / (elapsed_hours[peak_idx] - pre_times[0] + 0.001)
            else:
                rate = 0
        else:
            rate = 0
        
        spikes.append({
            'time': elapsed_hours[peak_idx],
            'height': spike_height,
            'prominence': properties['prominences'][i],
            'peak_value': o2_values[peak_idx],
            'baseline': baseline,
            'sharpness': rate
        })
    
    return sorted(spikes, key=lambda x: x['time'])

def analyze_control_wells_multiple_plates(n_plates=8):
    """Analyze control wells across multiple plates."""
    # Get plates from Step 1 data
    step1_df = pd.read_parquet(data_dir / "step1_quality_assessment_all_plates.parquet")
    plates = step1_df['plate_id'].unique()[:n_plates]
    
    # Get recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    events_df = None
    if event_path.exists():
        events_df = pd.read_parquet(event_path)
        events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    # Analyze each plate
    results = []
    
    for plate_id in plates:
        print(f"\nAnalyzing plate: {plate_id}")
        
        # Load control wells data
        result = load_control_wells_data(plate_id)
        if result is None:
            continue
        
        data, plate_start, control_wells = result
        
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
        
        # Detect spikes in each control well
        all_control_spikes = []
        spikes_by_well = {}
        
        for well_id in data['well_id'].unique():
            well_data = data[data['well_id'] == well_id]
            spikes = detect_spikes_in_controls(well_data)
            spikes_by_well[well_id] = spikes
            
            for spike in spikes:
                all_control_spikes.append({
                    'well_id': well_id,
                    'time': spike['time'],
                    'height': spike['height'],
                    'prominence': spike['prominence'],
                    'sharpness': spike['sharpness']
                })
        
        # Find synchronized events in controls
        if all_control_spikes:
            spike_df = pd.DataFrame(all_control_spikes)
            
            # Cluster spike times
            synchronized_events = []
            spike_times = sorted(spike_df['time'].unique())
            
            i = 0
            while i < len(spike_times):
                current_time = spike_times[i]
                
                # Find spikes within 2 hours
                cluster_mask = (spike_df['time'] >= current_time - 1) & \
                              (spike_df['time'] <= current_time + 1)
                cluster_spikes = spike_df[cluster_mask]
                
                n_wells = cluster_spikes['well_id'].nunique()
                
                if n_wells >= max(2, len(control_wells) * 0.4):  # At least 40% of controls
                    synchronized_events.append({
                        'time': cluster_spikes['time'].mean(),
                        'n_wells': n_wells,
                        'well_fraction': n_wells / len(control_wells),
                        'mean_height': cluster_spikes['height'].mean(),
                        'std_height': cluster_spikes['height'].std(),
                        'mean_sharpness': cluster_spikes['sharpness'].mean()
                    })
                    
                    # Skip past this cluster
                    while i < len(spike_times) and spike_times[i] <= current_time + 1:
                        i += 1
                else:
                    i += 1
        else:
            synchronized_events = []
        
        results.append({
            'plate_id': plate_id,
            'n_control_wells': len(control_wells),
            'data': data,
            'recorded_events': recorded_events,
            'synchronized_events': synchronized_events,
            'spikes_by_well': spikes_by_well,
            'plate_start': plate_start
        })
        
        print(f"  Control wells: {len(control_wells)}")
        print(f"  Synchronized events: {len(synchronized_events)}")
        print(f"  Recorded events: {len(recorded_events)}")
    
    return results

def create_control_wells_visualization(results):
    """Create comprehensive visualization of control wells analysis."""
    n_plates = len(results)
    fig, axes = plt.subplots(n_plates, 2, figsize=(16, 4*n_plates))
    if n_plates == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Media Change Detection in Control Wells Only', fontsize=16)
    
    for idx, result in enumerate(results):
        data = result['data']
        
        # Plot 1: Control well time series with detected events
        ax = axes[idx, 0]
        
        well_ids = data['well_id'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(well_ids)))
        
        for i, well_id in enumerate(well_ids):
            well_data = data[data['well_id'] == well_id]
            well_spikes = result['spikes_by_well'][well_id]
            
            # Plot time series
            ax.plot(well_data['elapsed_hours'], 
                   well_data['o2_percent'] + i*10,
                   color=colors[i], alpha=0.7, linewidth=1.5,
                   label=f"Control {well_id.split('_')[-1]}")
            
            # Mark detected spikes
            for spike in well_spikes:
                ax.scatter(spike['time'], spike['peak_value'] + i*10,
                          color=colors[i], s=80, edgecolor='black', 
                          linewidth=1, zorder=5)
        
        # Mark synchronized events
        for event in result['synchronized_events']:
            ax.axvline(event['time'], color='green', linestyle='-', 
                      alpha=0.8, linewidth=3,
                      label='Detected events' if event == result['synchronized_events'][0] else '')
        
        # Mark recorded events
        for i, event_time in enumerate(result['recorded_events']):
            ax.axvline(event_time, color='red', linestyle='--', 
                      alpha=0.7, linewidth=2,
                      label='Recorded events' if i == 0 else '')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('O₂ (%) - Wells offset')
        ax.set_title(f'Plate {idx+1}: Control Wells Time Series')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Event characteristics
        ax = axes[idx, 1]
        
        if result['synchronized_events']:
            events = result['synchronized_events']
            times = [e['time'] for e in events]
            heights = [e['mean_height'] for e in events]
            well_fractions = [e['well_fraction'] * 100 for e in events]
            
            # Scatter plot with size based on well coverage
            scatter = ax.scatter(times, heights, s=[w*5 for w in well_fractions], 
                               alpha=0.7, c=range(len(times)), cmap='viridis',
                               edgecolor='black', linewidth=1)
            
            # Add event labels
            for i, (time, height) in enumerate(zip(times, heights)):
                ax.annotate(f'E{i+1}', (time, height), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Mean Spike Height (% O₂)')
            ax.set_title(f'Detected Events (size = well coverage %)')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Event Number')
        else:
            ax.text(0.5, 0.5, 'No synchronized events detected', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12)
            ax.set_title('No Events Detected')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'control_wells_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_control_summary_analysis(results):
    """Create summary analysis of control wells across all plates."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Control Wells: Media Change Event Summary', fontsize=16)
    
    # Collect all synchronized events
    all_events = []
    for result in results:
        for event in result['synchronized_events']:
            all_events.append({
                'plate_id': result['plate_id'],
                'time': event['time'],
                'height': event['mean_height'],
                'well_fraction': event['well_fraction'],
                'n_wells': event['n_wells']
            })
    
    if not all_events:
        fig.text(0.5, 0.5, 'No synchronized events detected in control wells', 
                ha='center', va='center', fontsize=16)
        plt.savefig(fig_dir / 'control_wells_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    events_df = pd.DataFrame(all_events)
    
    # 1. Event timing distribution
    ax = axes[0, 0]
    ax.hist(events_df['time'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(events_df['time'].mean(), color='red', linestyle='--',
               label=f'Mean: {events_df["time"].mean():.1f}h')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Number of Events')
    ax.set_title('Distribution of Event Times in Controls')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Spike height distribution
    ax = axes[0, 1]
    ax.hist(events_df['height'], bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(events_df['height'].mean(), color='red', linestyle='--',
               label=f'Mean: {events_df["height"].mean():.1f}% O₂')
    ax.set_xlabel('Spike Height (% O₂)')
    ax.set_ylabel('Number of Events')
    ax.set_title('Distribution of Spike Heights in Controls')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Common event times across plates
    ax = axes[1, 0]
    
    # Cluster event times
    event_times = sorted(events_df['time'])
    clustered_times = []
    
    if event_times:
        current_cluster = [event_times[0]]
        
        for time in event_times[1:]:
            if time - current_cluster[-1] < 20:  # Within 20 hours
                current_cluster.append(time)
            else:
                clustered_times.append({
                    'mean_time': np.mean(current_cluster),
                    'count': len(current_cluster),
                    'std': np.std(current_cluster) if len(current_cluster) > 1 else 0
                })
                current_cluster = [time]
        
        clustered_times.append({
            'mean_time': np.mean(current_cluster),
            'count': len(current_cluster),
            'std': np.std(current_cluster) if len(current_cluster) > 1 else 0
        })
    
    if clustered_times:
        times = [c['mean_time'] for c in clustered_times]
        counts = [c['count'] for c in clustered_times]
        stds = [c['std'] for c in clustered_times]
        
        bars = ax.bar(range(len(times)), times, yerr=stds, capsize=5,
                     alpha=0.7, edgecolor='black')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'n={count}', ha='center', va='bottom')
        
        ax.set_xlabel('Event Cluster')
        ax.set_ylabel('Time (hours)')
        ax.set_title('Common Event Times Across Plates')
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([f'C{i+1}' for i in range(len(times))])
    
    # 4. Recommended thresholds
    ax = axes[1, 1]
    ax.axis('off')
    
    threshold_text = f"""
    CONTROL WELLS ANALYSIS SUMMARY
    
    Total plates analyzed: {len(results)}
    Total control events detected: {len(events_df)}
    
    EVENT CHARACTERISTICS:
    • Mean spike height: {events_df['height'].mean():.1f} ± {events_df['height'].std():.1f} % O₂
    • Mean event time: {events_df['time'].mean():.1f} ± {events_df['time'].std():.1f} hours
    • Mean well coverage: {events_df['well_fraction'].mean()*100:.0f}%
    
    RECOMMENDED THRESHOLDS:
    (Based on control wells only)
    
    • Min spike height: {events_df['height'].quantile(0.1):.1f} % O₂
    • Min well coverage: {events_df['well_fraction'].quantile(0.1)*100:.0f}%
    • Time clustering: ±10 hours
    
    Common event times:
    {[f'{c["mean_time"]:.0f}h' for c in clustered_times[:5]]}
    """
    
    ax.text(0.05, 0.95, threshold_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'control_wells_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run control wells analysis."""
    print("=== Media Change Detection Using Control Wells Only ===")
    print("Analyzing spikes in control wells to identify true media change events...\n")
    
    results = analyze_control_wells_multiple_plates(n_plates=8)
    
    if not results:
        print("No control wells data found!")
        return
    
    # Create visualizations
    create_control_wells_visualization(results)
    create_control_summary_analysis(results)
    
    print(f"\nVisualizations saved to: {fig_dir}")
    print("  - control_wells_analysis.png")
    print("  - control_wells_summary.png")
    
    # Print summary
    print("\n=== CONTROL WELLS SUMMARY ===")
    for result in results:
        print(f"\nPlate: {result['plate_id']}")
        print(f"  Control wells: {result['n_control_wells']}")
        print(f"  Synchronized events: {len(result['synchronized_events'])}")
        print(f"  Recorded events: {len(result['recorded_events'])}")
        
        if result['synchronized_events']:
            print("  Control events:")
            for i, event in enumerate(result['synchronized_events']):
                print(f"    {i+1}. {event['time']:.1f}h: {event['n_wells']} wells "
                      f"({event['well_fraction']*100:.0f}%), {event['mean_height']:.1f}% O₂")

if __name__ == "__main__":
    main()