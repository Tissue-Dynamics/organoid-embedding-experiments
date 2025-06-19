#!/usr/bin/env python3
"""
Strict plate-wide event detection with higher thresholds.
A real media change should cause a significant, sharp spike in O2.
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

class StrictMediaChangeDetector:
    """Stricter media change detection."""
    
    def __init__(self, min_spike_height=10.0, min_sharpness=5.0, window_hours=2.0):
        """
        Initialize with strict parameters.
        
        Args:
            min_spike_height: Minimum O2 change to consider (in O2%)
            min_sharpness: Minimum rate of change (O2%/hour)
            window_hours: Time window for spike detection
        """
        self.min_spike_height = min_spike_height
        self.min_sharpness = min_sharpness
        self.window_hours = window_hours
    
    def detect_spikes(self, time_series_data):
        """Detect only significant, sharp spikes."""
        if len(time_series_data) < 20:
            return []
        
        o2_values = time_series_data['o2_percent'].values
        elapsed_hours = time_series_data['elapsed_hours'].values
        
        # Calculate derivatives (rate of change)
        if len(o2_values) < 3:
            return []
        
        # Smooth data first to reduce noise
        from scipy.ndimage import uniform_filter1d
        smoothed_o2 = uniform_filter1d(o2_values, size=3)
        
        # Calculate rate of change
        dt = np.diff(elapsed_hours)
        do2 = np.diff(smoothed_o2)
        
        # Avoid division by zero
        dt[dt == 0] = 0.001
        rate_of_change = do2 / dt
        
        spikes = []
        
        # Look for sharp increases
        for i in range(1, len(rate_of_change)-1):
            # Check if this is a local maximum in rate of change
            if rate_of_change[i] > self.min_sharpness:
                # Look ahead to find the peak
                window_end = min(i + int(self.window_hours * 10), len(o2_values) - 1)
                
                # Find the actual peak value in the window
                window_values = o2_values[i:window_end]
                if len(window_values) > 0:
                    peak_idx_rel = np.argmax(window_values)
                    peak_idx = i + peak_idx_rel
                    
                    # Calculate spike height from baseline
                    baseline_start = max(0, i - 10)
                    baseline = np.median(o2_values[baseline_start:i])
                    spike_height = o2_values[peak_idx] - baseline
                    
                    if spike_height >= self.min_spike_height:
                        # Check if this is not too close to a previous spike
                        spike_time = elapsed_hours[i]
                        if not spikes or spike_time - spikes[-1]['time'] > 6.0:
                            spikes.append({
                                'time': spike_time,
                                'height': spike_height,
                                'sharpness': rate_of_change[i],
                                'baseline': baseline,
                                'peak_value': o2_values[peak_idx]
                            })
        
        return spikes

def load_and_analyze_plate_strict(plate_id):
    """Load plate data and detect spikes with strict criteria."""
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
    
    # Detect spikes for each well with strict criteria
    detector = StrictMediaChangeDetector(
        min_spike_height=10.0,  # At least 10% O2 change
        min_sharpness=5.0,      # At least 5% per hour rate
        window_hours=2.0
    )
    
    spike_times_by_well = {}
    wells = data['well_id'].unique()
    
    for well_id in wells:
        well_data = data[data['well_id'] == well_id].copy()
        spikes = detector.detect_spikes(well_data)
        spike_times_by_well[well_id] = spikes
    
    return data, spike_times_by_well

def find_synchronized_events(spike_times_by_well, sync_window=2.0, min_wells=10):
    """Find events that occur synchronously across many wells."""
    # Collect all spike times
    all_spike_times = []
    for well_id, spikes in spike_times_by_well.items():
        for spike in spikes:
            all_spike_times.append(spike['time'])
    
    if not all_spike_times:
        return []
    
    all_spike_times = sorted(all_spike_times)
    
    # Find synchronized events
    synchronized_events = []
    i = 0
    
    while i < len(all_spike_times):
        current_time = all_spike_times[i]
        
        # Count wells with spikes near this time
        wells_affected = []
        spike_heights = []
        
        for well_id, spikes in spike_times_by_well.items():
            for spike in spikes:
                if abs(spike['time'] - current_time) <= sync_window:
                    wells_affected.append(well_id)
                    spike_heights.append(spike['height'])
                    break
        
        if len(wells_affected) >= min_wells:
            synchronized_events.append({
                'time': current_time,
                'n_wells': len(wells_affected),
                'well_fraction': len(wells_affected) / len(spike_times_by_well),
                'mean_height': np.mean(spike_heights),
                'std_height': np.std(spike_heights)
            })
            
            # Skip past this event cluster
            while i < len(all_spike_times) and all_spike_times[i] - current_time < sync_window:
                i += 1
        else:
            i += 1
    
    return synchronized_events

def create_strict_visualization(plate_id):
    """Create visualization with strict spike detection."""
    print(f"\nAnalyzing plate {plate_id} with STRICT criteria...")
    
    # Load data and detect spikes
    data, spike_times_by_well = load_and_analyze_plate_strict(plate_id)
    
    if data is None:
        return None
    
    # Find synchronized events (at least 10 wells affected)
    synchronized_events = find_synchronized_events(
        spike_times_by_well, 
        sync_window=2.0,
        min_wells=10
    )
    
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
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 1, 1])
    fig.suptitle(f'Strict Media Change Detection: {plate_id}', fontsize=16)
    
    # Sample wells for visualization
    wells = sorted(data['well_id'].unique())
    n_sample = min(20, len(wells))
    step = len(wells) // n_sample
    sampled_wells = wells[::step][:n_sample]
    
    # Plot 1: Time series with detected spikes
    ax = axes[0]
    for i, well_id in enumerate(sampled_wells):
        well_data = data[data['well_id'] == well_id]
        
        # Plot time series
        ax.plot(well_data['elapsed_hours'], 
                well_data['o2_percent'] + i*5,  # Offset for visibility
                'b-', alpha=0.3, linewidth=0.8)
        
        # Mark detected spikes
        if well_id in spike_times_by_well:
            for spike in spike_times_by_well[well_id]:
                ax.scatter(spike['time'], 
                          spike['peak_value'] + i*5,
                          color='red', s=50, alpha=0.8, zorder=5)
    
    # Add synchronized event lines
    for event in synchronized_events:
        ax.axvline(event['time'], color='green', linestyle='-', 
                   alpha=0.7, linewidth=2,
                   label=f"Synchronized ({event['n_wells']} wells)" if event == synchronized_events[0] else '')
    
    # Add recorded event lines
    for i, event_time in enumerate(recorded_events):
        ax.axvline(event_time, color='blue', linestyle='--', 
                   alpha=0.7, linewidth=2,
                   label='Recorded' if i == 0 else '')
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('O₂ (%) - Wells stacked')
    ax.set_title(f'Time Series with Strict Spike Detection (min height: 10%, min sharpness: 5%/hour)')
    if synchronized_events or recorded_events:
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Spike detection summary
    ax = axes[1]
    
    # Count spikes per time bin
    time_bins = np.arange(0, data['elapsed_hours'].max(), 6)  # 6-hour bins
    spike_counts = np.zeros(len(time_bins) - 1)
    
    for well_id, spikes in spike_times_by_well.items():
        for spike in spikes:
            bin_idx = np.searchsorted(time_bins, spike['time']) - 1
            if 0 <= bin_idx < len(spike_counts):
                spike_counts[bin_idx] += 1
    
    ax.bar(time_bins[:-1], spike_counts, width=6, align='edge', alpha=0.7)
    
    # Add event markers
    for event in synchronized_events:
        ax.axvline(event['time'], color='green', linestyle='-', alpha=0.7, linewidth=2)
    for event_time in recorded_events:
        ax.axvline(event_time, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Number of Wells with Spikes')
    ax.set_title('Spike Detection Frequency (6-hour bins)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Event comparison
    ax = axes[2]
    ax.axis('off')
    
    # Summary statistics
    total_wells = len(spike_times_by_well)
    wells_with_spikes = sum(1 for spikes in spike_times_by_well.values() if spikes)
    total_spikes = sum(len(spikes) for spikes in spike_times_by_well.values())
    
    summary_text = f"""
    STRICT DETECTION SUMMARY
    
    Detection Criteria:
    • Minimum spike height: 10% O₂
    • Minimum sharpness: 5% O₂/hour
    • Synchronization window: 2 hours
    • Minimum wells for event: 10
    
    Results:
    • Total wells: {total_wells}
    • Wells with spikes: {wells_with_spikes} ({wells_with_spikes/total_wells*100:.1f}%)
    • Total spikes detected: {total_spikes}
    • Synchronized events: {len(synchronized_events)}
    • Recorded events: {len(recorded_events)}
    
    Synchronized Events:
    """
    
    for i, event in enumerate(synchronized_events):
        summary_text += f"\n    {i+1}. {event['time']:.1f}h: {event['n_wells']} wells ({event['well_fraction']*100:.0f}%), "
        summary_text += f"height: {event['mean_height']:.1f}±{event['std_height']:.1f} O₂%"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = fig_dir / f'strict_detection_{plate_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return synchronized_events, recorded_events

def main():
    """Run strict event detection analysis."""
    print("=== STRICT Media Change Event Detection ===")
    print("Using strict criteria: min 10% O2 spike, min 5%/hour rate")
    
    # Load spike data to get plates
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    plates = spike_df['plate_id'].unique()
    
    all_results = []
    
    for plate_id in plates:
        sync_events, rec_events = create_strict_visualization(plate_id)
        
        if sync_events is not None:
            all_results.append({
                'plate_id': plate_id,
                'n_recorded': len(rec_events),
                'n_synchronized': len(sync_events),
                'synchronized_events': sync_events,
                'recorded_events': rec_events
            })
    
    # Print summary
    print("\n=== SUMMARY ===")
    for result in all_results:
        print(f"\nPlate {result['plate_id']}:")
        print(f"  Recorded events: {result['n_recorded']}")
        print(f"  Detected synchronized events: {result['n_synchronized']}")
        
        if result['synchronized_events']:
            print("  Synchronized events:")
            for event in result['synchronized_events']:
                print(f"    - {event['time']:.1f}h: {event['n_wells']} wells, {event['mean_height']:.1f} O₂% spike")
    
    print(f"\nFigures saved to: {fig_dir}")

if __name__ == "__main__":
    main()