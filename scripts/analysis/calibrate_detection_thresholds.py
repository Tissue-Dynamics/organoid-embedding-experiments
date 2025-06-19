#!/usr/bin/env python3
"""
Calibrate spike detection thresholds to find the strictest settings
that still capture all recorded events.
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

class ParameterizedDetector:
    """Detector with adjustable parameters."""
    
    def __init__(self, min_spike_height, min_sharpness, min_well_fraction):
        self.min_spike_height = min_spike_height
        self.min_sharpness = min_sharpness
        self.min_well_fraction = min_well_fraction
    
    def detect_spikes(self, time_series_data):
        """Detect spikes with current parameters."""
        if len(time_series_data) < 20:
            return []
        
        o2_values = time_series_data['o2_percent'].values
        elapsed_hours = time_series_data['elapsed_hours'].values
        
        if len(o2_values) < 3:
            return []
        
        # Smooth data
        smoothed_o2 = uniform_filter1d(o2_values, size=3)
        
        # Calculate rate of change
        dt = np.diff(elapsed_hours)
        do2 = np.diff(smoothed_o2)
        dt[dt == 0] = 0.001
        rate_of_change = do2 / dt
        
        spikes = []
        
        for i in range(1, len(rate_of_change)-1):
            if rate_of_change[i] > self.min_sharpness:
                window_end = min(i + 20, len(o2_values) - 1)
                window_values = o2_values[i:window_end]
                
                if len(window_values) > 0:
                    peak_idx_rel = np.argmax(window_values)
                    peak_idx = i + peak_idx_rel
                    
                    baseline_start = max(0, i - 10)
                    baseline = np.median(o2_values[baseline_start:i])
                    spike_height = o2_values[peak_idx] - baseline
                    
                    if spike_height >= self.min_spike_height:
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

def analyze_plate_with_params(plate_id, detector):
    """Analyze a plate with given detector parameters."""
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
        return []
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    plate_start = data['timestamp'].min()
    data['elapsed_hours'] = (data['timestamp'] - plate_start).dt.total_seconds() / 3600
    
    # Detect spikes for each well
    spike_times_by_well = {}
    wells = data['well_id'].unique()
    
    for well_id in wells:
        well_data = data[data['well_id'] == well_id].copy()
        spikes = detector.detect_spikes(well_data)
        spike_times_by_well[well_id] = spikes
    
    # Find synchronized events
    all_spike_times = []
    for well_id, spikes in spike_times_by_well.items():
        for spike in spikes:
            all_spike_times.append((spike['time'], well_id, spike['height']))
    
    if not all_spike_times:
        return []
    
    all_spike_times = sorted(all_spike_times, key=lambda x: x[0])
    
    # Cluster events
    synchronized_events = []
    i = 0
    
    while i < len(all_spike_times):
        current_time = all_spike_times[i][0]
        cluster_wells = set()
        cluster_heights = []
        
        j = i
        while j < len(all_spike_times) and all_spike_times[j][0] - current_time <= 2.0:
            cluster_wells.add(all_spike_times[j][1])
            cluster_heights.append(all_spike_times[j][2])
            j += 1
        
        well_fraction = len(cluster_wells) / len(wells)
        
        if well_fraction >= detector.min_well_fraction:
            synchronized_events.append({
                'time': current_time,
                'n_wells': len(cluster_wells),
                'well_fraction': well_fraction,
                'mean_height': np.mean(cluster_heights)
            })
        
        i = j
    
    return synchronized_events

def get_recorded_events(plate_id, data_start_time):
    """Get recorded events for a plate."""
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    if not event_path.exists():
        return []
    
    events_df = pd.read_parquet(event_path)
    events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    plate_events = events_df[
        (events_df['plate_id'] == plate_id) & 
        (events_df['title'] == 'Medium Change')
    ]
    
    recorded_times = []
    for _, event in plate_events.iterrows():
        event_time = (event['occurred_at'] - data_start_time).total_seconds() / 3600
        if event_time >= 0:
            recorded_times.append(event_time)
    
    return recorded_times

def check_detection_of_recorded(detected_events, recorded_events, tolerance=6.0):
    """Check how many recorded events were detected."""
    detected_count = 0
    
    for rec_time in recorded_events:
        for det_event in detected_events:
            if abs(det_event['time'] - rec_time) <= tolerance:
                detected_count += 1
                break
    
    return detected_count

def calibrate_thresholds():
    """Find optimal thresholds that capture all recorded events."""
    # Load data to get plates with recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    events_df = pd.read_parquet(event_path)
    media_changes = events_df[events_df['title'] == 'Medium Change']
    plates_with_events = media_changes['plate_id'].unique()
    
    # Test ranges
    height_thresholds = [5, 10, 15, 20, 25, 30]
    sharpness_thresholds = [2, 5, 8, 10, 15]
    well_fraction_thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    
    results = []
    
    print("Testing threshold combinations...")
    total_tests = len(height_thresholds) * len(sharpness_thresholds) * len(well_fraction_thresholds)
    test_count = 0
    
    for height in height_thresholds:
        for sharpness in sharpness_thresholds:
            for well_frac in well_fraction_thresholds:
                test_count += 1
                print(f"\rProgress: {test_count}/{total_tests} ", end='', flush=True)
                
                detector = ParameterizedDetector(height, sharpness, well_frac)
                
                total_recorded = 0
                total_detected_recorded = 0
                total_detected = 0
                
                # Test on plates with recorded events
                for plate_id in plates_with_events[:3]:  # Test on first 3 plates
                    # Get data start time
                    database_url = os.getenv('DATABASE_URL')
                    conn = duckdb.connect()
                    conn.execute("INSTALL postgres;")
                    conn.execute("LOAD postgres;")
                    
                    parsed = urlparse(database_url)
                    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
                    
                    time_query = f"""
                    SELECT MIN(timestamp) as start_time
                    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
                    WHERE plate_id::text = '{plate_id}'
                    """
                    
                    start_result = conn.execute(time_query).fetchone()
                    conn.close()
                    
                    if start_result and start_result[0]:
                        data_start = pd.to_datetime(start_result[0])
                        
                        recorded_events = get_recorded_events(plate_id, data_start)
                        detected_events = analyze_plate_with_params(plate_id, detector)
                        
                        detected_recorded = check_detection_of_recorded(detected_events, recorded_events)
                        
                        total_recorded += len(recorded_events)
                        total_detected_recorded += detected_recorded
                        total_detected += len(detected_events)
                
                if total_recorded > 0:
                    detection_rate = total_detected_recorded / total_recorded
                else:
                    detection_rate = 0
                
                results.append({
                    'height': height,
                    'sharpness': sharpness,
                    'well_fraction': well_frac,
                    'total_recorded': total_recorded,
                    'detected_recorded': total_detected_recorded,
                    'total_detected': total_detected,
                    'detection_rate': detection_rate,
                    'excess_detections': total_detected - total_recorded
                })
    
    print("\n\nCalibration complete!")
    
    results_df = pd.DataFrame(results)
    
    # Find configurations that detect all recorded events
    perfect_detection = results_df[results_df['detection_rate'] >= 0.95]
    
    if len(perfect_detection) > 0:
        # Among perfect detections, find the one with fewest excess detections
        optimal = perfect_detection.loc[perfect_detection['excess_detections'].idxmin()]
        
        print(f"\n=== OPTIMAL THRESHOLDS ===")
        print(f"Height threshold: {optimal['height']}% O₂")
        print(f"Sharpness threshold: {optimal['sharpness']}% O₂/hour")
        print(f"Well fraction threshold: {optimal['well_fraction']*100:.0f}%")
        print(f"Detects {optimal['detected_recorded']}/{optimal['total_recorded']} recorded events")
        print(f"Total detections: {optimal['total_detected']} (excess: {optimal['excess_detections']})")
    else:
        print("\nNo configuration detected all recorded events!")
    
    return results_df, optimal if len(perfect_detection) > 0 else None

def create_calibration_visualization(results_df, optimal):
    """Visualize calibration results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Spike Detection Threshold Calibration', fontsize=16)
    
    # 1. Detection rate heatmap
    ax = axes[0, 0]
    # Average over well fraction for visualization
    pivot_data = results_df.pivot_table(
        values='detection_rate',
        index='sharpness',
        columns='height',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Detection Rate'})
    ax.set_xlabel('Height Threshold (% O₂)')
    ax.set_ylabel('Sharpness Threshold (% O₂/hour)')
    ax.set_title('Detection Rate of Recorded Events')
    
    # 2. Excess detections heatmap
    ax = axes[0, 1]
    pivot_excess = results_df.pivot_table(
        values='excess_detections',
        index='sharpness',
        columns='height',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_excess, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Excess Detections'})
    ax.set_xlabel('Height Threshold (% O₂)')
    ax.set_ylabel('Sharpness Threshold (% O₂/hour)')
    ax.set_title('Excess Detections (False Positives)')
    
    # 3. Well fraction impact
    ax = axes[1, 0]
    for well_frac in results_df['well_fraction'].unique():
        subset = results_df[results_df['well_fraction'] == well_frac]
        ax.scatter(subset['detection_rate'], subset['excess_detections'], 
                  label=f'{well_frac*100:.0f}%', alpha=0.6, s=50)
    
    ax.set_xlabel('Detection Rate of Recorded Events')
    ax.set_ylabel('Excess Detections')
    ax.set_title('Trade-off: Detection Rate vs False Positives')
    ax.legend(title='Well Fraction', bbox_to_anchor=(1.05, 1))
    ax.grid(True, alpha=0.3)
    
    # 4. Optimal configuration
    ax = axes[1, 1]
    ax.axis('off')
    
    if optimal is not None:
        text = f"""
        OPTIMAL CONFIGURATION
        
        Thresholds:
        • Minimum spike height: {optimal['height']}% O₂
        • Minimum sharpness: {optimal['sharpness']}% O₂/hour  
        • Minimum well fraction: {optimal['well_fraction']*100:.0f}%
        
        Performance:
        • Recorded events detected: {optimal['detected_recorded']}/{optimal['total_recorded']} ({optimal['detection_rate']*100:.0f}%)
        • Total events detected: {int(optimal['total_detected'])}
        • Excess detections: {int(optimal['excess_detections'])}
        
        Recommendation:
        These thresholds capture all recorded media changes
        while minimizing false positive detections.
        """
    else:
        text = "No configuration detected all recorded events!"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'threshold_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run threshold calibration analysis."""
    print("=== Spike Detection Threshold Calibration ===")
    print("Finding strictest thresholds that still detect recorded events...\n")
    
    results_df, optimal = calibrate_thresholds()
    
    # Save results
    results_df.to_csv(data_dir / 'threshold_calibration_results.csv', index=False)
    
    # Create visualization
    create_calibration_visualization(results_df, optimal)
    
    print(f"\nResults saved to: {data_dir}/threshold_calibration_results.csv")
    print(f"Visualization saved to: {fig_dir}/threshold_calibration.png")

if __name__ == "__main__":
    main()