#!/usr/bin/env python3
"""
Validate event detection by overlaying detected events on actual time series data.
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
import warnings
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# Setup paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "step3_validation"

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def load_time_series_data(plate_id, well_ids=None):
    """Load actual time series data for validation."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # Build well filter if specific wells requested
    well_filter = ""
    if well_ids is not None and len(well_ids) > 0:
        well_list = "', '".join(well_ids)
        well_filter = f"AND (plate_id::text || '_' || well_number::text) IN ('{well_list}')"
    
    query = f"""
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND is_excluded = false
    {well_filter}
    ORDER BY well_id, timestamp
    """
    
    data = conn.execute(query).fetchdf()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate elapsed hours
    plate_start = data['timestamp'].min()
    data['elapsed_hours'] = (data['timestamp'] - plate_start).dt.total_seconds() / 3600
    
    conn.close()
    return data

def load_event_data_for_plate(plate_id):
    """Load media change events for a specific plate."""
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    
    if not event_path.exists():
        print("Warning: Event data not found.")
        return pd.DataFrame()
    
    events_df = pd.read_parquet(event_path)
    events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    # Filter for this plate and media changes
    plate_events = events_df[
        (events_df['plate_id'] == plate_id) & 
        (events_df['title'] == 'Medium Change')
    ].copy()
    
    return plate_events.sort_values('occurred_at')

def create_time_series_with_events_overlay():
    """Create figure showing actual time series with detected events."""
    # Load spike features to get plates and wells
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    
    # Get unique plates
    plates = spike_df['plate_id'].unique()[:2]  # Use first 2 plates
    
    fig, axes = plt.subplots(len(plates), 2, figsize=(16, 6*len(plates)))
    if len(plates) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Time Series Data with Media Change Events Overlay', fontsize=16, y=0.995)
    
    for plate_idx, plate_id in enumerate(plates):
        print(f"\nProcessing plate {plate_idx + 1}/{len(plates)}: {plate_id}")
        
        # Get wells with spike data for this plate
        plate_spikes = spike_df[spike_df['plate_id'] == plate_id]
        wells_with_spikes = plate_spikes['well_id'].unique()[:3]  # First 3 wells
        
        if len(wells_with_spikes) == 0:
            continue
        
        # Load actual time series data
        ts_data = load_time_series_data(plate_id, wells_with_spikes)
        
        # Load event data
        event_data = load_event_data_for_plate(plate_id)
        
        # Convert event times to elapsed hours
        if len(event_data) > 0 and len(ts_data) > 0:
            plate_start = ts_data['timestamp'].min()
            event_data['elapsed_hours'] = (event_data['occurred_at'] - plate_start).dt.total_seconds() / 3600
        
        # Plot individual wells
        ax = axes[plate_idx, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(wells_with_spikes)))
        
        for well_idx, well_id in enumerate(wells_with_spikes):
            well_data = ts_data[ts_data['well_id'] == well_id]
            
            if len(well_data) > 0:
                ax.plot(well_data['elapsed_hours'], well_data['o2_percent'], 
                       label=f'Well {well_id.split("_")[-1]}', 
                       alpha=0.8, linewidth=1.5, color=colors[well_idx])
        
        # Add event lines
        if len(event_data) > 0:
            for _, event in event_data.iterrows():
                if 'elapsed_hours' in event and 0 <= event['elapsed_hours'] <= ts_data['elapsed_hours'].max():
                    ax.axvline(event['elapsed_hours'], color='red', linestyle='--', 
                              alpha=0.7, linewidth=2)
            
            # Add one event label
            if len(event_data) > 0 and 'elapsed_hours' in event_data.columns:
                first_event = event_data.iloc[0]
                if 0 <= first_event['elapsed_hours'] <= ts_data['elapsed_hours'].max():
                    ax.text(first_event['elapsed_hours'], ax.get_ylim()[1]*0.95, 
                           'Media\nChange', ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('O₂ (%)')
        ax.set_title(f'Plate {plate_idx + 1}: Individual Wells')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot with spike characterization overlay
        ax2 = axes[plate_idx, 1]
        
        # Choose one well to show in detail
        if len(wells_with_spikes) > 0:
            detail_well = wells_with_spikes[0]
            well_data = ts_data[ts_data['well_id'] == detail_well]
            well_spikes = plate_spikes[plate_spikes['well_id'] == detail_well]
            
            if len(well_data) > 0:
                # Plot time series
                ax2.plot(well_data['elapsed_hours'], well_data['o2_percent'], 
                        'b-', linewidth=2, label='O₂ Data')
                
                # Overlay spike characterizations
                for _, spike in well_spikes.iterrows():
                    event_time = spike['event_time_hours']
                    
                    # Mark event time
                    ax2.axvline(event_time, color='red', linestyle='--', 
                               alpha=0.7, linewidth=2)
                    
                    # Show pre-spike baseline
                    ax2.axhline(spike['pre_spike_mean'], 
                               xmin=(event_time-6)/well_data['elapsed_hours'].max(), 
                               xmax=event_time/well_data['elapsed_hours'].max(),
                               color='green', linestyle=':', linewidth=2, alpha=0.7)
                    
                    # Mark peak
                    peak_time = event_time + spike['peak_time_relative']
                    ax2.scatter(peak_time, spike['peak_absolute_value'], 
                               color='orange', s=100, zorder=5, 
                               label='Peak' if spike.name == 0 else '')
                    
                    # Show recovery time if available
                    if pd.notna(spike['recovery_time']):
                        recovery_time = event_time + spike['recovery_time']
                        ax2.scatter(recovery_time, spike['pre_spike_mean'], 
                                   color='purple', s=100, marker='v', zorder=5,
                                   label='Recovery' if spike.name == 0 else '')
                        
                        # Draw recovery period
                        ax2.fill_betweenx([well_data['o2_percent'].min(), 
                                          well_data['o2_percent'].max()],
                                         event_time, recovery_time,
                                         alpha=0.2, color='yellow', 
                                         label='Recovery Period' if spike.name == 0 else '')
                
                ax2.set_xlabel('Time (hours)')
                ax2.set_ylabel('O₂ (%)')
                ax2.set_title(f'Well {detail_well}: Spike Characterization')
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'time_series_with_events_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_spike_detection_validation():
    """Create detailed view of spike detection for individual events."""
    # Load spike features
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    
    # Select a few representative spikes
    sample_spikes = spike_df.sample(min(6, len(spike_df)), random_state=42)
    
    n_spikes = len(sample_spikes)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle('Spike Detection Validation: Detailed Views', fontsize=16)
    
    for idx, (_, spike) in enumerate(sample_spikes.iterrows()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        # Load time series data for this well
        ts_data = load_time_series_data(spike['plate_id'], [spike['well_id']])
        well_data = ts_data[ts_data['well_id'] == spike['well_id']]
        
        if len(well_data) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        
        # Define window around event
        event_time = spike['event_time_hours']
        window_start = max(0, event_time - 12)
        window_end = min(well_data['elapsed_hours'].max(), event_time + 24)
        
        # Filter to window
        window_mask = (well_data['elapsed_hours'] >= window_start) & \
                     (well_data['elapsed_hours'] <= window_end)
        window_data = well_data[window_mask]
        
        if len(window_data) > 0:
            # Plot time series
            ax.plot(window_data['elapsed_hours'], window_data['o2_percent'], 
                   'b-', linewidth=2, alpha=0.8)
            
            # Event line
            ax.axvline(event_time, color='red', linestyle='--', linewidth=2,
                      label='Media Change')
            
            # Pre-spike baseline
            ax.axhspan(spike['pre_spike_mean'] - spike['pre_spike_std'],
                      spike['pre_spike_mean'] + spike['pre_spike_std'],
                      xmin=0, xmax=(event_time-window_start)/(window_end-window_start),
                      alpha=0.3, color='green', label='Baseline ± SD')
            ax.axhline(spike['pre_spike_mean'], color='green', linestyle=':')
            
            # Peak
            peak_time = event_time + spike['peak_time_relative']
            ax.scatter(peak_time, spike['peak_absolute_value'], 
                      color='orange', s=100, zorder=5, label='Peak')
            
            # Annotate peak height
            ax.annotate(f'Height: {spike["peak_height"]:.1f}',
                       xy=(peak_time, spike['peak_absolute_value']),
                       xytext=(peak_time + 2, spike['peak_absolute_value'] + 5),
                       arrowprops=dict(arrowstyle='->', color='orange'),
                       fontsize=9)
            
            # Recovery point if available
            if pd.notna(spike['recovery_time']):
                recovery_time = event_time + spike['recovery_time']
                if recovery_time <= window_end:
                    ax.scatter(recovery_time, spike['pre_spike_mean'], 
                              color='purple', s=100, marker='v', zorder=5)
                    ax.annotate(f'Recovery: {spike["recovery_time"]:.1f}h',
                               xy=(recovery_time, spike['pre_spike_mean']),
                               xytext=(recovery_time + 2, spike['pre_spike_mean'] - 5),
                               arrowprops=dict(arrowstyle='->', color='purple'),
                               fontsize=9)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('O₂ (%)')
            ax.set_title(f'Event {spike["event_number"]} - Well {spike["well_id"].split("_")[-1]}')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_spikes, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'spike_detection_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_event_timing_comparison():
    """Compare detected event times with actual event data."""
    # Load spike features
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get unique plates
    plates = spike_df['plate_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(plates)))
    
    for plate_idx, plate_id in enumerate(plates[:5]):  # First 5 plates
        # Get spike events for this plate
        plate_spikes = spike_df[spike_df['plate_id'] == plate_id]
        
        # Load actual event data
        event_data = load_event_data_for_plate(plate_id)
        
        if len(event_data) == 0:
            continue
        
        # Need to align times - get first timestamp from time series
        wells = plate_spikes['well_id'].unique()
        if len(wells) > 0:
            ts_data = load_time_series_data(plate_id, [wells[0]])
            if len(ts_data) > 0:
                plate_start = ts_data['timestamp'].min()
                event_data['elapsed_hours'] = (event_data['occurred_at'] - plate_start).dt.total_seconds() / 3600
                
                # Plot actual events
                y_pos = plate_idx * 2
                ax.scatter(event_data['elapsed_hours'], [y_pos] * len(event_data), 
                          color=colors[plate_idx], s=100, marker='o', 
                          label=f'Plate {plate_idx+1} - Actual' if plate_idx == 0 else '')
                
                # Plot detected events
                detected_times = plate_spikes['event_time_hours'].unique()
                ax.scatter(detected_times, [y_pos + 0.5] * len(detected_times),
                          color=colors[plate_idx], s=100, marker='^',
                          label=f'Plate {plate_idx+1} - Detected' if plate_idx == 0 else '')
                
                # Connect matching events
                for detected in detected_times:
                    # Find closest actual event
                    if len(event_data) > 0 and 'elapsed_hours' in event_data.columns:
                        closest_idx = np.abs(event_data['elapsed_hours'] - detected).idxmin()
                        closest_actual = event_data.loc[closest_idx, 'elapsed_hours']
                        
                        if abs(detected - closest_actual) < 6:  # Within 6 hours
                            ax.plot([closest_actual, detected], [y_pos, y_pos + 0.5],
                                   'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Time Since Experiment Start (hours)')
    ax.set_ylabel('Plate')
    ax.set_title('Event Timing Comparison: Actual vs Detected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis labels
    ax.set_yticks([i*2 + 0.25 for i in range(len(plates[:5]))])
    ax.set_yticklabels([f'Plate {i+1}' for i in range(len(plates[:5]))])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'event_timing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate validation figures showing real data with events."""
    print("Generating event detection validation figures...")
    
    # Check if spike data exists
    spike_data_path = data_dir / "step3_spike_features.parquet"
    if not spike_data_path.exists():
        print(f"Error: Spike data not found at {spike_data_path}")
        return
    
    print("1. Creating time series with events overlay...")
    create_time_series_with_events_overlay()
    
    print("2. Creating spike detection validation...")
    create_spike_detection_validation()
    
    print("3. Creating event timing comparison...")
    create_event_timing_comparison()
    
    print(f"\n✅ Validation figures created in: {fig_dir}")
    print("\nNew figures generated:")
    print("  - time_series_with_events_overlay.png")
    print("  - spike_detection_validation.png")
    print("  - event_timing_comparison.png")

if __name__ == "__main__":
    main()