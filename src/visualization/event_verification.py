#!/usr/bin/env python3
"""
Verify Media Change Events - Visual Inspection
Show actual time series data with detected events overlaid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import duckdb
import os
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_verification"
fig_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("=" * 80)
print("MEDIA CHANGE EVENT VERIFICATION")
print("=" * 80)

# Load media change events
events_df = pd.read_parquet(results_dir / "media_change_events.parquet")
print(f"\n📊 Loaded {len(events_df):,} media change events")

# Get unique plates with events
plates_with_events = events_df['plate_id'].unique()
print(f"   Plates with events: {len(plates_with_events)}")

# Connect to database
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

# Function to get time series data for a plate
def get_plate_data(plate_id, conn):
    """Get all time series data for a plate"""
    query = f"""
    SELECT 
        plate_id || '_' || well_number as well_id,
        well_number,
        timestamp,
        median_o2 as oxygen
    FROM postgres.processed_data
    WHERE plate_id = '{plate_id}'
      AND is_excluded = false
    ORDER BY well_number, timestamp
    """
    return conn.execute(query).fetchdf()

# Create detailed visualization for each plate
for plate_idx, plate_id in enumerate(plates_with_events[:3]):  # Show first 3 plates
    print(f"\n🔍 Processing plate {plate_idx + 1}/{min(3, len(plates_with_events))}: {plate_id}")
    
    # Get time series data
    plate_data = get_plate_data(plate_id, conn)
    
    if len(plate_data) == 0:
        print(f"   ⚠️ No data found for plate {plate_id}")
        continue
    
    # Get events for this plate
    plate_events = events_df[events_df['plate_id'] == plate_id].copy()
    event_times = plate_events['event_time_hours'].unique()
    
    print(f"   Found {len(event_times)} unique event times")
    
    # Calculate time in hours for all data
    min_time = plate_data['timestamp'].min()
    plate_data['hours'] = (plate_data['timestamp'] - min_time).dt.total_seconds() / 3600
    
    # Get unique wells
    wells = sorted(plate_data['well_number'].unique())
    
    # Identify control wells
    control_wells = [1, 2, 3, 4]  # Common control well positions
    
    # Create figure with subplots
    n_subplots = min(8, len(wells))  # Show up to 8 wells
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 2.5 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    fig.suptitle(f'Media Change Event Verification - Plate {plate_id[:8]}...', 
                 fontsize=16, fontweight='bold')
    
    # Plot each well
    for idx, well_num in enumerate(wells[:n_subplots]):
        ax = axes[idx]
        
        # Get data for this well
        well_data = plate_data[plate_data['well_number'] == well_num].copy()
        well_data = well_data.sort_values('hours')
        
        # Plot oxygen data
        line_color = 'blue' if well_num in control_wells else 'gray'
        line_alpha = 1.0 if well_num in control_wells else 0.7
        
        ax.plot(well_data['hours'], well_data['oxygen'], 
                color=line_color, alpha=line_alpha, linewidth=1.5)
        
        # Add event markers
        for event_time in event_times:
            ax.axvline(event_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add shaded region for spike recovery period (6 hours)
            ax.axvspan(event_time, event_time + 6, alpha=0.2, color='yellow')
        
        # Formatting
        ax.set_ylabel(f'Well {well_num}\n{"(Control)" if well_num in control_wells else ""}', 
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        
        # Add well statistics
        stats_text = f'μ={well_data["oxygen"].mean():.1f}, σ={well_data["oxygen"].std():.1f}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set common x-label
    axes[-1].set_xlabel('Time (hours)', fontsize=12)
    axes[0].set_title('Oxygen Level (%) - Red lines = Detected Media Changes, Yellow = Recovery Period', 
                     fontsize=12)
    
    # Add legend
    control_patch = mpatches.Patch(color='blue', label='Control Wells')
    treated_patch = mpatches.Patch(color='gray', label='Treated Wells')
    event_line = mpatches.Patch(color='red', label='Media Change Events')
    recovery_patch = mpatches.Patch(color='yellow', alpha=0.2, label='Recovery Period (6h)')
    
    axes[0].legend(handles=[control_patch, treated_patch, event_line, recovery_patch], 
                  loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'plate_{plate_idx+1}_event_verification.png', dpi=200, bbox_inches='tight')
    plt.close()

# Create summary visualization showing event timing across all plates
print("\n📊 Creating event timing summary...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Media Change Events Summary Across All Plates', fontsize=16, fontweight='bold')

# Plot 1: Event timeline for each plate
ax1.set_title('Event Timeline by Plate', fontsize=14)

for idx, plate_id in enumerate(plates_with_events):
    plate_events = events_df[events_df['plate_id'] == plate_id]
    event_times = plate_events['event_time_hours'].unique()
    
    # Plot events as vertical lines
    for event_time in event_times:
        ax1.plot([event_time, event_time], [idx - 0.3, idx + 0.3], 
                'ro-', linewidth=2, markersize=8)
    
    # Add plate label
    ax1.text(-10, idx, f'Plate {idx+1}\n({plate_id[:8]}...)', 
             ha='right', va='center', fontsize=10)

ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Plate', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim(-20, 400)
ax1.set_ylim(-1, len(plates_with_events))

# Add typical experiment phases
phases = [
    (0, 48, 'Baseline/Dosing', 'lightblue'),
    (48, 168, 'Acute Response', 'lightgreen'),
    (168, 336, 'Chronic Exposure', 'lightyellow'),
    (336, 400, 'Late Phase', 'lightcoral')
]

for start, end, label, color in phases:
    ax1.axvspan(start, end, alpha=0.2, color=color, label=label)

ax1.legend(loc='upper right', fontsize=10)

# Plot 2: Event frequency histogram
ax2.set_title('Distribution of Media Change Times', fontsize=14)

all_event_times = events_df['event_time_hours'].values
ax2.hist(all_event_times, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax2.set_xlabel('Time (hours)', fontsize=12)
ax2.set_ylabel('Number of Events', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add statistics
mean_time = np.mean(all_event_times)
median_time = np.median(all_event_times)
ax2.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_time:.1f}h')
ax2.axvline(median_time, color='green', linestyle='--', linewidth=2, 
           label=f'Median: {median_time:.1f}h')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(fig_dir / 'event_timing_summary_all_plates.png', dpi=200, bbox_inches='tight')
plt.close()

# Create detailed event statistics
print("\n📋 EVENT STATISTICS BY PLATE:")
print("-" * 80)
print(f"{'Plate':<40} {'N Events':<10} {'Event Times (hours)':<40}")
print("-" * 80)

for plate_id in plates_with_events:
    plate_events = events_df[events_df['plate_id'] == plate_id]
    event_times = sorted(plate_events['event_time_hours'].unique())
    n_events = len(event_times)
    
    # Format event times
    event_str = ', '.join([f'{t:.1f}' for t in event_times[:5]])
    if len(event_times) > 5:
        event_str += '...'
    
    print(f"{plate_id:<40} {n_events:<10} {event_str:<40}")

# Calculate inter-event intervals
print("\n📊 INTER-EVENT INTERVALS:")
intervals = []
for plate_id in plates_with_events:
    plate_events = events_df[events_df['plate_id'] == plate_id]
    event_times = sorted(plate_events['event_time_hours'].unique())
    
    if len(event_times) > 1:
        plate_intervals = np.diff(event_times)
        intervals.extend(plate_intervals)

if intervals:
    print(f"   Mean interval: {np.mean(intervals):.1f} ± {np.std(intervals):.1f} hours")
    print(f"   Median interval: {np.median(intervals):.1f} hours")
    print(f"   Min interval: {np.min(intervals):.1f} hours")
    print(f"   Max interval: {np.max(intervals):.1f} hours")

# Create spike characterization plot for a sample plate
print("\n🔍 Creating detailed spike characterization...")

sample_plate = plates_with_events[0]
plate_data = get_plate_data(sample_plate, conn)

if len(plate_data) > 0:
    # Calculate time in hours
    min_time = plate_data['timestamp'].min()
    plate_data['hours'] = (plate_data['timestamp'] - min_time).dt.total_seconds() / 3600
    
    # Get control wells
    control_wells = [1, 2, 3, 4]
    control_data = plate_data[plate_data['well_number'].isin(control_wells)]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Spike Characterization - Plate {sample_plate[:8]}...', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: All control wells overlaid
    ax = axes[0, 0]
    for well_num in control_wells:
        well_data = control_data[control_data['well_number'] == well_num]
        ax.plot(well_data['hours'], well_data['oxygen'], 
                alpha=0.7, linewidth=1, label=f'Well {well_num}')
    
    # Add events
    plate_events = events_df[events_df['plate_id'] == sample_plate]
    for event_time in plate_events['event_time_hours'].unique():
        ax.axvline(event_time, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Oxygen (%)')
    ax.set_title('Control Wells with Media Change Events')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoom on first event
    ax = axes[0, 1]
    first_event = sorted(plate_events['event_time_hours'].unique())[0]
    
    for well_num in control_wells[:2]:  # Show first 2 control wells
        well_data = control_data[control_data['well_number'] == well_num]
        mask = (well_data['hours'] >= first_event - 12) & (well_data['hours'] <= first_event + 12)
        
        if mask.any():
            ax.plot(well_data.loc[mask, 'hours'], well_data.loc[mask, 'oxygen'], 
                   'o-', alpha=0.8, linewidth=2, markersize=4, label=f'Well {well_num}')
    
    ax.axvline(first_event, color='red', linestyle='--', linewidth=2, label='Media Change')
    ax.axvspan(first_event, first_event + 6, alpha=0.2, color='yellow', label='Recovery')
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Oxygen (%)')
    ax.set_title(f'Detailed View: Event at {first_event:.1f}h')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average control well response
    ax = axes[1, 0]
    
    # Calculate average response around each event
    window_before = 6  # hours before event
    window_after = 12  # hours after event
    
    for event_idx, event_time in enumerate(sorted(plate_events['event_time_hours'].unique()[:3])):
        responses = []
        
        for well_num in control_wells:
            well_data = control_data[control_data['well_number'] == well_num]
            mask = (well_data['hours'] >= event_time - window_before) & \
                   (well_data['hours'] <= event_time + window_after)
            
            if mask.sum() > 10:
                # Align to event time
                aligned_data = well_data.loc[mask].copy()
                aligned_data['relative_hours'] = aligned_data['hours'] - event_time
                responses.append(aligned_data[['relative_hours', 'oxygen']])
        
        if responses:
            # Combine and average
            combined = pd.concat(responses)
            avg_response = combined.groupby(pd.cut(combined['relative_hours'], 
                                                   bins=np.linspace(-window_before, window_after, 50)))['oxygen'].mean()
            
            x_vals = np.linspace(-window_before, window_after, len(avg_response))
            ax.plot(x_vals, avg_response.values, linewidth=2, 
                   label=f'Event {event_idx+1} ({event_time:.1f}h)')
    
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(0, 6, alpha=0.2, color='yellow')
    ax.set_xlabel('Time Relative to Media Change (hours)')
    ax.set_ylabel('Average Oxygen (%)')
    ax.set_title('Average Control Well Response to Media Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Spike height distribution
    ax = axes[1, 1]
    
    # Calculate spike heights
    spike_heights = []
    for event_time in plate_events['event_time_hours'].unique():
        for well_num in control_wells:
            well_data = control_data[control_data['well_number'] == well_num]
            
            # Get baseline before spike
            baseline_mask = (well_data['hours'] >= event_time - 6) & \
                           (well_data['hours'] < event_time - 1)
            spike_mask = (well_data['hours'] >= event_time - 1) & \
                        (well_data['hours'] <= event_time + 2)
            
            if baseline_mask.sum() > 3 and spike_mask.sum() > 3:
                baseline = well_data.loc[baseline_mask, 'oxygen'].median()
                spike_max = well_data.loc[spike_mask, 'oxygen'].max()
                spike_height = spike_max - baseline
                
                if spike_height > 5:  # Minimum spike height
                    spike_heights.append(spike_height)
    
    if spike_heights:
        ax.hist(spike_heights, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
        ax.axvline(np.median(spike_heights), color='red', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(spike_heights):.1f}%')
        ax.set_xlabel('Spike Height (% O₂)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Media Change Spike Heights')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'spike_characterization_detailed.png', dpi=200, bbox_inches='tight')
    plt.close()

print(f"\n✅ Event verification complete!")
print(f"   Figures saved to: {fig_dir}")
print(f"\n📊 Key files generated:")
print(f"   - plate_*_event_verification.png (detailed well-by-well view)")
print(f"   - event_timing_summary_all_plates.png (overview of all events)")
print(f"   - spike_characterization_detailed.png (spike analysis)")

conn.close()