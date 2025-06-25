#!/usr/bin/env python3
"""
Comprehensive Event Timing Analysis
Analyze accuracy of recorded events vs actual spikes and identify missing events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
import os
from scipy.signal import find_peaks
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("=" * 80)
print("COMPREHENSIVE EVENT TIMING ANALYSIS")
print("=" * 80)

# Load recorded events
events_df = pd.read_parquet(results_dir / "media_change_events.parquet")
print(f"\n📊 Loaded {len(events_df):,} recorded media change events")

# Connect to database
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

def detect_actual_spikes(plate_data, control_wells=[1, 2, 3, 4], min_spike_height=8.0, prominence=5.0):
    """
    Detect actual oxygen spikes in control wells using signal processing
    Returns list of spike times in hours
    """
    # Get control well data
    control_data = plate_data[plate_data['well_number'].isin(control_wells)].copy()
    
    if len(control_data) == 0:
        return []
    
    # Calculate time in hours
    min_time = control_data['timestamp'].min()
    control_data['hours'] = (control_data['timestamp'] - min_time).dt.total_seconds() / 3600
    
    # Average across control wells at each timepoint
    avg_data = control_data.groupby('hours')['oxygen'].mean().reset_index()
    avg_data = avg_data.sort_values('hours')
    
    if len(avg_data) < 50:  # Need sufficient data
        return []
    
    # Smooth the data slightly to reduce noise
    from scipy.signal import savgol_filter
    if len(avg_data) > 11:
        try:
            smoothed = savgol_filter(avg_data['oxygen'].values, window_length=11, polyorder=2)
        except:
            smoothed = avg_data['oxygen'].values
    else:
        smoothed = avg_data['oxygen'].values
    
    # Find peaks (spikes upward)
    peaks, properties = find_peaks(smoothed, 
                                  height=min_spike_height, 
                                  prominence=prominence,
                                  distance=20)  # At least 20 time points apart
    
    # Convert peak indices to hours
    spike_times = avg_data.iloc[peaks]['hours'].values
    spike_heights = smoothed[peaks]
    
    return spike_times, spike_heights, avg_data, smoothed

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

# Analyze each plate systematically
plates_with_events = events_df['plate_id'].unique()
analysis_results = []

print(f"\n🔍 Analyzing {len(plates_with_events)} plates with recorded events...")

for plate_idx, plate_id in enumerate(plates_with_events):
    print(f"\n--- Plate {plate_idx + 1}/{len(plates_with_events)}: {plate_id[:8]}... ---")
    
    # Get plate data
    plate_data = get_plate_data(plate_id, conn)
    
    if len(plate_data) == 0:
        print("   ❌ No data found")
        continue
    
    # Get recorded events for this plate
    recorded_events = events_df[events_df['plate_id'] == plate_id]['event_time_hours'].unique()
    recorded_events = sorted(recorded_events)
    
    # Detect actual spikes
    try:
        spike_times, spike_heights, avg_data, smoothed = detect_actual_spikes(plate_data)
        
        print(f"   📋 Recorded events: {len(recorded_events)} at times {[f'{t:.1f}h' for t in recorded_events]}")
        print(f"   🔍 Detected spikes: {len(spike_times)} at times {[f'{t:.1f}h' for t in spike_times]}")
        
        # Analyze timing discrepancies
        timing_discrepancies = []
        matched_events = []
        
        for recorded_time in recorded_events:
            # Find closest detected spike within ±12 hours
            if len(spike_times) > 0:
                time_diffs = np.abs(spike_times - recorded_time)
                closest_idx = np.argmin(time_diffs)
                closest_time = spike_times[closest_idx]
                min_diff = time_diffs[closest_idx]
                
                if min_diff <= 12:  # Within 12 hours
                    timing_discrepancies.append({
                        'plate_id': plate_id,
                        'recorded_time': recorded_time,
                        'actual_spike_time': closest_time,
                        'time_difference': closest_time - recorded_time,
                        'abs_difference': min_diff,
                        'spike_height': spike_heights[closest_idx]
                    })
                    matched_events.append(closest_time)
                    print(f"   ✅ Event at {recorded_time:.1f}h → Spike at {closest_time:.1f}h (Δ{closest_time-recorded_time:+.1f}h)")
                else:
                    timing_discrepancies.append({
                        'plate_id': plate_id,
                        'recorded_time': recorded_time,
                        'actual_spike_time': None,
                        'time_difference': None,
                        'abs_difference': None,
                        'spike_height': None
                    })
                    print(f"   ❌ Event at {recorded_time:.1f}h → No matching spike found")
        
        # Find unmatched spikes (potential missing events)
        unmatched_spikes = []
        for spike_time, spike_height in zip(spike_times, spike_heights):
            # Check if this spike is within 12h of any recorded event
            if len(recorded_events) == 0 or np.min(np.abs(recorded_events - spike_time)) > 12:
                unmatched_spikes.append({
                    'plate_id': plate_id,
                    'spike_time': spike_time,
                    'spike_height': spike_height
                })
                print(f"   🔍 Unmatched spike at {spike_time:.1f}h (height: {spike_height:.1f}%) - Potential missing event")
        
        analysis_results.append({
            'plate_id': plate_id,
            'n_recorded_events': len(recorded_events),
            'n_detected_spikes': len(spike_times),
            'n_matched': len([d for d in timing_discrepancies if d['actual_spike_time'] is not None]),
            'n_unmatched_events': len([d for d in timing_discrepancies if d['actual_spike_time'] is None]),
            'n_unmatched_spikes': len(unmatched_spikes),
            'timing_discrepancies': timing_discrepancies,
            'unmatched_spikes': unmatched_spikes,
            'avg_data': avg_data,
            'smoothed': smoothed,
            'spike_times': spike_times,
            'spike_heights': spike_heights,
            'recorded_events': recorded_events
        })
        
    except Exception as e:
        print(f"   ❌ Error analyzing plate: {e}")
        continue

# Create comprehensive summary
print(f"\n" + "=" * 60)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("=" * 60)

# Flatten timing discrepancies
all_discrepancies = []
all_unmatched_spikes = []

for result in analysis_results:
    all_discrepancies.extend(result['timing_discrepancies'])
    all_unmatched_spikes.extend(result['unmatched_spikes'])

# Convert to DataFrames
discrepancies_df = pd.DataFrame(all_discrepancies)
unmatched_spikes_df = pd.DataFrame(all_unmatched_spikes)

print(f"\n📊 TIMING ACCURACY ANALYSIS:")
print(f"   Total recorded events: {len(discrepancies_df)}")

if len(discrepancies_df) > 0:
    matched_df = discrepancies_df.dropna(subset=['actual_spike_time'])
    unmatched_events_df = discrepancies_df[discrepancies_df['actual_spike_time'].isna()]
    
    print(f"   Matched to actual spikes: {len(matched_df)} ({len(matched_df)/len(discrepancies_df)*100:.1f}%)")
    print(f"   No matching spike found: {len(unmatched_events_df)} ({len(unmatched_events_df)/len(discrepancies_df)*100:.1f}%)")
    
    if len(matched_df) > 0:
        print(f"\n📏 TIMING DISCREPANCIES (for matched events):")
        print(f"   Mean time difference: {matched_df['time_difference'].mean():+.1f} ± {matched_df['time_difference'].std():.1f} hours")
        print(f"   Median time difference: {matched_df['time_difference'].median():+.1f} hours")
        print(f"   Range: {matched_df['time_difference'].min():+.1f} to {matched_df['time_difference'].max():+.1f} hours")
        print(f"   Mean absolute error: {matched_df['abs_difference'].mean():.1f} ± {matched_df['abs_difference'].std():.1f} hours")

print(f"\n🔍 MISSING EVENTS ANALYSIS:")
print(f"   Unmatched spikes (potential missing events): {len(unmatched_spikes_df)}")

if len(unmatched_spikes_df) > 0:
    print(f"   Mean spike height: {unmatched_spikes_df['spike_height'].mean():.1f} ± {unmatched_spikes_df['spike_height'].std():.1f} %O₂")
    print(f"   Spike height range: {unmatched_spikes_df['spike_height'].min():.1f} to {unmatched_spikes_df['spike_height'].max():.1f} %O₂")

# Create detailed visualizations
print(f"\n📈 Creating detailed analysis visualizations...")

# Figure 1: Timing discrepancy analysis
if len(matched_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Event Timing Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Time difference distribution
    ax = axes[0, 0]
    ax.hist(matched_df['time_difference'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect timing')
    ax.axvline(matched_df['time_difference'].mean(), color='orange', linestyle='--', linewidth=2, 
              label=f'Mean: {matched_df["time_difference"].mean():+.1f}h')
    ax.set_xlabel('Time Difference (Actual - Recorded) [hours]')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Timing Discrepancies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Absolute error vs spike height
    ax = axes[0, 1]
    scatter = ax.scatter(matched_df['spike_height'], matched_df['abs_difference'], 
                        alpha=0.7, c=matched_df['recorded_time'], cmap='viridis')
    ax.set_xlabel('Spike Height (%O₂)')
    ax.set_ylabel('Absolute Timing Error (hours)')
    ax.set_title('Timing Error vs Spike Height')
    plt.colorbar(scatter, ax=ax, label='Recorded Time (hours)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Timing bias by plate
    ax = axes[1, 0]
    plate_bias = matched_df.groupby('plate_id')['time_difference'].mean()
    plate_bias.plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_xlabel('Plate')
    ax.set_ylabel('Mean Timing Bias (hours)')
    ax.set_title('Average Timing Bias by Plate')
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative timing error
    ax = axes[1, 1]
    sorted_errors = np.sort(matched_df['abs_difference'])
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cumulative, linewidth=2, color='darkblue')
    ax.axvline(sorted_errors[int(0.5 * len(sorted_errors))], color='red', linestyle='--', 
              label=f'Median: {np.median(sorted_errors):.1f}h')
    ax.axvline(sorted_errors[int(0.9 * len(sorted_errors))], color='orange', linestyle='--',
              label=f'90th percentile: {np.percentile(sorted_errors, 90):.1f}h')
    ax.set_xlabel('Absolute Timing Error (hours)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Timing Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'timing_discrepancy_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

# Figure 2: Missing events analysis
if len(unmatched_spikes_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Missing Events Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Unmatched spike heights
    ax = axes[0, 0]
    ax.hist(unmatched_spikes_df['spike_height'], bins=15, edgecolor='black', alpha=0.7, color='lightcoral')
    ax.set_xlabel('Spike Height (%O₂)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Unmatched Spike Heights')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Unmatched spike timing
    ax = axes[0, 1]
    ax.hist(unmatched_spikes_df['spike_time'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Frequency')
    ax.set_title('Timing Distribution of Unmatched Spikes')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Missing events by plate
    ax = axes[1, 0]
    missing_by_plate = unmatched_spikes_df.groupby('plate_id').size()
    missing_by_plate.plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_xlabel('Plate')
    ax.set_ylabel('Number of Missing Events')
    ax.set_title('Missing Events by Plate')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Spike height vs timing for missing events
    ax = axes[1, 1]
    scatter = ax.scatter(unmatched_spikes_df['spike_time'], unmatched_spikes_df['spike_height'], 
                        alpha=0.7, c=range(len(unmatched_spikes_df)), cmap='plasma')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Spike Height (%O₂)')
    ax.set_title('Missing Events: Height vs Timing')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'missing_events_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

# Figure 3: Example plates showing detected vs recorded events
print(f"\n📊 Creating example plate comparisons...")

n_examples = min(3, len(analysis_results))
fig, axes = plt.subplots(n_examples, 1, figsize=(16, 5 * n_examples))
if n_examples == 1:
    axes = [axes]

fig.suptitle('Detected Spikes vs Recorded Events - Example Plates', fontsize=16, fontweight='bold')

for idx, result in enumerate(analysis_results[:n_examples]):
    ax = axes[idx]
    
    # Plot averaged control well data
    avg_data = result['avg_data']
    smoothed = result['smoothed']
    
    ax.plot(avg_data['hours'], avg_data['oxygen'], alpha=0.5, color='gray', linewidth=1, label='Raw average')
    ax.plot(avg_data['hours'], smoothed, color='blue', linewidth=2, label='Smoothed average')
    
    # Mark detected spikes
    for spike_time, spike_height in zip(result['spike_times'], result['spike_heights']):
        ax.scatter(spike_time, spike_height, color='red', s=100, marker='^', 
                  label='Detected spike' if spike_time == result['spike_times'][0] else "")
    
    # Mark recorded events
    for event_time in result['recorded_events']:
        ax.axvline(event_time, color='green', linestyle='--', alpha=0.7, linewidth=2,
                  label='Recorded event' if event_time == result['recorded_events'][0] else "")
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Average Oxygen (%)')
    ax.set_title(f'Plate {idx+1}: {result["plate_id"][:8]}... - '
                f'{result["n_detected_spikes"]} spikes, {result["n_recorded_events"]} recorded events')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'detected_vs_recorded_examples.png', dpi=200, bbox_inches='tight')
plt.close()

# Save detailed results
print(f"\n💾 Saving detailed analysis results...")

# Save timing discrepancies
if len(discrepancies_df) > 0:
    discrepancies_df.to_parquet(results_dir / 'event_timing_discrepancies.parquet')
    print(f"   Saved timing discrepancies: {len(discrepancies_df)} records")

# Save unmatched spikes
if len(unmatched_spikes_df) > 0:
    unmatched_spikes_df.to_parquet(results_dir / 'unmatched_spikes_missing_events.parquet')
    print(f"   Saved unmatched spikes: {len(unmatched_spikes_df)} records")

# Create summary report
summary_stats = {
    'total_plates_analyzed': len(analysis_results),
    'total_recorded_events': len(discrepancies_df),
    'total_detected_spikes': sum(result['n_detected_spikes'] for result in analysis_results),
    'matched_events': len(matched_df) if len(discrepancies_df) > 0 else 0,
    'unmatched_events': len(unmatched_events_df) if len(discrepancies_df) > 0 else 0,
    'potential_missing_events': len(unmatched_spikes_df),
    'mean_timing_error_hours': matched_df['time_difference'].mean() if len(matched_df) > 0 else None,
    'median_timing_error_hours': matched_df['time_difference'].median() if len(matched_df) > 0 else None,
    'mean_absolute_error_hours': matched_df['abs_difference'].mean() if len(matched_df) > 0 else None
}

pd.Series(summary_stats).to_json(results_dir / 'event_timing_analysis_summary.json')

print(f"\n✅ Event timing analysis complete!")
print(f"   Figures saved to: {fig_dir}")
print(f"   Data saved to: {results_dir}")

print(f"\n🎯 KEY FINDINGS:")
if len(matched_df) > 0:
    print(f"   - Timing accuracy: {len(matched_df)}/{len(discrepancies_df)} events matched ({len(matched_df)/len(discrepancies_df)*100:.1f}%)")
    print(f"   - Average timing bias: {matched_df['time_difference'].mean():+.1f} hours")
    print(f"   - Typical error: ±{matched_df['abs_difference'].mean():.1f} hours")
if len(unmatched_spikes_df) > 0:
    print(f"   - Potential missing events: {len(unmatched_spikes_df)} unmatched spikes")
    print(f"   - Missing event strength: {unmatched_spikes_df['spike_height'].mean():.1f} ± {unmatched_spikes_df['spike_height'].std():.1f} %O₂")

conn.close()