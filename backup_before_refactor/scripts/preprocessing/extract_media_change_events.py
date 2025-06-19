#!/usr/bin/env python3
"""
Extract Media Change Events from Control Wells
Regenerate the media change event data that was deleted.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import os
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

print("=" * 80)
print("EXTRACTING MEDIA CHANGE EVENTS FROM CONTROL WELLS")
print("=" * 80)

# Connect to database
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

# Get well map data to identify control wells
print("\n📊 Loading well map data...")

# Load well map data first
well_map_query = """
SELECT 
    plate_id,
    well_number,
    plate_id || '_' || well_number as well_id,
    is_excluded
FROM postgres.well_map_data
WHERE is_excluded = false
"""
well_map_df = conn.execute(well_map_query).fetchdf()

# Load drug mapping from our integrated data
print("\n📊 Loading drug mapping...")
if (results_dir / "wells_drugs_integrated.parquet").exists():
    drug_map_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")
    # Merge with well map
    well_map_df = well_map_df.merge(
        drug_map_df[['well_id', 'drug', 'concentration']],
        on='well_id',
        how='left'
    )
else:
    print("⚠️ No drug mapping found, will identify control wells by well number pattern")
    # Assume wells ending in 1-4 or containing certain patterns
    well_map_df['drug'] = 'Unknown'
    well_map_df['concentration'] = np.nan

# Identify control wells
print("\n🔍 Identifying control wells...")
control_conditions = [
    well_map_df['drug'].str.lower().str.contains('dmso', na=False),
    well_map_df['drug'].str.lower().str.contains('control', na=False),
    well_map_df['drug'].str.lower().str.contains('media', na=False),
    well_map_df['concentration'] == 0,
    well_map_df['concentration'].isna(),
    # Also check well numbers - often controls are in specific columns
    well_map_df['well_number'].isin([1, 2, 3, 4])  # Common control well positions
]
control_mask = pd.concat(control_conditions, axis=1).any(axis=1)
control_wells = well_map_df[control_mask].copy()

print(f"   Found {len(control_wells):,} control wells")
print(f"   Across {control_wells['plate_id'].nunique()} plates")

# Function to detect spikes in time series
def detect_spikes(times, values, min_spike_height=11.1, window_size=5):
    """Detect positive spikes in oxygen data"""
    spikes = []
    
    if len(values) < window_size * 2:
        return spikes
    
    for i in range(window_size, len(values) - window_size):
        # Get baseline before potential spike
        baseline = np.median(values[i-window_size:i])
        
        # Get potential spike value
        spike_val = values[i]
        
        # Calculate spike height
        spike_height = spike_val - baseline
        
        # Check if it's a significant positive spike
        if spike_height >= min_spike_height:
            # Check recovery after spike
            post_spike = np.median(values[i+1:i+window_size+1])
            if post_spike < spike_val - 5:  # Must drop by at least 5% O2
                spikes.append({
                    'time': times[i],
                    'spike_height': spike_height,
                    'baseline': baseline,
                    'spike_value': spike_val
                })
    
    return spikes

# Process each plate
all_events = []
plates_processed = 0

for plate_id in tqdm(control_wells['plate_id'].unique(), desc="Processing plates"):
    plate_controls = control_wells[control_wells['plate_id'] == plate_id]
    
    if len(plate_controls) < 10:
        continue
    
    # Get time series data for control wells - need to use plate_id and well_number
    plate_id_str = plate_controls['plate_id'].iloc[0]
    well_numbers = plate_controls['well_number'].unique()
    well_numbers_str = ','.join(map(str, well_numbers))
    
    ts_query = f"""
    SELECT 
        plate_id || '_' || well_number as well_id,
        timestamp, 
        median_o2 as oxygen 
    FROM postgres.processed_data
    WHERE plate_id = '{plate_id_str}'
      AND well_number IN ({well_numbers_str})
      AND is_excluded = false
    ORDER BY well_number, timestamp
    """
    
    ts_data = conn.execute(ts_query).fetchdf()
    
    if len(ts_data) == 0:
        continue
    
    # Process each well
    plate_spikes = []
    
    for well_id in plate_controls['well_id']:
        well_data = ts_data[ts_data['well_id'] == well_id].copy()
        
        if len(well_data) < 50:
            continue
        
        # Calculate time in hours
        well_data['hours'] = (well_data['timestamp'] - well_data['timestamp'].min()).dt.total_seconds() / 3600
        
        # Detect spikes
        times = well_data['hours'].values
        values = well_data['oxygen'].values
        
        spikes = detect_spikes(times, values)
        
        for spike in spikes:
            plate_spikes.append({
                'well_id': well_id,
                'plate_id': plate_id,
                'event_time_hours': spike['time'],
                'spike_height': spike['spike_height'],
                'baseline_o2': spike['baseline'],
                'spike_o2': spike['spike_value']
            })
    
    if not plate_spikes:
        continue
    
    # Cluster spikes by time to find synchronized events
    spike_df = pd.DataFrame(plate_spikes)
    spike_df = spike_df.sort_values('event_time_hours')
    
    # Group spikes within 2-hour windows
    clustered_events = []
    current_cluster = []
    
    for _, spike in spike_df.iterrows():
        if not current_cluster:
            current_cluster = [spike]
        elif abs(spike['event_time_hours'] - current_cluster[0]['event_time_hours']) <= 2.0:
            current_cluster.append(spike)
        else:
            # Process current cluster
            if len(current_cluster) >= len(plate_controls) * 0.4:  # At least 40% of control wells
                event_time = np.median([s['event_time_hours'] for s in current_cluster])
                event_height = np.median([s['spike_height'] for s in current_cluster])
                
                for well in plate_controls['well_id']:
                    all_events.append({
                        'well_id': well,
                        'plate_id': plate_id,
                        'event_time_hours': event_time,
                        'median_spike_height': event_height,
                        'n_wells_spiking': len(current_cluster),
                        'pct_wells_spiking': len(current_cluster) / len(plate_controls) * 100
                    })
            
            current_cluster = [spike]
    
    # Don't forget last cluster
    if current_cluster and len(current_cluster) >= len(plate_controls) * 0.4:
        event_time = np.median([s['event_time_hours'] for s in current_cluster])
        event_height = np.median([s['spike_height'] for s in current_cluster])
        
        for well in plate_controls['well_id']:
            all_events.append({
                'well_id': well,
                'plate_id': plate_id,
                'event_time_hours': event_time,
                'median_spike_height': event_height,
                'n_wells_spiking': len(current_cluster),
                'pct_wells_spiking': len(current_cluster) / len(plate_controls) * 100
            })
    
    plates_processed += 1

# Create events dataframe
events_df = pd.DataFrame(all_events)

# Extend events to all wells on the same plate
print(f"\n📊 Extending events to all wells on plate...")
plate_events = events_df[['plate_id', 'event_time_hours', 'median_spike_height', 
                         'n_wells_spiking', 'pct_wells_spiking']].drop_duplicates()

# Get all wells for event extension
all_wells_query = """
SELECT 
    plate_id || '_' || well_number as well_id,
    plate_id
FROM postgres.well_map_data
WHERE is_excluded = false
"""
all_wells_df = conn.execute(all_wells_query).fetchdf()

# Merge to get events for all wells
extended_events = all_wells_df.merge(plate_events, on='plate_id', how='inner')

# Add event numbering
extended_events = extended_events.sort_values(['plate_id', 'event_time_hours'])
extended_events['event_number'] = extended_events.groupby('plate_id').cumcount() + 1

print(f"\n📋 EVENT SUMMARY:")
print(f"   Total media change events detected: {len(plate_events)}")
print(f"   Plates with events: {plate_events['plate_id'].nunique()}")
print(f"   Wells with events: {extended_events['well_id'].nunique():,}")
print(f"   Average events per plate: {len(plate_events) / plate_events['plate_id'].nunique():.1f}")

# Save results
output_file = results_dir / "media_change_events.parquet"
extended_events.to_parquet(output_file, index=False)
print(f"\n💾 Saved media change events to: {output_file}")

# Also save summary statistics
summary_stats = {
    'total_events': len(plate_events),
    'plates_with_events': plate_events['plate_id'].nunique(),
    'wells_with_events': extended_events['well_id'].nunique(),
    'avg_events_per_plate': len(plate_events) / plate_events['plate_id'].nunique(),
    'median_spike_height': plate_events['median_spike_height'].median(),
    'median_well_coverage': plate_events['pct_wells_spiking'].median()
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(results_dir / "media_change_events_summary.csv", index=False)

print("\n✅ Media change event extraction complete!")

conn.close()