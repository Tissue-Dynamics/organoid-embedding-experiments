#!/usr/bin/env python3
"""
Create expanded improved events dataset with well_id column
"""

import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import os

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("Creating expanded improved events dataset...")

# Load improved events
improved_events_df = pd.read_parquet(results_dir / "improved_media_change_events.parquet")
print(f"Loaded {len(improved_events_df)} improved events")

# Connect to database
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

# Get all wells for the plates with events
plates_with_events = improved_events_df['plate_id'].unique()
print(f"Plates with events: {len(plates_with_events)}")

# Get well mapping with drug information
well_map_query = f"""
SELECT DISTINCT 
    p.plate_id,
    p.well_number,
    p.plate_id || '_' || p.well_number as well_id,
    w.drug,
    w.concentration
FROM postgres.processed_data p
LEFT JOIN postgres.well_map_data w 
  ON p.plate_id = w.plate_id AND p.well_number = w.well_number
WHERE p.is_excluded = false
  AND p.plate_id IN ('{"','".join(plates_with_events)}')
ORDER BY p.plate_id, p.well_number
"""

well_map = conn.execute(well_map_query).fetchdf()
print(f"Found {len(well_map)} wells across plates")

# Create expanded events
expanded_events = []

print(f"Improved events sample:")
print(improved_events_df.head())
print(f"\nWell map sample:")
print(well_map.head())

# Check plate ID matching
event_plates = set(improved_events_df['plate_id'].unique())
well_plates = set(well_map['plate_id'].unique())
print(f"\nEvent plates: {len(event_plates)}")
print(f"Well map plates: {len(well_plates)}")
print(f"Intersection: {len(event_plates & well_plates)}")

if len(event_plates & well_plates) == 0:
    print("No matching plates! Debugging...")
    print(f"Event plate example: '{list(event_plates)[0]}'")
    print(f"Well plate example: '{list(well_plates)[0]}'")
    print(f"Are they equal? {list(event_plates)[0] == list(well_plates)[0]}")

# Just use all events and wells for now
for i, event in improved_events_df.iterrows():
    for j, well in well_map.iterrows():
        if well['plate_id'] == event['plate_id']:
            expanded_events.append({
                'plate_id': event['plate_id'],
                'well_id': well['well_id'],
                'well_number': well['well_number'],
                'event_time_hours': event['event_time_hours'],
                'event_type': event['event_type'],
                'confidence': event['confidence'],
                'drug': well['drug'],
                'concentration': well['concentration']
            })

expanded_events_df = pd.DataFrame(expanded_events)
print(f"Created {len(expanded_events_df)} expanded event records")

# Save expanded dataset
expanded_events_df.to_parquet(results_dir / "improved_media_change_events_expanded.parquet", index=False)
print(f"Saved to: {results_dir / 'improved_media_change_events_expanded.parquet'}")

# Show sample
print(f"\nSample records:")
print(expanded_events_df.head())

print(f"\nSummary:")
print(f"  Wells: {expanded_events_df['well_id'].nunique()}")
print(f"  Events per well: {len(expanded_events_df) / expanded_events_df['well_id'].nunique():.1f}")
print(f"  Drugs: {expanded_events_df['drug'].nunique()}")

conn.close()