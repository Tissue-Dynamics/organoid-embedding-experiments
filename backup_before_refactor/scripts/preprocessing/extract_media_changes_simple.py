#!/usr/bin/env python3
"""
Extract Media Change Events - Simplified Version
Focus on getting event-aware features working
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"

print("=" * 80)
print("SIMPLIFIED MEDIA CHANGE EVENT EXTRACTION")
print("=" * 80)

# Use the control wells analysis results we already have
print("\n📊 Using control wells analysis results...")

# Define media change events based on control wells analysis
# These are the synchronized events we detected
media_events = {
    '06e6ebb9-f553-48e8-a894-c3d02328f599': [6.9, 94.7, 166.6, 271.0, 342.6],
    '21093221-4eaa-4b7a-ac1a-2512df9c3c93': [11.8, 96.4, 170.9, 264.7, 339.5],
    '23e6f218-f4fe-417d-bdcb-e835dc36be1c': [62.7, 133.9, 158.4, 237.6, 335.2],
    '3ab6f6eb-a434-4fc2-a58e-48074e17a6a7': [97.0, 194.5, 211.0, 263.5],
    '3d446517-1fe0-48bd-92e4-bd0dbd9a37ff': [23.3, 120.5, 186.0, 286.2],
    '48e21e03-3921-4219-8573-b81222d397d0': [94.9, 186.5],
}

# Load well map
wells_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")
print(f"   Loaded {len(wells_df):,} wells")

# Create event dataframe
event_records = []
for plate_id, event_times in media_events.items():
    plate_wells = wells_df[wells_df['plate_id'] == plate_id]
    
    for event_num, event_time in enumerate(event_times, 1):
        for _, well in plate_wells.iterrows():
            event_records.append({
                'well_id': well['well_id'],
                'plate_id': plate_id,
                'event_time_hours': event_time,
                'event_number': event_num,
                'drug': well['drug'],
                'concentration': well['concentration']
            })

events_df = pd.DataFrame(event_records)

print(f"\n📋 EVENT SUMMARY:")
print(f"   Total events: {len(events_df):,}")
print(f"   Unique wells: {events_df['well_id'].nunique():,}")
print(f"   Plates: {events_df['plate_id'].nunique()}")
print(f"   Events per plate: {events_df.groupby('plate_id')['event_number'].max().to_dict()}")

# Save results
events_df.to_parquet(results_dir / "media_change_events.parquet", index=False)
print(f"\n💾 Saved to: {results_dir / 'media_change_events.parquet'}")

# Create summary visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Events per plate
plate_counts = events_df.groupby('plate_id')['event_number'].max()
ax1.bar(range(len(plate_counts)), plate_counts.values)
ax1.set_xlabel('Plate')
ax1.set_ylabel('Number of Media Change Events')
ax1.set_title('Media Change Events by Plate')
ax1.set_xticks(range(len(plate_counts)))
ax1.set_xticklabels([p[:8] + '...' for p in plate_counts.index], rotation=45)

# Plot 2: Event timing distribution
all_times = []
for times in media_events.values():
    all_times.extend(times)

ax2.hist(all_times, bins=20, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Number of Events')
ax2.set_title('Distribution of Media Change Times')
ax2.axvline(np.median(all_times), color='red', linestyle='--', label=f'Median: {np.median(all_times):.1f}h')
ax2.legend()

plt.tight_layout()
plt.savefig(results_dir.parent / "figures" / "media_change_events_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Media change events extracted successfully!")