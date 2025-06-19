#!/usr/bin/env python3
"""
Event-Aware Features Plan: Step-by-step approach to integrate media changes
Let's understand what we have and what we need before building.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
results_dir = project_root / "results" / "data"

print("=" * 80)
print("EVENT-AWARE FEATURES: UNDERSTANDING WHAT WE NEED")
print("=" * 80)

# 1. Load what we have
print("\n📊 STEP 1: INVENTORY OF AVAILABLE DATA")
print("-" * 40)

# Wells with drugs
wells_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")
print(f"✅ Wells data: {len(wells_df):,} wells")

# Check for event data
event_files = list(results_dir.glob("*event*.parquet"))
print(f"\n📁 Event-related files found:")
for f in event_files:
    print(f"   • {f.name}")

# Load Step 1 quality data
step1_df = pd.read_parquet(results_dir / "step1_quality_assessment_all_plates.parquet")
print(f"\n✅ Step 1 data: {len(step1_df):,} wells with quality metrics")

# 2. Understand media change events
print("\n📊 STEP 2: UNDERSTANDING MEDIA CHANGE EVENTS")
print("-" * 40)

# Check if we have control-based events
if (results_dir / "step3_production_media_change_events.parquet").exists():
    events_df = pd.read_parquet(results_dir / "step3_production_media_change_events.parquet")
    print(f"✅ Production events: {len(events_df):,} event records")
    print(f"   Plates with events: {events_df['plate_id'].nunique()}")
    print(f"   Wells with events: {events_df['well_id'].nunique()}")
    
    # Sample event timing
    print(f"\n⏰ Event timing distribution:")
    print(events_df['event_time_hours'].describe().round(1))
else:
    print("❌ No production event data found")

# Check for spike detection results
spike_files = list(results_dir.glob("*spike*.csv"))
if spike_files:
    print(f"\n📊 Spike analysis files:")
    for f in spike_files:
        print(f"   • {f.name}")

# 3. Plan the features we need
print("\n" + "="*80)
print("FEATURE EXTRACTION PLAN")
print("="*80)

print("\n🎯 GOAL: Extract oxygen features BETWEEN media changes")

print("\n📋 FEATURE CATEGORIES NEEDED:")

print("\n1️⃣ INTER-EVENT FEATURES (between media changes):")
print("   • Baseline oxygen level (first 6h after media change)")
print("   • Oxygen consumption rate (slope)")
print("   • Oxygen variability (CV)")
print("   • Time to minimum oxygen")
print("   • Recovery patterns")

print("\n2️⃣ EVENT RESPONSE FEATURES:")
print("   • Spike height (% O2 increase)")
print("   • Recovery time to baseline")
print("   • Post-spike baseline shift")
print("   • Spike sharpness (rate of change)")

print("\n3️⃣ TEMPORAL PROGRESSION FEATURES:")
print("   • Early phase (0-48h) vs late phase patterns")
print("   • Change in consumption rate over time")
print("   • Event response magnitude over time")
print("   • Cumulative oxygen deficit")

print("\n4️⃣ DOSE-RESPONSE FEATURES:")
print("   • Concentration-dependent consumption rate")
print("   • Concentration-dependent event response")
print("   • EC50 for oxygen effects")

# 4. Check data availability
print("\n" + "="*80)
print("DATA AVAILABILITY CHECK")
print("="*80)

# Count wells with both quality data and events
quality_wells = set(step1_df[step1_df['cv_o2'] <= 0.95]['well_id'])
print(f"\n📊 Quality wells (CV ≤ 0.95): {len(quality_wells):,}")

if 'events_df' in locals():
    event_wells = set(events_df['well_id'].unique())
    overlap_wells = quality_wells.intersection(event_wells)
    print(f"📊 Wells with events: {len(event_wells):,}")
    print(f"📊 Quality wells WITH events: {len(overlap_wells):,}")
    
    # Check drug representation
    wells_with_both = wells_df[wells_df['well_id'].isin(overlap_wells)]
    print(f"\n💊 Drugs represented in quality+event wells:")
    drug_counts = wells_with_both['drug'].value_counts()
    print(f"   Total drugs: {len(drug_counts)}")
    print(f"   Drugs with ≥8 wells: {(drug_counts >= 8).sum()}")
    print(f"   Drugs with ≥4 wells: {(drug_counts >= 4).sum()}")

# 5. Next steps
print("\n" + "="*80)
print("RECOMMENDED APPROACH")
print("="*80)

print("\n🚀 STEP-BY-STEP PLAN:")
print("\n1. Load time series data for wells with events")
print("2. Segment by media change events") 
print("3. Extract features from each segment")
print("4. Aggregate to drug level")
print("5. Correlate with DILI")

print("\n⚠️  CRITICAL DECISIONS:")
print("   • Include wells with <4 events? (more data vs cleaner patterns)")
print("   • How to handle missing events? (interpolate vs exclude)")
print("   • Normalize by control wells? (removes batch effects)")

print("\n📊 EXPECTED OUTCOME:")
print("   • ~100-200 drugs with event-aware features")
print("   • ~20-50 features per drug")
print("   • Should improve DILI correlation from r=0.26 to r>0.4")

print("\n✅ Ready to proceed with implementation!")