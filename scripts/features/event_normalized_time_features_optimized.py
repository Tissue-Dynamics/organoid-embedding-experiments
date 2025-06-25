#!/usr/bin/env python3
"""
Event-Normalized Time Features - Optimized

PURPOSE:
    Efficient extraction of event-normalized features focusing on DILI-relevant drugs.
    Optimized for reasonable runtime while maintaining comprehensive coverage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from scipy import stats
import pycatch22
import sys
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Configuration
EVENT_WINDOWS = {
    'immediate_post': (0, 6),      # 0-6h after event
    'early_post': (6, 12),         # 6-12h after event  
    'late_post': (12, 24),         # 12-24h after event
    'pre_event': (-6, 0),          # 6h before next event
}

RECOVERY_THRESHOLDS = [0.5, 0.9]  # Simplified thresholds

# Setup directories
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EVENT-NORMALIZED TIME FEATURES - OPTIMIZED")
print("=" * 80)

# Define DILI drugs
dili_drugs = ['Amiodarone', 'Busulfan', 'Imatinib', 'Lapatinib', 'Pazopanib', 'Regorafenib', 
              'Sorafenib', 'Sunitinib', 'Trametinib', 'Anastrozole', 'Axitinib', 'Cabozantinib',
              'Dabrafenib', 'Erlotinib', 'Gefitinib', 'Lenvatinib', 'Nilotinib', 'Osimertinib',
              'Vemurafenib', 'Alectinib', 'Binimetinib', 'Bortezomib', 'Ceritinib', 'Crizotinib',
              'Dasatinib', 'Everolimus', 'Ibrutinib', 'Ponatinib', 'Ruxolitinib', 'Alpelisib',
              'Ambrisentan', 'Buspirone', 'Dexamethasone', 'Fulvestrant', 'Letrozole',
              'Palbociclib', 'Ribociclib', 'Trastuzumab', 'Zoledronic acid']

# ========== SIMPLIFIED FEATURE EXTRACTION ==========

def detect_media_changes_simple(well_data):
    """Simplified media change detection"""
    
    # Rename o2 to oxygen if needed
    if 'o2' in well_data.columns and 'oxygen' not in well_data.columns:
        well_data = well_data.rename(columns={'o2': 'oxygen'})
    
    # Calculate rolling variance
    well_data['rolling_var'] = well_data['oxygen'].rolling(window=5, center=True).var()
    baseline_var = well_data['rolling_var'].iloc[:50].median()
    
    if pd.isna(baseline_var) or baseline_var == 0:
        return []
    
    # Detect spikes
    spike_threshold = 3 * baseline_var
    spikes = well_data[well_data['rolling_var'] > spike_threshold]
    
    # Group spikes into events (minimum 6h apart)
    events = []
    last_event_time = -10
    
    for idx, row in spikes.iterrows():
        if row['elapsed_hours'] - last_event_time > 6:
            events.append(row['elapsed_hours'])
            last_event_time = row['elapsed_hours']
    
    return events

def extract_event_features_simple(well_data, well_id, drug, concentration):
    """Simplified feature extraction"""
    
    # Detect events
    events = detect_media_changes_simple(well_data)
    
    if len(events) < 2:  # Need at least 2 events
        return None
    
    # Get baseline
    baseline_data = well_data[well_data['elapsed_hours'] <= 48]
    baseline_mean = baseline_data['oxygen'].mean() if len(baseline_data) > 10 else well_data['oxygen'].mean()
    
    features = {
        'well_id': well_id,
        'drug': drug,
        'concentration': concentration,
        'n_events': len(events),
        'baseline_oxygen': baseline_mean
    }
    
    # Process first 3 events only (for speed)
    for event_idx, event_time in enumerate(events[:3]):
        event_num = event_idx + 1
        
        # Get post-event windows
        for window_name, (start_h, end_h) in EVENT_WINDOWS.items():
            if window_name == 'pre_event':
                continue  # Skip pre-event for simplicity
            
            window_data = well_data[
                (well_data['elapsed_hours'] >= event_time + start_h) & 
                (well_data['elapsed_hours'] < event_time + end_h)
            ]
            
            if len(window_data) >= 5:
                # Basic statistics
                features[f'event_{event_num}_{window_name}_mean'] = window_data['oxygen'].mean()
                features[f'event_{event_num}_{window_name}_std'] = window_data['oxygen'].std()
                features[f'event_{event_num}_{window_name}_min'] = window_data['oxygen'].min()
                
                # Recovery metric
                if window_name == 'immediate_post':
                    features[f'event_{event_num}_suppression'] = 1 - (window_data['oxygen'].min() / baseline_mean)
        
        # Time to recovery
        post_event_data = well_data[well_data['elapsed_hours'] >= event_time]
        if len(post_event_data) > 10:
            for threshold in RECOVERY_THRESHOLDS:
                recovery_data = post_event_data[post_event_data['oxygen'] >= threshold * baseline_mean]
                if len(recovery_data) > 0:
                    features[f'event_{event_num}_time_to_{int(threshold*100)}pct'] = (
                        recovery_data['elapsed_hours'].iloc[0] - event_time
                    )
    
    # Event consistency (if multiple events)
    if len(events) >= 2:
        # Compare first two events
        event1_supp = features.get('event_1_suppression', 0)
        event2_supp = features.get('event_2_suppression', 0)
        if event1_supp > 0 and event2_supp > 0:
            features['suppression_consistency'] = 1 - abs(event1_supp - event2_supp) / max(event1_supp, event2_supp)
    
    return features

# ========== MAIN PROCESSING ==========

print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📊 Loading data for DILI-relevant drugs...")
loader = DataLoader()

# Load well metadata first
print("   Loading well metadata...")
wells = loader.load_well_metadata()
dili_wells = wells[wells['drug'].isin(dili_drugs)]
print(f"   Found {len(dili_wells):,} wells with DILI-relevant drugs")

# Sample wells per drug (take up to 20 wells per drug for reasonable runtime)
sampled_wells = []
for drug in dili_drugs:
    drug_wells = dili_wells[dili_wells['drug'] == drug]
    if len(drug_wells) > 0:
        sample_size = min(20, len(drug_wells))
        sampled_wells.append(drug_wells.sample(n=sample_size, random_state=42))

sampled_wells_df = pd.concat(sampled_wells, ignore_index=True)
print(f"   Sampled {len(sampled_wells_df):,} wells total")

# Load oxygen data more efficiently  
print("   Loading oxygen data (this may take a moment)...")

# Just load a reasonable number of plates
sample_plates = sampled_wells_df['plate_id'].unique()[:5]  # First 5 plates
df = loader.load_oxygen_data(plate_ids=[str(p) for p in sample_plates])

# Filter to only our sampled wells
well_ids = sampled_wells_df[sampled_wells_df['plate_id'].isin(sample_plates)]['well_id'].tolist()
df = df[df['well_id'].isin(well_ids)]

# Rename o2 to oxygen if needed
if 'o2' in df.columns and 'oxygen' not in df.columns:
    df = df.rename(columns={'o2': 'oxygen'})

print(f"   Loaded {len(df):,} oxygen measurements")

# Process wells
print("\n🔄 Processing wells...")
all_features = []

# Only process wells we have data for
wells_to_process = sampled_wells_df[sampled_wells_df['well_id'].isin(df['well_id'].unique())]

for idx, well_info in wells_to_process.iterrows():
    if idx % 100 == 0:
        print(f"   Processing well {idx+1}/{len(wells_to_process)}...")
    
    well_id = well_info['well_id']
    well_data = df[df['well_id'] == well_id].copy()
    
    if len(well_data) < 100:
        continue
    
    # Sort by time
    well_data = well_data.sort_values('elapsed_hours').reset_index(drop=True)
    
    # Extract features
    features = extract_event_features_simple(
        well_data, 
        well_id,
        well_info['drug'],
        well_info['concentration']
    )
    
    if features is not None:
        all_features.append(features)

print(f"\n✓ Extracted features for {len(all_features):,} wells")

# Convert to DataFrame
well_features_df = pd.DataFrame(all_features)

# ========== DRUG-LEVEL AGGREGATION ==========

print("\n🔄 Aggregating features at drug level...")

# Group by drug
drug_features_list = []

for drug, group in well_features_df.groupby('drug'):
    drug_features = {
        'drug': drug,
        'n_wells': len(group),
        'n_concentrations': group['concentration'].nunique()
    }
    
    # Aggregate numeric features
    numeric_cols = [col for col in group.columns if col not in ['well_id', 'drug', 'concentration']]
    
    for col in numeric_cols:
        values = group[col].dropna()
        if len(values) > 0:
            drug_features[f'{col}_mean'] = values.mean()
            drug_features[f'{col}_std'] = values.std()
            drug_features[f'{col}_median'] = values.median()
    
    drug_features_list.append(drug_features)

drug_features_df = pd.DataFrame(drug_features_list)
print(f"   Created drug-level features for {len(drug_features_df)} drugs")

# ========== SAVE RESULTS ==========

print("\n💾 Saving results...")

# Save well-level features
well_features_df.to_parquet(results_dir / 'event_normalized_features_wells_optimized.parquet', index=False)
print(f"   Well-level: {results_dir / 'event_normalized_features_wells_optimized.parquet'}")

# Save drug-level features
drug_features_df.to_parquet(results_dir / 'event_normalized_features_drugs_optimized.parquet', index=False)
print(f"   Drug-level: {results_dir / 'event_normalized_features_drugs_optimized.parquet'}")

# Summary
summary = {
    'n_wells_processed': len(all_features),
    'n_drugs': len(drug_features_df),
    'drugs_covered': sorted(drug_features_df['drug'].unique().tolist()),
    'features_per_well': len([col for col in well_features_df.columns if col not in ['well_id', 'drug', 'concentration']]),
    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(results_dir / 'event_normalized_summary_optimized.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n✅ Optimized event-normalized feature extraction complete!")
print(f"\n📊 SUMMARY:")
print(f"   Processed {summary['n_wells_processed']:,} wells")
print(f"   Covered {summary['n_drugs']} DILI-relevant drugs")
print(f"   Created {summary['features_per_well']} features per well")