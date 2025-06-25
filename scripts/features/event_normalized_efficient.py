#!/usr/bin/env python3
"""
Event-Normalized Features - Efficient Processing

PURPOSE:
    Most efficient extraction of event-normalized features, optimized for speed.
    Focuses on the most predictive features identified in previous analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Configuration - only the most important windows based on our results
EVENT_WINDOWS = {
    'immediate_post': (0, 6),      
    'late_post': (12, 24),         
}

# Setup directories
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EVENT-NORMALIZED FEATURES - EFFICIENT PROCESSING")
print("=" * 80)

# DILI drugs
dili_drugs = [
    'Amiodarone', 'Busulfan', 'Imatinib', 'Lapatinib', 'Pazopanib', 'Regorafenib', 
    'Sorafenib', 'Sunitinib', 'Trametinib', 'Anastrozole', 'Axitinib', 'Cabozantinib',
    'Dabrafenib', 'Erlotinib', 'Gefitinib', 'Lenvatinib', 'Nilotinib', 'Osimertinib',
    'Vemurafenib', 'Alectinib', 'Binimetinib', 'Bortezomib', 'Ceritinib', 'Crizotinib',
    'Dasatinib', 'Everolimus', 'Ibrutinib', 'Ponatinib', 'Ruxolitinib', 'Alpelisib',
    'Ambrisentan', 'Buspirone', 'Dexamethasone', 'Fulvestrant', 'Letrozole',
    'Palbociclib', 'Ribociclib', 'Trastuzumab', 'Zoledronic acid'
]

# ========== ULTRA-FAST FEATURE EXTRACTION ==========

def extract_key_features(well_data, well_id, drug, concentration):
    """Extract only the most predictive features"""
    
    if len(well_data) < 100:
        return None
    
    # Ensure oxygen column exists
    if 'o2' in well_data.columns and 'oxygen' not in well_data.columns:
        well_data = well_data.rename(columns={'o2': 'oxygen'})
    
    oxygen = well_data['oxygen'].values
    times = well_data['elapsed_hours'].values
    
    # Quick event detection - just find large variance spikes
    rolling_var = pd.Series(oxygen).rolling(window=5).var().fillna(0)
    baseline_var = np.median(rolling_var[:30])
    
    if baseline_var == 0:
        return None
    
    # Find events
    events = []
    spike_times = times[rolling_var > 3 * baseline_var]
    
    # Group spikes
    last_event = -10
    for t in spike_times:
        if t - last_event > 6:
            events.append(t)
            last_event = t
    
    if len(events) < 2:
        return None
    
    # Take first 3 events
    events = events[:3]
    
    # Baseline
    baseline_mean = np.mean(oxygen[times <= 48])
    
    features = {
        'drug': drug,
        'n_events': len(events),
        'baseline_oxygen': baseline_mean
    }
    
    # Extract key features for each event (focusing on event 3 which was most predictive)
    for i, event_time in enumerate(events):
        event_num = i + 1
        
        # Late post window (most predictive)
        late_mask = (times >= event_time + 12) & (times < event_time + 24)
        if np.sum(late_mask) >= 5:
            late_values = oxygen[late_mask]
            features[f'event_{event_num}_late_post_min'] = np.min(late_values)
            features[f'event_{event_num}_late_post_mean'] = np.mean(late_values)
            features[f'event_{event_num}_late_post_std'] = np.std(late_values)
        
        # Immediate post (for suppression)
        imm_mask = (times >= event_time) & (times < event_time + 6)
        if np.sum(imm_mask) >= 5:
            imm_values = oxygen[imm_mask]
            features[f'event_{event_num}_suppression'] = 1 - (np.min(imm_values) / baseline_mean)
    
    # Consistency metric
    if 'event_1_suppression' in features and 'event_2_suppression' in features:
        e1 = features['event_1_suppression']
        e2 = features['event_2_suppression']
        if e1 > 0 and e2 > 0:
            features['suppression_consistency'] = 1 - abs(e1 - e2) / max(e1, e2)
    
    return features

# ========== MAIN PROCESSING ==========

print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize loader
loader = DataLoader()

print("\n📊 Loading data for all DILI drugs...")

# Get all DILI wells
wells = loader.load_well_metadata()
dili_wells = wells[wells['drug'].isin(dili_drugs)]

print(f"   Found {len(dili_wells):,} wells for {dili_wells['drug'].nunique()} DILI drugs")

# Sample wells - take up to 40 wells per drug
sampled_wells = []
for drug in dili_drugs:
    drug_wells = dili_wells[dili_wells['drug'] == drug]
    if len(drug_wells) > 0:
        n_sample = min(40, len(drug_wells))
        sampled_wells.append(drug_wells.sample(n=n_sample, random_state=42))

sampled_wells_df = pd.concat(sampled_wells, ignore_index=True)
print(f"   Sampled {len(sampled_wells_df):,} wells")

# Process by loading limited plates at a time
unique_plates = sampled_wells_df['plate_id'].unique()
print(f"   Processing {len(unique_plates)} plates...")

all_features = []
plates_per_batch = 3

for i in range(0, len(unique_plates), plates_per_batch):
    batch_plates = unique_plates[i:i+plates_per_batch]
    print(f"\n   Processing plates {i+1}-{min(i+plates_per_batch, len(unique_plates))} of {len(unique_plates)}...")
    
    # Load batch data
    batch_data = loader.load_oxygen_data(plate_ids=[str(p) for p in batch_plates])
    
    # Process wells in this batch
    batch_wells = sampled_wells_df[sampled_wells_df['plate_id'].isin(batch_plates)]
    
    for _, well_info in batch_wells.iterrows():
        well_id = well_info['well_id']
        well_data = batch_data[batch_data['well_id'] == well_id]
        
        if len(well_data) > 0:
            well_data = well_data.sort_values('elapsed_hours')
            
            features = extract_key_features(
                well_data,
                well_id,
                well_info['drug'],
                well_info['concentration']
            )
            
            if features is not None:
                all_features.append(features)
    
    print(f"      Processed {len(batch_wells)} wells, extracted {len(all_features)} total features")

print(f"\n✓ Extracted features for {len(all_features)} wells")

# Convert to DataFrame
well_features_df = pd.DataFrame(all_features)

# Drug-level aggregation
print("\n🔄 Aggregating at drug level...")

drug_features = []
for drug in well_features_df['drug'].unique():
    drug_data = well_features_df[well_features_df['drug'] == drug]
    
    drug_feat = {
        'drug': drug,
        'n_wells': len(drug_data)
    }
    
    # Aggregate key features
    key_features = [col for col in drug_data.columns if col not in ['drug']]
    
    for feat in key_features:
        values = drug_data[feat].dropna()
        if len(values) > 0:
            drug_feat[f'{feat}_mean'] = values.mean()
            drug_feat[f'{feat}_std'] = values.std()
            drug_feat[f'{feat}_median'] = values.median()
    
    drug_features.append(drug_feat)

drug_features_df = pd.DataFrame(drug_features)
print(f"   Created features for {len(drug_features_df)} drugs")

# Save results
print("\n💾 Saving results...")

drug_features_df.to_parquet(results_dir / 'event_normalized_features_drugs_final.parquet', index=False)
print(f"   Saved to: {results_dir / 'event_normalized_features_drugs_final.parquet'}")

# Summary
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_wells_processed': len(all_features),
    'n_drugs_covered': len(drug_features_df),
    'drugs_found': sorted(drug_features_df['drug'].tolist()),
    'drugs_missing': sorted(list(set(dili_drugs) - set(drug_features_df['drug'].tolist())))
}

with open(results_dir / 'event_normalized_summary_final.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n✅ Efficient event-normalized extraction complete!")
print(f"   Covered {summary['n_drugs_covered']} / {len(dili_drugs)} DILI drugs")