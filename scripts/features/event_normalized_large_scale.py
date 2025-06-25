#!/usr/bin/env python3
"""
Event-Normalized Time Features - Large Scale Batch Processing

PURPOSE:
    Efficient large-scale extraction of event-normalized features for all DILI-relevant drugs.
    Uses batch processing and optimized queries to handle the full dataset.

APPROACH:
    - Process drugs in batches to manage memory
    - Simplified feature extraction for speed
    - Focus on key event-normalized metrics
    - Parallel processing where beneficial
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from scipy import stats
import sys
from datetime import datetime
import json
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Configuration
EVENT_WINDOWS = {
    'immediate_post': (0, 6),      
    'early_post': (6, 12),         
    'late_post': (12, 24),         
}

RECOVERY_THRESHOLDS = [0.5, 0.9]

# Setup directories
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EVENT-NORMALIZED FEATURES - LARGE SCALE BATCH PROCESSING")
print("=" * 80)

# Define all DILI drugs
dili_drugs = [
    'Amiodarone', 'Busulfan', 'Imatinib', 'Lapatinib', 'Pazopanib', 'Regorafenib', 
    'Sorafenib', 'Sunitinib', 'Trametinib', 'Anastrozole', 'Axitinib', 'Cabozantinib',
    'Dabrafenib', 'Erlotinib', 'Gefitinib', 'Lenvatinib', 'Nilotinib', 'Osimertinib',
    'Vemurafenib', 'Alectinib', 'Binimetinib', 'Bortezomib', 'Ceritinib', 'Crizotinib',
    'Dasatinib', 'Everolimus', 'Ibrutinib', 'Ponatinib', 'Ruxolitinib', 'Alpelisib',
    'Ambrisentan', 'Buspirone', 'Dexamethasone', 'Fulvestrant', 'Letrozole',
    'Palbociclib', 'Ribociclib', 'Trastuzumab', 'Zoledronic acid'
]

# ========== SIMPLIFIED FEATURE EXTRACTION ==========

def detect_events_simple(oxygen_values, time_values):
    """Ultra-simple event detection based on variance spikes"""
    if len(oxygen_values) < 50:
        return []
    
    # Rolling variance
    rolling_var = pd.Series(oxygen_values).rolling(window=5, center=True).var()
    baseline_var = np.nanmedian(rolling_var[:50])
    
    if baseline_var == 0 or np.isnan(baseline_var):
        return []
    
    # Find spikes
    spike_indices = np.where(rolling_var > 3 * baseline_var)[0]
    
    # Group into events (at least 6 hours apart)
    events = []
    last_event_time = -10
    
    for idx in spike_indices:
        if idx < len(time_values) and time_values[idx] - last_event_time > 6:
            events.append(time_values[idx])
            last_event_time = time_values[idx]
    
    return events[:3]  # Max 3 events for consistency

def extract_features_fast(oxygen_values, time_values, well_id, drug, concentration):
    """Fast feature extraction focused on key metrics"""
    
    if len(oxygen_values) < 100:
        return None
    
    # Detect events
    events = detect_events_simple(oxygen_values, time_values)
    
    if len(events) < 2:
        return None
    
    # Baseline
    baseline_mask = time_values <= 48
    baseline_mean = np.mean(oxygen_values[baseline_mask]) if np.sum(baseline_mask) > 10 else np.mean(oxygen_values[:50])
    
    features = {
        'well_id': well_id,
        'drug': drug,
        'concentration': concentration,
        'n_events': len(events),
        'baseline_oxygen': baseline_mean
    }
    
    # Process each event
    for event_idx, event_time in enumerate(events):
        event_num = event_idx + 1
        
        # Get windows
        for window_name, (start_h, end_h) in EVENT_WINDOWS.items():
            window_mask = (time_values >= event_time + start_h) & (time_values < event_time + end_h)
            window_values = oxygen_values[window_mask]
            
            if len(window_values) >= 5:
                features[f'event_{event_num}_{window_name}_mean'] = np.mean(window_values)
                features[f'event_{event_num}_{window_name}_std'] = np.std(window_values)
                features[f'event_{event_num}_{window_name}_min'] = np.min(window_values)
                
                if window_name == 'immediate_post':
                    features[f'event_{event_num}_suppression'] = 1 - (np.min(window_values) / baseline_mean)
        
        # Recovery time (simplified)
        post_event_mask = time_values >= event_time
        post_event_values = oxygen_values[post_event_mask]
        post_event_times = time_values[post_event_mask]
        
        if len(post_event_values) > 10:
            for threshold in RECOVERY_THRESHOLDS:
                recovery_mask = post_event_values >= threshold * baseline_mean
                if np.any(recovery_mask):
                    recovery_idx = np.where(recovery_mask)[0][0]
                    features[f'event_{event_num}_time_to_{int(threshold*100)}pct'] = post_event_times[recovery_idx] - event_time
    
    # Event consistency
    if 'event_1_suppression' in features and 'event_2_suppression' in features:
        e1_supp = features['event_1_suppression']
        e2_supp = features['event_2_suppression']
        if e1_supp > 0 and e2_supp > 0:
            features['suppression_consistency'] = 1 - abs(e1_supp - e2_supp) / max(e1_supp, e2_supp)
    
    return features

# ========== BATCH PROCESSING ==========

def process_drug_batch(drug_list, loader):
    """Process a batch of drugs"""
    
    # Get wells for these drugs
    wells = loader.load_well_metadata()
    drug_wells = wells[wells['drug'].isin(drug_list)]
    
    if len(drug_wells) == 0:
        return []
    
    # Sample wells per drug (max 30 per drug for reasonable size)
    sampled_wells = []
    for drug in drug_list:
        dw = drug_wells[drug_wells['drug'] == drug]
        if len(dw) > 0:
            sample_size = min(30, len(dw))
            sampled_wells.append(dw.sample(n=sample_size, random_state=42))
    
    if not sampled_wells:
        return []
        
    sampled_wells_df = pd.concat(sampled_wells, ignore_index=True)
    
    # Get unique plates
    unique_plates = sampled_wells_df['plate_id'].unique()
    
    # Load data plate by plate
    all_features = []
    
    for plate_id in unique_plates:
        # Get wells for this plate
        plate_wells = sampled_wells_df[sampled_wells_df['plate_id'] == plate_id]
        
        # Load plate data
        try:
            plate_data = loader.load_oxygen_data(plate_ids=[str(plate_id)])
            
            # Rename o2 to oxygen if needed
            if 'o2' in plate_data.columns and 'oxygen' not in plate_data.columns:
                plate_data = plate_data.rename(columns={'o2': 'oxygen'})
            
            # Process each well
            for _, well_info in plate_wells.iterrows():
                well_id = well_info['well_id']
                well_data = plate_data[plate_data['well_id'] == well_id]
                
                if len(well_data) < 100:
                    continue
                
                # Sort by time
                well_data = well_data.sort_values('elapsed_hours')
                
                # Extract features
                features = extract_features_fast(
                    well_data['oxygen'].values,
                    well_data['elapsed_hours'].values,
                    well_id,
                    well_info['drug'],
                    well_info['concentration']
                )
                
                if features is not None:
                    all_features.append(features)
        
        except Exception as e:
            print(f"   Error processing plate {plate_id}: {e}")
            continue
    
    return all_features

# ========== MAIN PROCESSING ==========

print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize data loader
loader = DataLoader()

# Process drugs in batches
batch_size = 5  # Process 5 drugs at a time
all_well_features = []

print(f"\n📊 Processing {len(dili_drugs)} DILI-relevant drugs in batches of {batch_size}...")

for i in range(0, len(dili_drugs), batch_size):
    batch_drugs = dili_drugs[i:i+batch_size]
    print(f"\n🔄 Batch {i//batch_size + 1}/{(len(dili_drugs) + batch_size - 1)//batch_size}")
    print(f"   Drugs: {', '.join(batch_drugs)}")
    
    # Process batch
    batch_features = process_drug_batch(batch_drugs, loader)
    all_well_features.extend(batch_features)
    
    print(f"   Extracted features for {len(batch_features)} wells")
    
    # Clean up memory
    gc.collect()

print(f"\n✓ Total features extracted: {len(all_well_features)} wells")

# Convert to DataFrame
well_features_df = pd.DataFrame(all_well_features)

# ========== DRUG-LEVEL AGGREGATION ==========

print("\n🔄 Aggregating features at drug level...")

drug_features_list = []

for drug in dili_drugs:
    drug_data = well_features_df[well_features_df['drug'] == drug]
    
    if len(drug_data) == 0:
        continue
    
    drug_features = {
        'drug': drug,
        'n_wells': len(drug_data),
        'n_concentrations': drug_data['concentration'].nunique()
    }
    
    # Aggregate numeric features
    numeric_cols = [col for col in drug_data.columns if col not in ['well_id', 'drug', 'concentration']]
    
    for col in numeric_cols:
        values = drug_data[col].dropna()
        if len(values) > 0:
            drug_features[f'{col}_mean'] = values.mean()
            drug_features[f'{col}_std'] = values.std()
            drug_features[f'{col}_median'] = values.median()
            
            # Add percentiles for key features
            if any(key in col for key in ['suppression', 'time_to', 'consistency']):
                drug_features[f'{col}_p25'] = values.quantile(0.25)
                drug_features[f'{col}_p75'] = values.quantile(0.75)
    
    drug_features_list.append(drug_features)

drug_features_df = pd.DataFrame(drug_features_list)
print(f"   Created drug-level features for {len(drug_features_df)} drugs")

# ========== SAVE RESULTS ==========

print("\n💾 Saving results...")

# Save well-level features
well_features_df.to_parquet(results_dir / 'event_normalized_features_wells_large.parquet', index=False)
print(f"   Well-level: {results_dir / 'event_normalized_features_wells_large.parquet'}")

# Save drug-level features
drug_features_df.to_parquet(results_dir / 'event_normalized_features_drugs_large.parquet', index=False)
print(f"   Drug-level: {results_dir / 'event_normalized_features_drugs_large.parquet'}")

# Summary statistics
summary = {
    'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_wells_processed': len(well_features_df),
    'n_drugs_covered': len(drug_features_df),
    'drugs_processed': sorted(drug_features_df['drug'].unique().tolist()),
    'features_per_well': len([col for col in well_features_df.columns if col not in ['well_id', 'drug', 'concentration']]),
    'drug_coverage': {
        'requested': len(dili_drugs),
        'found': len(drug_features_df),
        'missing': sorted(list(set(dili_drugs) - set(drug_features_df['drug'].unique())))
    },
    'well_statistics': {
        'mean_wells_per_drug': well_features_df.groupby('drug').size().mean(),
        'min_wells_per_drug': well_features_df.groupby('drug').size().min(),
        'max_wells_per_drug': well_features_df.groupby('drug').size().max()
    }
}

with open(results_dir / 'event_normalized_summary_large.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n✅ Large-scale event-normalized feature extraction complete!")
print(f"\n📊 FINAL SUMMARY:")
print(f"   Processed {summary['n_wells_processed']:,} wells")
print(f"   Covered {summary['n_drugs_covered']} / {len(dili_drugs)} DILI-relevant drugs")
print(f"   Created {summary['features_per_well']} features per well")
print(f"   Mean wells per drug: {summary['well_statistics']['mean_wells_per_drug']:.1f}")

if summary['drug_coverage']['missing']:
    print(f"\n⚠️  Missing drugs: {', '.join(summary['drug_coverage']['missing'][:5])}")
    if len(summary['drug_coverage']['missing']) > 5:
        print(f"   ... and {len(summary['drug_coverage']['missing']) - 5} more")