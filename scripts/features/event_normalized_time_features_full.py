#!/usr/bin/env python3
"""
Event-Normalized Time Features - Full Scale

PURPOSE:
    Full-scale extraction of event-normalized time features for all available data.
    This version processes all plates to ensure good overlap with DILI classifications.

OPTIMIZATIONS:
    - Parallel processing of wells
    - Batch feature extraction
    - Memory-efficient processing
    - Progress tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.interpolate import interp1d
import pycatch22
from sklearn.preprocessing import StandardScaler
import sys
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
    'extended_post': (24, 48),     # 24-48h after event
    'pre_event': (-6, 0),          # 6h before next event
    'full_cycle': (0, -1)          # Full inter-event period
}

RECOVERY_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]  # Fraction of baseline recovered

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_normalized"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EVENT-NORMALIZED TIME FEATURES - FULL SCALE")
print("=" * 80)

# ========== EVENT DETECTION AND ALIGNMENT ==========

def detect_media_changes(well_data, control_events=None):
    """Detect media change events using variance spikes"""
    
    # Rename o2 to oxygen if needed
    if 'o2' in well_data.columns and 'oxygen' not in well_data.columns:
        well_data = well_data.rename(columns={'o2': 'oxygen'})
    
    # Calculate rolling variance
    well_data['rolling_var'] = well_data['oxygen'].rolling(
        window=5, center=True, min_periods=3
    ).var()
    
    # Find baseline variance
    baseline_var = well_data[well_data['elapsed_hours'] <= 48]['rolling_var'].median()
    
    if np.isnan(baseline_var) or baseline_var == 0:
        baseline_var = well_data['rolling_var'].median()
    
    # Detect spikes
    well_data['var_spike'] = well_data['rolling_var'] > (3 * baseline_var)
    
    # Group consecutive spikes
    events = []
    in_event = False
    event_start = None
    
    for idx, row in well_data.iterrows():
        if row['var_spike'] and not in_event:
            in_event = True
            event_start = idx
        elif not row['var_spike'] and in_event:
            in_event = False
            # Take middle of spike as event time
            event_idx = (event_start + idx) // 2
            events.append({
                'event_time': well_data.loc[event_idx, 'elapsed_hours'],
                'event_idx': event_idx
            })
    
    # If using control events, align to nearest control timing
    if control_events is not None and len(events) > 0:
        aligned_events = []
        for event in events:
            # Find nearest control event
            distances = [abs(event['event_time'] - ce) for ce in control_events]
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] < 6:  # Within 6 hours
                aligned_events.append({
                    'event_time': control_events[nearest_idx],
                    'event_idx': event['event_idx'],
                    'aligned': True
                })
            else:
                aligned_events.append(event)
        events = aligned_events
    
    return events

def create_event_normalized_timeseries(well_data, events):
    """Create time series with event-normalized time coordinates"""
    
    if len(events) == 0:
        return pd.DataFrame()
    
    normalized_segments = []
    
    for i, event in enumerate(events):
        # Get data around this event
        if i == 0:
            # First event: from start to halfway to next event
            start_time = 0
            end_time = events[i+1]['event_time'] if i+1 < len(events) else well_data['elapsed_hours'].max()
            end_time = event['event_time'] + (end_time - event['event_time']) / 2
        else:
            # Subsequent events: from halfway from previous to halfway to next
            start_time = events[i-1]['event_time'] + (event['event_time'] - events[i-1]['event_time']) / 2
            if i+1 < len(events):
                end_time = event['event_time'] + (events[i+1]['event_time'] - event['event_time']) / 2
            else:
                end_time = well_data['elapsed_hours'].max()
        
        # Extract segment
        segment = well_data[
            (well_data['elapsed_hours'] >= start_time) & 
            (well_data['elapsed_hours'] <= end_time)
        ].copy()
        
        if len(segment) == 0:
            continue
        
        # Add event-normalized time
        segment['event_number'] = i + 1
        segment['hours_since_event'] = segment['elapsed_hours'] - event['event_time']
        
        # Time to next event
        if i+1 < len(events):
            segment['hours_to_next_event'] = events[i+1]['event_time'] - segment['elapsed_hours']
        else:
            segment['hours_to_next_event'] = np.nan
        
        normalized_segments.append(segment)
    
    if normalized_segments:
        return pd.concat(normalized_segments, ignore_index=True)
    else:
        return pd.DataFrame()

# ========== FEATURE EXTRACTION ==========

def extract_window_features(data, oxygen_col='oxygen'):
    """Extract features from a time window"""
    
    if len(data) < 5:
        return {}
    
    oxygen_values = data[oxygen_col].values
    
    features = {
        'mean': np.mean(oxygen_values),
        'std': np.std(oxygen_values),
        'cv': np.std(oxygen_values) / np.mean(oxygen_values) if np.mean(oxygen_values) != 0 else 0,
        'min': np.min(oxygen_values),
        'max': np.max(oxygen_values),
        'range': np.max(oxygen_values) - np.min(oxygen_values),
        'slope': 0,
        'n_points': len(oxygen_values)
    }
    
    # Linear trend
    if len(oxygen_values) >= 3:
        x = np.arange(len(oxygen_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, oxygen_values)
        features['slope'] = slope
        features['trend_r2'] = r_value**2
        features['trend_p'] = p_value
    
    # Catch22 features if enough points
    if len(oxygen_values) >= 10:
        try:
            catch22_results = pycatch22.catch22_all(oxygen_values)
            for i, name in enumerate(catch22_results['names']):
                features[f'catch22_{name}'] = catch22_results['values'][i]
        except:
            pass
    
    return features

def extract_recovery_metrics(post_event_data, baseline_value):
    """Extract recovery trajectory metrics"""
    
    if len(post_event_data) < 5 or baseline_value == 0:
        return {}
    
    metrics = {}
    oxygen_values = post_event_data['oxygen'].values
    time_values = post_event_data['hours_since_event'].values
    
    # Normalize to baseline
    normalized_values = oxygen_values / baseline_value
    
    # Time to recover to various thresholds
    for threshold in RECOVERY_THRESHOLDS:
        recovery_idx = np.where(normalized_values >= threshold)[0]
        if len(recovery_idx) > 0:
            metrics[f'time_to_{int(threshold*100)}pct_recovery'] = time_values[recovery_idx[0]]
        else:
            metrics[f'time_to_{int(threshold*100)}pct_recovery'] = np.nan
    
    # Recovery rate (slope of first 6 hours)
    early_data = post_event_data[post_event_data['hours_since_event'] <= 6]
    if len(early_data) >= 3:
        x = early_data['hours_since_event'].values
        y = early_data['oxygen'].values
        slope, _, _, _, _ = stats.linregress(x, y)
        metrics['early_recovery_rate'] = slope / baseline_value  # Normalized rate
    
    # Maximum suppression
    metrics['max_suppression'] = 1 - (np.min(normalized_values))
    metrics['max_suppression_time'] = time_values[np.argmin(normalized_values)]
    
    return metrics

def extract_event_normalized_features(well_data, well_id, drug, concentration):
    """Extract all event-normalized features for a well"""
    
    # Detect events
    events = detect_media_changes(well_data)
    
    if len(events) == 0:
        return None
    
    # Create event-normalized timeseries
    normalized_data = create_event_normalized_timeseries(well_data, events)
    
    if len(normalized_data) == 0:
        return None
    
    # Get baseline reference
    baseline_data = well_data[well_data['elapsed_hours'] <= 48]
    baseline_value = baseline_data['oxygen'].mean() if len(baseline_data) > 0 else well_data['oxygen'].mean()
    
    all_features = {
        'well_id': well_id,
        'drug': drug,
        'concentration': concentration,
        'n_events': len(events),
        'baseline_oxygen': baseline_value
    }
    
    # Extract features for each event
    event_features_list = []
    
    for event_num in normalized_data['event_number'].unique():
        event_data = normalized_data[normalized_data['event_number'] == event_num]
        
        event_features = {f'event_{event_num}_present': 1}
        
        # Window-based features
        for window_name, (start_h, end_h) in EVENT_WINDOWS.items():
            if window_name == 'pre_event' and pd.isna(event_data['hours_to_next_event'].iloc[0]):
                continue  # Skip pre-event for last event
            
            if window_name == 'pre_event':
                # Use hours_to_next_event for pre-event window
                window_data = event_data[
                    (event_data['hours_to_next_event'] >= 0) & 
                    (event_data['hours_to_next_event'] <= 6)
                ]
            elif window_name == 'full_cycle':
                window_data = event_data
            else:
                # Post-event windows
                window_data = event_data[
                    (event_data['hours_since_event'] >= start_h) & 
                    (event_data['hours_since_event'] < end_h)
                ]
            
            if len(window_data) >= 5:
                window_features = extract_window_features(window_data)
                for feat_name, feat_val in window_features.items():
                    event_features[f'event_{event_num}_{window_name}_{feat_name}'] = feat_val
        
        # Recovery metrics for post-event data
        post_event_data = event_data[event_data['hours_since_event'] >= 0]
        if len(post_event_data) >= 5:
            recovery_metrics = extract_recovery_metrics(post_event_data, baseline_value)
            for metric_name, metric_val in recovery_metrics.items():
                event_features[f'event_{event_num}_{metric_name}'] = metric_val
        
        event_features_list.append(event_features)
    
    # Aggregate across events
    if event_features_list:
        # Convert to DataFrame for easier aggregation
        event_df = pd.DataFrame(event_features_list)
        
        # Calculate mean, std, cv across events for each feature
        for col in event_df.columns:
            if col.startswith('event_') and '_present' not in col:
                values = event_df[col].dropna()
                if len(values) > 0:
                    all_features[f'{col}_mean'] = values.mean()
                    if len(values) > 1:
                        all_features[f'{col}_std'] = values.std()
                        all_features[f'{col}_cv'] = values.std() / values.mean() if values.mean() != 0 else 0
        
        # Event consistency metrics
        if len(event_features_list) > 1:
            # Check consistency of key features across events
            consistency_features = ['immediate_post_mean', 'early_recovery_rate', 'max_suppression']
            
            for feat in consistency_features:
                feat_cols = [col for col in event_df.columns if feat in col]
                if feat_cols:
                    values_matrix = event_df[feat_cols].values
                    # Calculate pairwise correlations between events
                    if values_matrix.shape[1] > 1:
                        corr_matrix = np.corrcoef(values_matrix.T)
                        # Take mean of off-diagonal elements
                        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                        if mask.sum() > 0:
                            all_features[f'{feat}_event_consistency'] = np.nanmean(corr_matrix[mask])
    
    return all_features

# ========== PARALLEL PROCESSING FUNCTION ==========

def process_well_batch(batch_data):
    """Process a batch of wells (for parallel processing)"""
    wells, df = batch_data
    results = []
    
    for idx, well_info in wells.iterrows():
        well_id = well_info['well_id']
        well_data = df[df['well_id'] == well_id].copy()
        
        # Sort by time
        well_data = well_data.sort_values('elapsed_hours').reset_index(drop=True)
        
        # Extract features
        features = extract_event_normalized_features(
            well_data, 
            well_id,
            well_info['drug'],
            well_info['concentration']
        )
        
        if features is not None:
            results.append(features)
    
    return results

# ========== MAIN PROCESSING ==========

print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📊 Loading oxygen consumption data...")
loader = DataLoader()

# Load ALL data
print("   Loading full dataset...")
df = loader.load_oxygen_data(limit=None)  # No limit - process all data

# Rename o2 to oxygen if needed
if 'o2' in df.columns and 'oxygen' not in df.columns:
    df = df.rename(columns={'o2': 'oxygen'})

print(f"   Loaded {len(df):,} oxygen measurements")

# Get unique wells
wells = df.groupby('well_id').agg({
    'plate_id': 'first',
    'drug': 'first', 
    'concentration': 'first'
}).reset_index()

print(f"   Found {len(wells):,} unique wells")

# Focus on drugs with DILI data
dili_drugs = ['Amiodarone', 'Busulfan', 'Imatinib', 'Lapatinib', 'Pazopanib', 'Regorafenib', 
              'Sorafenib', 'Sunitinib', 'Trametinib', 'Anastrozole', 'Axitinib', 'Cabozantinib',
              'Dabrafenib', 'Erlotinib', 'Gefitinib', 'Lenvatinib', 'Nilotinib', 'Osimertinib',
              'Vemurafenib', 'Alectinib', 'Binimetinib', 'Bortezomib', 'Ceritinib', 'Crizotinib',
              'Dasatinib', 'Everolimus', 'Ibrutinib', 'Ponatinib', 'Ruxolitinib', 'Alpelisib',
              'Ambrisentan', 'Buspirone', 'Dexamethasone', 'Fulvestrant', 'Letrozole',
              'Palbociclib', 'Ribociclib', 'Trastuzumab', 'Zoledronic acid']

dili_wells = wells[wells['drug'].isin(dili_drugs)]
print(f"   Focusing on {len(dili_wells):,} wells with DILI-relevant drugs")

# Prepare for parallel processing
n_cores = max(1, multiprocessing.cpu_count() - 1)
print(f"\n🚀 Starting parallel processing with {n_cores} cores...")

# Split wells into batches
batch_size = max(10, len(dili_wells) // (n_cores * 10))  # Aim for ~10 batches per core
well_batches = []

for i in range(0, len(dili_wells), batch_size):
    batch = dili_wells.iloc[i:i+batch_size]
    # Only include relevant data for this batch
    well_ids = batch['well_id'].values
    batch_df = df[df['well_id'].isin(well_ids)]
    well_batches.append((batch, batch_df))

print(f"   Created {len(well_batches)} batches (~{batch_size} wells each)")

# Process in parallel
all_features = []
completed_batches = 0

with ProcessPoolExecutor(max_workers=n_cores) as executor:
    # Submit all batches
    future_to_batch = {executor.submit(process_well_batch, batch): i 
                       for i, batch in enumerate(well_batches)}
    
    # Process completed batches
    for future in as_completed(future_to_batch):
        batch_idx = future_to_batch[future]
        try:
            batch_results = future.result()
            all_features.extend(batch_results)
            completed_batches += 1
            
            if completed_batches % 10 == 0:
                print(f"   Processed {completed_batches}/{len(well_batches)} batches...")
                
        except Exception as e:
            print(f"   Error in batch {batch_idx}: {e}")

print(f"\n✓ Extracted features for {len(all_features):,} wells")

# Convert to DataFrame
well_features_df = pd.DataFrame(all_features)

# ========== DRUG-LEVEL AGGREGATION ==========

print("\n🔄 Aggregating features at drug level...")

# Group by drug and concentration
drug_features_list = []

for (drug, conc), group in well_features_df.groupby(['drug', 'concentration']):
    drug_features = {
        'drug': drug,
        'concentration': conc,
        'n_wells': len(group)
    }
    
    # Aggregate numeric features
    numeric_cols = [col for col in group.columns if col not in ['well_id', 'drug', 'concentration']]
    
    for col in numeric_cols:
        values = group[col].dropna()
        if len(values) > 0:
            drug_features[f'{col}_mean'] = values.mean()
            drug_features[f'{col}_std'] = values.std()
            if len(values) >= 4:  # CV only if 4 replicates
                drug_features[f'{col}_cv'] = values.std() / values.mean() if values.mean() != 0 else 0
    
    drug_features_list.append(drug_features)

drug_features_df = pd.DataFrame(drug_features_list)
print(f"   Created drug-level features for {len(drug_features_df):,} drug-concentration combinations")
print(f"   Unique drugs: {drug_features_df['drug'].nunique()}")

# ========== SAVE RESULTS ==========

print("\n💾 Saving event-normalized features...")

# Save well-level features
well_features_df.to_parquet(results_dir / 'event_normalized_features_wells_full.parquet', index=False)
print(f"   Well-level features: {results_dir / 'event_normalized_features_wells_full.parquet'}")

# Save drug-level features
drug_features_df.to_parquet(results_dir / 'event_normalized_features_drugs_full.parquet', index=False)
print(f"   Drug-level features: {results_dir / 'event_normalized_features_drugs_full.parquet'}")

# Feature summary
feature_summary = {
    'n_wells_processed': len(all_features),
    'n_drugs': len(drug_features_df['drug'].unique()),
    'n_features_per_well': len([col for col in well_features_df.columns if col not in ['well_id', 'drug', 'concentration']]),
    'drugs_covered': sorted(drug_features_df['drug'].unique().tolist()),
    'event_windows': EVENT_WINDOWS,
    'recovery_thresholds': RECOVERY_THRESHOLDS,
    'feature_categories': {
        'window_features': len([col for col in well_features_df.columns if any(w in col for w in EVENT_WINDOWS.keys())]),
        'recovery_metrics': len([col for col in well_features_df.columns if 'recovery' in col or 'suppression' in col]),
        'consistency_metrics': len([col for col in well_features_df.columns if 'consistency' in col]),
        'catch22_features': len([col for col in well_features_df.columns if 'catch22' in col])
    }
}

with open(results_dir / 'event_normalized_features_summary_full.json', 'w') as f:
    json.dump(feature_summary, f, indent=2)

print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n✅ Full-scale event-normalized feature extraction complete!")
print(f"\n📊 SUMMARY:")
print(f"   Processed {feature_summary['n_wells_processed']:,} wells")
print(f"   Created {feature_summary['n_features_per_well']:,} features per well")
print(f"   Covered {feature_summary['n_drugs']} unique drugs")
print(f"\n🎯 Key features extracted:")
print(f"   - Post-event recovery trajectories (0-48h)")
print(f"   - Pre-event anticipation patterns (-6-0h)")
print(f"   - Recovery time to baseline thresholds")
print(f"   - Event response consistency metrics")
print(f"   - Event-aligned catch22 features")