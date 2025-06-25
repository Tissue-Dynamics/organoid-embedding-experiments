#!/usr/bin/env python3
"""
Extract Event-Aware Features Using Improved Events - Simplified Approach
Use original events dataset format but with improved timing corrections
"""

import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import os
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "improved_event_features"
fig_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("=" * 80)
print("EXTRACTING EVENT-AWARE FEATURES WITH IMPROVED TIMING")
print("=" * 80)

# Load original events and timing corrections
original_events_df = pd.read_parquet(results_dir / "media_change_events.parquet")
timing_corrections_df = pd.read_parquet(results_dir / "event_timing_corrections.parquet")
missing_events_df = pd.read_parquet(results_dir / "recovered_missing_events.parquet")

print(f"\n📊 Loaded datasets:")
print(f"   Original events: {len(original_events_df):,} records")
print(f"   Timing corrections: {len(timing_corrections_df):,} corrections")
print(f"   Missing events: {len(missing_events_df):,} recovered events")

# Apply timing corrections to original events
corrected_events_df = original_events_df.copy()

# Create plate-level correction mapping
plate_corrections = {}
for _, corr in timing_corrections_df.iterrows():
    plate_id = corr['plate_id']
    if plate_id not in plate_corrections:
        plate_corrections[plate_id] = {}
    
    # Store correction for each original time
    orig_time = corr['original_time']
    corr_time = corr['corrected_time']
    plate_corrections[plate_id][orig_time] = corr_time

print(f"\n🔧 Applying {len(timing_corrections_df)} timing corrections...")

# Apply corrections
corrections_applied = 0
for i, event in corrected_events_df.iterrows():
    plate_id = event['plate_id']
    event_time = event['event_time_hours']
    
    if plate_id in plate_corrections and event_time in plate_corrections[plate_id]:
        new_time = plate_corrections[plate_id][event_time]
        corrected_events_df.loc[i, 'event_time_hours'] = new_time
        corrected_events_df.loc[i, 'event_type'] = 'corrected'
        corrections_applied += 1

corrected_events_df['confidence'] = 'high'  # Original events are high confidence

print(f"   Applied {corrections_applied} timing corrections")

# Add missing events (create well-level records for missing events)
if len(missing_events_df) > 0:
    print(f"\n➕ Adding {len(missing_events_df)} missing events...")
    
    # Connect to database to get well mapping
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")
    
    additional_events = []
    
    for _, missing in missing_events_df.iterrows():
        plate_id = missing['plate_id']
        missing_time = missing['missing_event_time']
        
        # Get wells for this plate
        wells_query = f"""
        SELECT DISTINCT 
            plate_id || '_' || well_number as well_id,
            well_number,
            plate_id
        FROM postgres.processed_data
        WHERE plate_id = '{plate_id}'
          AND is_excluded = false
        """
        
        try:
            wells_data = conn.execute(wells_query).fetchdf()
            
            for _, well in wells_data.iterrows():
                # Get drug info from original events
                well_drug_info = original_events_df[
                    original_events_df['well_id'] == well['well_id']
                ]
                
                if len(well_drug_info) > 0:
                    drug_info = well_drug_info.iloc[0]
                    
                    # Confidence based on spike height
                    spike_height = missing['spike_height']
                    if spike_height > 30:
                        confidence = 'high'
                    elif spike_height > 15:
                        confidence = 'medium'
                    else:
                        confidence = 'low'
                    
                    additional_events.append({
                        'plate_id': plate_id,
                        'well_id': well['well_id'],
                        'well_number': well['well_number'],
                        'event_time_hours': missing_time,
                        'drug': drug_info['drug'],
                        'concentration': drug_info['concentration'],
                        'event_type': 'missing_recovered',
                        'confidence': confidence
                    })
        except Exception as e:
            print(f"   Error processing missing events for plate {plate_id}: {e}")
            continue
    
    conn.close()
    
    if additional_events:
        additional_events_df = pd.DataFrame(additional_events)
        
        # Combine with corrected events
        enhanced_events_df = pd.concat([corrected_events_df, additional_events_df], ignore_index=True)
        print(f"   Added {len(additional_events_df)} missing event records")
    else:
        enhanced_events_df = corrected_events_df
else:
    enhanced_events_df = corrected_events_df

print(f"\n📊 Enhanced events dataset:")
print(f"   Total events: {len(enhanced_events_df):,}")
print(f"   Wells covered: {enhanced_events_df['well_id'].nunique():,}")
print(f"   Event types: {enhanced_events_df['event_type'].value_counts().to_dict()}")
print(f"   Confidence levels: {enhanced_events_df['confidence'].value_counts().to_dict()}")

# Now extract features using the enhanced events
print(f"\n🔄 Extracting event-aware features with enhanced events...")

# Function to extract features between events (enhanced version)
def extract_inter_event_features(times, values, event_times, confidence_scores=None):
    """Extract features between media change events with confidence weighting"""
    
    features = {}
    event_times = sorted(event_times)
    
    if len(event_times) == 0:
        return features
    
    # Create segments between events
    segments = []
    
    # Pre-treatment segment (before first event)
    if event_times[0] > times[0] + 12:  # Need at least 12h before first event
        segments.append((times[0], event_times[0], 'pre_treatment', 1.0))
    
    # Inter-event segments
    for i in range(len(event_times) - 1):
        # Skip first 6 hours after media change (spike recovery)
        start = event_times[i] + 6
        end = event_times[i + 1]
        
        # Weight by confidence of both events
        conf_weight = 1.0
        if confidence_scores is not None and len(confidence_scores) > i + 1:
            conf1 = 1.0 if confidence_scores[i] == 'high' else 0.7 if confidence_scores[i] == 'medium' else 0.4
            conf2 = 1.0 if confidence_scores[i+1] == 'high' else 0.7 if confidence_scores[i+1] == 'medium' else 0.4
            conf_weight = (conf1 + conf2) / 2
        
        if end - start > 12:  # Need at least 12 hours of clean data
            segments.append((start, end, f'segment_{i+1}', conf_weight))
    
    # Final segment (after last event)
    if len(event_times) > 0 and times[-1] - event_times[-1] > 18:
        final_weight = 1.0
        if confidence_scores is not None and len(confidence_scores) > 0:
            final_weight = 1.0 if confidence_scores[-1] == 'high' else 0.7 if confidence_scores[-1] == 'medium' else 0.4
        segments.append((event_times[-1] + 6, times[-1], 'final_segment', final_weight))
    
    # Extract features for each segment
    segment_features = []
    
    for start, end, segment_name, weight in segments:
        mask = (times >= start) & (times <= end)
        if mask.sum() < 10:
            continue
            
        seg_times = times[mask]
        seg_values = values[mask]
        
        # Basic statistics
        seg_feat = {
            'segment': segment_name,
            'duration_hours': end - start,
            'n_points': len(seg_values),
            'confidence_weight': weight,
            'mean_o2': np.mean(seg_values),
            'std_o2': np.std(seg_values),
            'cv_o2': np.std(seg_values) / np.mean(seg_values) if np.mean(seg_values) > 0 else np.nan,
            'min_o2': np.min(seg_values),
            'max_o2': np.max(seg_values),
            'range_o2': np.max(seg_values) - np.min(seg_values),
            'median_o2': np.median(seg_values)
        }
        
        # Consumption rate (linear fit)
        if len(seg_times) > 3:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(seg_times - seg_times[0], seg_values)
                seg_feat['consumption_rate'] = -slope  # Negative slope = consumption
                seg_feat['consumption_r2'] = r_value ** 2
                seg_feat['baseline_o2'] = intercept
            except:
                seg_feat['consumption_rate'] = np.nan
                seg_feat['consumption_r2'] = 0
                seg_feat['baseline_o2'] = np.mean(seg_values)
        
        # Temporal dynamics
        min_idx = np.argmin(seg_values)
        seg_feat['time_to_min'] = seg_times[min_idx] - start
        seg_feat['min_time_fraction'] = (seg_times[min_idx] - start) / (end - start)
        
        segment_features.append(seg_feat)
    
    # Aggregate across segments with confidence weighting
    if segment_features:
        # Weighted means
        total_weight = sum(sf['confidence_weight'] for sf in segment_features)
        numeric_cols = [col for col in segment_features[0].keys() 
                       if col not in ['segment', 'confidence_weight'] and not np.isnan(segment_features[0][col])]
        
        for col in numeric_cols:
            weighted_values = []
            weights = []
            
            for sf in segment_features:
                if not np.isnan(sf.get(col, np.nan)):
                    weighted_values.append(sf[col] * sf['confidence_weight'])
                    weights.append(sf['confidence_weight'])
            
            if weighted_values and sum(weights) > 0:
                features[f'{col}_mean'] = sum(weighted_values) / sum(weights)
                
                if len(weighted_values) > 1:
                    mean_val = features[f'{col}_mean']
                    weighted_var = sum(w * (v/w - mean_val)**2 for v, w in zip(weighted_values, weights)) / sum(weights)
                    features[f'{col}_std'] = np.sqrt(weighted_var)
                else:
                    features[f'{col}_std'] = 0
        
        # Temporal progression analysis
        if len(segment_features) > 1:
            # Early vs late consumption rates
            early_segments = [sf for sf in segment_features[:2] if 'consumption_rate' in sf]
            late_segments = [sf for sf in segment_features[-2:] if 'consumption_rate' in sf]
            
            if early_segments and late_segments:
                early_rates = [sf['consumption_rate'] for sf in early_segments if not np.isnan(sf['consumption_rate'])]
                late_rates = [sf['consumption_rate'] for sf in late_segments if not np.isnan(sf['consumption_rate'])]
                
                if early_rates and late_rates:
                    features['consumption_acceleration'] = np.mean(late_rates) - np.mean(early_rates)
                    features['consumption_ratio'] = np.mean(late_rates) / np.mean(early_rates) if np.mean(early_rates) != 0 else np.nan
        
        # Overall experiment characteristics
        features['n_segments'] = len(segment_features)
        features['total_monitored_hours'] = sum(sf['duration_hours'] * sf['confidence_weight'] for sf in segment_features) / total_weight
        features['avg_confidence'] = total_weight / len(segment_features)
    
    return features

# Connect to database for time series data
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

# Process wells in batches
batch_size = 50
well_ids = enhanced_events_df['well_id'].unique()
all_features = []

print(f"\n🔄 Processing {len(well_ids):,} wells...")

for batch_start in tqdm(range(0, len(well_ids), batch_size), desc="Processing batches"):
    batch_wells = well_ids[batch_start:batch_start + batch_size]
    
    # Parse well_ids to get plate_id and well_number
    well_parts = []
    for w in batch_wells:
        parts = w.split('_')
        if len(parts) >= 2:
            plate_id = '_'.join(parts[:-1])
            try:
                well_number = int(parts[-1])
                well_parts.append((plate_id, well_number, w))
            except ValueError:
                continue
    
    if not well_parts:
        continue
    
    # Build query conditions
    conditions = []
    for plate_id, well_number, well_id in well_parts:
        conditions.append(f"(plate_id = '{plate_id}' AND well_number = {well_number})")
    
    where_clause = " OR ".join(conditions)
    
    ts_query = f"""
    SELECT 
        plate_id || '_' || well_number as well_id,
        timestamp,
        median_o2 as oxygen
    FROM postgres.processed_data
    WHERE ({where_clause})
      AND is_excluded = false
    ORDER BY plate_id, well_number, timestamp
    """
    
    try:
        ts_data = conn.execute(ts_query).fetchdf()
    except Exception as e:
        print(f"\nError in batch {batch_start}: {e}")
        continue
    
    if len(ts_data) == 0:
        continue
    
    # Process each well
    for well_id in batch_wells:
        well_data = ts_data[ts_data['well_id'] == well_id].copy()
        
        if len(well_data) < 50:
            continue
        
        # Get enhanced events for this well
        well_events = enhanced_events_df[enhanced_events_df['well_id'] == well_id].copy()
        
        if len(well_events) == 0:
            continue
        
        # Sort events by time
        well_events = well_events.sort_values('event_time_hours')
        
        # Calculate time in hours
        well_data['hours'] = (well_data['timestamp'] - well_data['timestamp'].min()).dt.total_seconds() / 3600
        
        # Extract features with confidence weighting
        features = extract_inter_event_features(
            well_data['hours'].values,
            well_data['oxygen'].values,
            well_events['event_time_hours'].values,
            well_events['confidence'].values
        )
        
        if features:
            features['well_id'] = well_id
            features['drug'] = well_events.iloc[0]['drug']
            features['concentration'] = well_events.iloc[0]['concentration']
            features['plate_id'] = well_events.iloc[0]['plate_id']
            features['n_events'] = len(well_events)
            features['n_corrected_events'] = (well_events['event_type'] == 'corrected').sum()
            features['n_missing_recovered'] = (well_events['event_type'] == 'missing_recovered').sum()
            features['n_high_conf_events'] = (well_events['confidence'] == 'high').sum()
            features['n_medium_conf_events'] = (well_events['confidence'] == 'medium').sum()
            features['n_low_conf_events'] = (well_events['confidence'] == 'low').sum()
            
            all_features.append(features)

conn.close()

# Create features dataframe
features_df = pd.DataFrame(all_features)

print(f"\n📊 ENHANCED EVENT-AWARE FEATURE RESULTS:")
print(f"   Wells with features: {len(features_df):,}")
print(f"   Drugs represented: {features_df['drug'].nunique()}")
print(f"   Feature columns: {len([c for c in features_df.columns if c not in ['well_id', 'drug', 'concentration', 'plate_id']])}")

# Aggregate to drug level
print(f"\n🧬 Aggregating to drug level...")

numeric_cols = [col for col in features_df.columns 
               if col not in ['well_id', 'drug', 'concentration', 'plate_id'] 
               and features_df[col].dtype in ['float64', 'int64']]

drug_features = features_df.groupby('drug')[numeric_cols].agg(['mean', 'std', 'count']).reset_index()

# Flatten column names
drug_features.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in drug_features.columns.values]

# Remove drugs with insufficient data
min_wells_per_drug = 3
drug_features = drug_features[drug_features['n_events_count'] >= min_wells_per_drug]

print(f"   Drug-level features: {len(drug_features)} drugs (≥{min_wells_per_drug} wells each)")

# Save results
features_df.to_parquet(results_dir / "enhanced_event_aware_features_wells.parquet", index=False)
drug_features.to_parquet(results_dir / "enhanced_event_aware_features_drugs.parquet", index=False)

print(f"\n💾 Saved enhanced features to:")
print(f"   Well-level: {results_dir / 'enhanced_event_aware_features_wells.parquet'}")
print(f"   Drug-level: {results_dir / 'enhanced_event_aware_features_drugs.parquet'}")

print(f"\n✅ Enhanced event-aware feature extraction complete!")
print(f"   Features now include timing corrections and recovered missing events")
print(f"   Confidence weighting applied to improve feature quality")