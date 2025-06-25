#!/usr/bin/env python3
"""
Extract Event-Aware Features Using Improved Event Detection
Features extracted BETWEEN corrected and recovered media change events
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
print("EXTRACTING EVENT-AWARE FEATURES WITH IMPROVED EVENTS")
print("=" * 80)

# Load improved media change events
improved_events_df = pd.read_parquet(results_dir / "improved_media_change_events_expanded.parquet")
print(f"\n📊 Loaded {len(improved_events_df):,} improved media change events")
print(f"   Covering {improved_events_df['well_id'].nunique():,} wells")

# Load original events for comparison
original_events_df = pd.read_parquet(results_dir / "media_change_events.parquet")
print(f"📊 Original events: {len(original_events_df):,} records")

# Connect to database
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

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
            'median_o2': np.median(seg_values),
            'q25_o2': np.percentile(seg_values, 25),
            'q75_o2': np.percentile(seg_values, 75),
            'iqr_o2': np.percentile(seg_values, 75) - np.percentile(seg_values, 25)
        }
        
        # Consumption rate (linear fit)
        if len(seg_times) > 3:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(seg_times - seg_times[0], seg_values)
                seg_feat['consumption_rate'] = -slope  # Negative slope = consumption
                seg_feat['consumption_r2'] = r_value ** 2
                seg_feat['consumption_p_value'] = p_value
                seg_feat['baseline_o2'] = intercept
                seg_feat['consumption_stderr'] = std_err
                
                # Predicted vs actual variance
                predicted = intercept + slope * (seg_times - seg_times[0])
                seg_feat['explained_variance'] = 1 - np.var(seg_values - predicted) / np.var(seg_values)
            except:
                seg_feat['consumption_rate'] = np.nan
                seg_feat['consumption_r2'] = 0
                seg_feat['consumption_p_value'] = 1
                seg_feat['baseline_o2'] = np.mean(seg_values)
        
        # Temporal dynamics
        min_idx = np.argmin(seg_values)
        max_idx = np.argmax(seg_values)
        seg_feat['time_to_min'] = seg_times[min_idx] - start
        seg_feat['time_to_max'] = seg_times[max_idx] - start
        seg_feat['min_time_fraction'] = (seg_times[min_idx] - start) / (end - start)
        seg_feat['max_time_fraction'] = (seg_times[max_idx] - start) / (end - start)
        
        # Oxygen depletion characteristics
        if len(seg_values) > 5:
            # Early vs late comparison
            mid_point = len(seg_values) // 2
            early_mean = np.mean(seg_values[:mid_point])
            late_mean = np.mean(seg_values[mid_point:])
            seg_feat['early_late_ratio'] = late_mean / early_mean if early_mean > 0 else np.nan
            seg_feat['oxygen_decline'] = early_mean - late_mean
            
            # Variability in early vs late phases
            seg_feat['early_cv'] = np.std(seg_values[:mid_point]) / early_mean if early_mean > 0 else np.nan
            seg_feat['late_cv'] = np.std(seg_values[mid_point:]) / late_mean if late_mean > 0 else np.nan
        
        # Plateau detection (sustained periods at low oxygen)
        low_threshold = np.percentile(seg_values, 20)
        low_mask = seg_values <= low_threshold
        if low_mask.any():
            seg_feat['low_oxygen_fraction'] = np.sum(low_mask) / len(seg_values)
            seg_feat['longest_low_stretch'] = max([len(list(g)) for k, g in np.groupby(low_mask) if k])
        else:
            seg_feat['low_oxygen_fraction'] = 0
            seg_feat['longest_low_stretch'] = 0
        
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
                
                # Weighted standard deviation
                if len(weighted_values) > 1:
                    mean_val = features[f'{col}_mean']
                    weighted_var = sum(w * (v/w - mean_val)**2 for v, w in zip(weighted_values, weights)) / sum(weights)
                    features[f'{col}_std'] = np.sqrt(weighted_var)
                    features[f'{col}_cv'] = features[f'{col}_std'] / features[f'{col}_mean'] if features[f'{col}_mean'] > 0 else 0
                else:
                    features[f'{col}_std'] = 0
                    features[f'{col}_cv'] = 0
        
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
            
            # Segment-to-segment variability
            consumption_rates = [sf['consumption_rate'] for sf in segment_features if 'consumption_rate' in sf and not np.isnan(sf['consumption_rate'])]
            if len(consumption_rates) > 1:
                features['consumption_consistency'] = 1 / (1 + np.std(consumption_rates))
        
        # Overall experiment characteristics
        features['n_segments'] = len(segment_features)
        features['total_monitored_hours'] = sum(sf['duration_hours'] * sf['confidence_weight'] for sf in segment_features) / total_weight
        features['avg_confidence'] = total_weight / len(segment_features)
        features['total_weight'] = total_weight
    
    return features

# Get drug metadata for feature extraction
drug_query = """
SELECT DISTINCT 
    plate_id || '_' || well_number as well_id,
    plate_id,
    well_number,
    drug,
    concentration
FROM postgres.well_map_data 
WHERE drug IS NOT NULL
"""
drug_metadata = conn.execute(drug_query).fetchdf()

print(f"\n🧬 Drug metadata: {len(drug_metadata):,} wells with drug information")

# Process wells in batches
batch_size = 50  # Smaller batches for more complex processing
well_ids = improved_events_df['well_id'].unique()
all_features = []

print(f"\n🔄 Processing {len(well_ids):,} wells with improved events...")

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
        
        # Get improved events for this well
        well_events = improved_events_df[improved_events_df['well_id'] == well_id].copy()
        
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
            
            # Add drug metadata
            drug_info = drug_metadata[drug_metadata['well_id'] == well_id]
            if len(drug_info) > 0:
                features['drug'] = drug_info.iloc[0]['drug']
                features['concentration'] = drug_info.iloc[0]['concentration']
                features['plate_id'] = drug_info.iloc[0]['plate_id']
            else:
                # Skip wells without drug metadata
                continue
            
            features['n_events'] = len(well_events)
            features['n_high_conf_events'] = (well_events['confidence'] == 'high').sum()
            features['n_medium_conf_events'] = (well_events['confidence'] == 'medium').sum()
            features['n_low_conf_events'] = (well_events['confidence'] == 'low').sum()
            
            all_features.append(features)

# Create features dataframe
features_df = pd.DataFrame(all_features)

print(f"\n📊 IMPROVED EVENT-AWARE FEATURE RESULTS:")
print(f"   Wells with features: {len(features_df):,}")
print(f"   Drugs represented: {features_df['drug'].nunique()}")
print(f"   Feature columns: {len([c for c in features_df.columns if c not in ['well_id', 'drug', 'concentration', 'plate_id']])}")

# Compare with original features
original_features_df = pd.read_parquet(results_dir / "event_aware_features_wells.parquet")
print(f"\n📈 COMPARISON WITH ORIGINAL:")
print(f"   Original wells: {len(original_features_df):,}")
print(f"   Improved wells: {len(features_df):,}")
print(f"   Change: {len(features_df) - len(original_features_df):+,} wells ({(len(features_df)/len(original_features_df)-1)*100:+.1f}%)")

# Aggregate to drug level
print(f"\n🧬 Aggregating to drug level...")

# Group by drug and aggregate with proper handling of NaN values
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
features_df.to_parquet(results_dir / "improved_event_aware_features_wells.parquet", index=False)
drug_features.to_parquet(results_dir / "improved_event_aware_features_drugs.parquet", index=False)

print(f"\n💾 Saved improved features to:")
print(f"   Well-level: {results_dir / 'improved_event_aware_features_wells.parquet'}")
print(f"   Drug-level: {results_dir / 'improved_event_aware_features_drugs.parquet'}")

# Create comprehensive comparison visualization
print(f"\n📊 Creating comparison visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Original vs Improved Event-Aware Features', fontsize=16, fontweight='bold')

# Load original drug features for comparison
original_drug_features = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")

# Plot 1: Number of drugs comparison
ax = axes[0, 0]
comparison_data = [len(original_drug_features), len(drug_features)]
bars = ax.bar(['Original', 'Improved'], comparison_data, color=['lightblue', 'lightcoral'], alpha=0.7)
ax.set_ylabel('Number of Drugs')
ax.set_title('Drug Coverage Comparison')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, comparison_data):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{value}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Consumption rate comparison
ax = axes[0, 1]
if 'consumption_rate_mean_mean' in original_drug_features.columns and 'consumption_rate_mean_mean' in drug_features.columns:
    orig_rates = original_drug_features['consumption_rate_mean_mean'].dropna()
    impr_rates = drug_features['consumption_rate_mean_mean'].dropna()
    
    ax.hist(orig_rates, bins=20, alpha=0.5, label=f'Original (n={len(orig_rates)})', color='blue', density=True)
    ax.hist(impr_rates, bins=20, alpha=0.5, label=f'Improved (n={len(impr_rates)})', color='red', density=True)
    ax.set_xlabel('Mean Consumption Rate (%O₂/hour)')
    ax.set_ylabel('Density')
    ax.set_title('Consumption Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 3: Feature quality comparison (CV)
ax = axes[0, 2]
if 'cv_o2_mean_mean' in original_drug_features.columns and 'cv_o2_mean_mean' in drug_features.columns:
    orig_cv = original_drug_features['cv_o2_mean_mean'].dropna()
    impr_cv = drug_features['cv_o2_mean_mean'].dropna()
    
    ax.boxplot([orig_cv, impr_cv], labels=['Original', 'Improved'])
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Measurement Quality (CV)')
    ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Monitoring time comparison
ax = axes[1, 0]
if 'total_monitored_hours_mean' in original_drug_features.columns and 'total_monitored_hours_mean' in drug_features.columns:
    orig_hours = original_drug_features['total_monitored_hours_mean'].dropna()
    impr_hours = drug_features['total_monitored_hours_mean'].dropna()
    
    ax.scatter(range(len(orig_hours)), sorted(orig_hours), alpha=0.6, label='Original', color='blue')
    ax.scatter(range(len(impr_hours)), sorted(impr_hours), alpha=0.6, label='Improved', color='red')
    ax.set_xlabel('Drug Rank')
    ax.set_ylabel('Total Monitored Hours')
    ax.set_title('Clean Monitoring Time per Drug')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 5: Confidence distribution (new feature)
ax = axes[1, 1]
if 'avg_confidence_mean' in drug_features.columns:
    conf_values = drug_features['avg_confidence_mean'].dropna()
    ax.hist(conf_values, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(conf_values.median(), color='red', linestyle='--', 
               label=f'Median: {conf_values.median():.2f}')
    ax.set_xlabel('Average Event Confidence')
    ax.set_ylabel('Number of Drugs')
    ax.set_title('Event Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 6: Number of events comparison
ax = axes[1, 2]
if 'n_events_mean' in original_drug_features.columns and 'n_events_mean' in drug_features.columns:
    orig_events = original_drug_features['n_events_mean'].dropna()
    impr_events = drug_features['n_events_mean'].dropna()
    
    ax.scatter(orig_events, impr_events, alpha=0.6)
    ax.plot([0, max(orig_events.max(), impr_events.max())], 
            [0, max(orig_events.max(), impr_events.max())], 'r--', alpha=0.5)
    ax.set_xlabel('Original Events per Drug')
    ax.set_ylabel('Improved Events per Drug')
    ax.set_title('Events per Drug: Original vs Improved')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "original_vs_improved_features_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Feature statistics comparison
print(f"\n📈 DETAILED FEATURE COMPARISON:")

# Compare key metrics
metrics_comparison = {}

if 'consumption_rate_mean_mean' in original_drug_features.columns and 'consumption_rate_mean_mean' in drug_features.columns:
    orig_rate = original_drug_features['consumption_rate_mean_mean'].median()
    impr_rate = drug_features['consumption_rate_mean_mean'].median()
    metrics_comparison['consumption_rate_median'] = (orig_rate, impr_rate, impr_rate - orig_rate)

if 'cv_o2_mean_mean' in original_drug_features.columns and 'cv_o2_mean_mean' in drug_features.columns:
    orig_cv = original_drug_features['cv_o2_mean_mean'].median()
    impr_cv = drug_features['cv_o2_mean_mean'].median()
    metrics_comparison['cv_median'] = (orig_cv, impr_cv, impr_cv - orig_cv)

if 'total_monitored_hours_mean' in original_drug_features.columns and 'total_monitored_hours_mean' in drug_features.columns:
    orig_hours = original_drug_features['total_monitored_hours_mean'].median()
    impr_hours = drug_features['total_monitored_hours_mean'].median()
    metrics_comparison['monitored_hours_median'] = (orig_hours, impr_hours, impr_hours - orig_hours)

for metric, (orig, impr, diff) in metrics_comparison.items():
    pct_change = (diff / orig * 100) if orig != 0 else 0
    print(f"   {metric}: {orig:.3f} → {impr:.3f} ({diff:+.3f}, {pct_change:+.1f}%)")

print(f"\n✅ Improved event-aware feature extraction complete!")
print(f"   Enhanced with {len(drug_features) - len(original_drug_features)} additional drugs")
print(f"   Confidence-weighted features now available")

conn.close()