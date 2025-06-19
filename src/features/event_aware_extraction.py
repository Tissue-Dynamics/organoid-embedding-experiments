#!/usr/bin/env python3
"""
Extract Event-Aware Features for Drug Analysis
Features extracted BETWEEN media change events to avoid artifacts
"""

import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
import os
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_aware_features"
fig_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("=" * 80)
print("EXTRACTING EVENT-AWARE FEATURES")
print("=" * 80)

# Load media change events
events_df = pd.read_parquet(results_dir / "media_change_events.parquet")
print(f"\n📊 Loaded {len(events_df):,} media change events")
print(f"   Covering {events_df['well_id'].nunique():,} wells")

# Connect to database
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

# Function to extract features between events
def extract_inter_event_features(times, values, event_times):
    """Extract features between media change events"""
    
    features = {}
    event_times = sorted(event_times)
    
    # Add start and end times
    segments = []
    if event_times[0] > times[0]:
        segments.append((times[0], event_times[0], 'pre_treatment'))
    
    for i in range(len(event_times) - 1):
        # Skip first 6 hours after media change (spike recovery)
        start = event_times[i] + 6
        end = event_times[i + 1]
        if end - start > 12:  # Need at least 12 hours of data
            segments.append((start, end, f'segment_{i+1}'))
    
    # Last segment
    if len(event_times) > 0 and times[-1] - event_times[-1] > 18:
        segments.append((event_times[-1] + 6, times[-1], 'final_segment'))
    
    # Extract features for each segment
    segment_features = []
    
    for start, end, segment_name in segments:
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
            'mean_o2': np.mean(seg_values),
            'std_o2': np.std(seg_values),
            'cv_o2': np.std(seg_values) / np.mean(seg_values) if np.mean(seg_values) > 0 else np.nan,
            'min_o2': np.min(seg_values),
            'max_o2': np.max(seg_values),
            'range_o2': np.max(seg_values) - np.min(seg_values)
        }
        
        # Consumption rate (linear fit)
        if len(seg_times) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(seg_times - seg_times[0], seg_values)
            seg_feat['consumption_rate'] = -slope  # Negative slope = consumption
            seg_feat['consumption_r2'] = r_value ** 2
            seg_feat['baseline_o2'] = intercept
        
        # Time to minimum
        min_idx = np.argmin(seg_values)
        seg_feat['time_to_min'] = seg_times[min_idx] - start
        seg_feat['min_o2_fraction'] = (end - seg_times[min_idx]) / (end - start)
        
        segment_features.append(seg_feat)
    
    # Aggregate across segments
    if segment_features:
        # Mean features across all segments
        numeric_cols = [col for col in segment_features[0].keys() if col != 'segment']
        for col in numeric_cols:
            values = [sf[col] for sf in segment_features if not np.isnan(sf[col])]
            if values:
                features[f'{col}_mean'] = np.mean(values)
                features[f'{col}_std'] = np.std(values) if len(values) > 1 else 0
                features[f'{col}_cv'] = features[f'{col}_std'] / features[f'{col}_mean'] if features[f'{col}_mean'] > 0 else 0
        
        # Temporal progression
        if len(segment_features) > 1:
            # Early vs late consumption
            early_consumption = [sf['consumption_rate'] for sf in segment_features[:2] if 'consumption_rate' in sf]
            late_consumption = [sf['consumption_rate'] for sf in segment_features[-2:] if 'consumption_rate' in sf]
            
            if early_consumption and late_consumption:
                features['consumption_change'] = np.mean(late_consumption) - np.mean(early_consumption)
                features['consumption_ratio'] = np.mean(late_consumption) / np.mean(early_consumption) if np.mean(early_consumption) != 0 else np.nan
        
        features['n_segments'] = len(segment_features)
        features['total_monitored_hours'] = sum(sf['duration_hours'] for sf in segment_features)
    
    return features

# Process wells in batches
batch_size = 100
well_ids = events_df['well_id'].unique()
all_features = []

print(f"\n🔄 Processing {len(well_ids):,} wells...")

for batch_start in tqdm(range(0, len(well_ids), batch_size), desc="Processing batches"):
    batch_wells = well_ids[batch_start:batch_start + batch_size]
    
    # Get time series data for batch
    wells_str = "','".join(batch_wells)
    
    # Parse well_ids to get plate_id and well_number
    well_parts = [w.split('_') for w in batch_wells]
    
    # Build query conditions
    conditions = []
    for parts in well_parts:
        if len(parts) >= 2:
            plate_id = '_'.join(parts[:-1])
            well_number = parts[-1]
            conditions.append(f"(plate_id = '{plate_id}' AND well_number = {well_number})")
    
    if not conditions:
        continue
    
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
        
        # Get events for this well
        well_events = events_df[events_df['well_id'] == well_id]
        
        if len(well_events) == 0:
            continue
        
        # Calculate time in hours
        well_data['hours'] = (well_data['timestamp'] - well_data['timestamp'].min()).dt.total_seconds() / 3600
        
        # Extract features
        features = extract_inter_event_features(
            well_data['hours'].values,
            well_data['oxygen'].values,
            well_events['event_time_hours'].values
        )
        
        if features:
            features['well_id'] = well_id
            features['drug'] = well_events.iloc[0]['drug']
            features['concentration'] = well_events.iloc[0]['concentration']
            features['n_events'] = len(well_events)
            all_features.append(features)

# Create features dataframe
features_df = pd.DataFrame(all_features)

print(f"\n📊 FEATURE EXTRACTION RESULTS:")
print(f"   Wells with features: {len(features_df):,}")
print(f"   Drugs represented: {features_df['drug'].nunique()}")
print(f"   Feature columns: {len(features_df.columns) - 3}")  # Exclude id columns

# Aggregate to drug level
print(f"\n🧬 Aggregating to drug level...")

# Group by drug and aggregate
numeric_cols = [col for col in features_df.columns if col not in ['well_id', 'drug', 'concentration']]
drug_features = features_df.groupby('drug')[numeric_cols].agg(['mean', 'std', 'count'])

# Flatten column names
drug_features.columns = ['_'.join(col).strip() for col in drug_features.columns.values]
drug_features = drug_features.reset_index()

print(f"   Drug-level features: {len(drug_features)} drugs")

# Save results
features_df.to_parquet(results_dir / "event_aware_features_wells.parquet", index=False)
drug_features.to_parquet(results_dir / "event_aware_features_drugs.parquet", index=False)

print(f"\n💾 Saved features to:")
print(f"   Well-level: {results_dir / 'event_aware_features_wells.parquet'}")
print(f"   Drug-level: {results_dir / 'event_aware_features_drugs.parquet'}")

# Create summary visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Event-Aware Feature Summary', fontsize=14, fontweight='bold')

# Plot 1: Consumption rate distribution
ax = axes[0, 0]
if 'consumption_rate_mean_mean' in drug_features.columns:
    consumption_rates = drug_features['consumption_rate_mean_mean'].dropna()
    ax.hist(consumption_rates, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(consumption_rates.median(), color='red', linestyle='--', 
               label=f'Median: {consumption_rates.median():.3f}')
    ax.set_xlabel('Mean Oxygen Consumption Rate (%O₂/hour)')
    ax.set_ylabel('Number of Drugs')
    ax.set_title('Oxygen Consumption Rate Distribution')
    ax.legend()

# Plot 2: Temporal progression
ax = axes[0, 1]
if 'consumption_change_mean' in drug_features.columns:
    consumption_changes = drug_features['consumption_change_mean'].dropna()
    colors = ['green' if x < 0 else 'red' for x in consumption_changes]
    ax.scatter(range(len(consumption_changes)), consumption_changes, c=colors, alpha=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Drug Index')
    ax.set_ylabel('Consumption Rate Change (Late - Early)')
    ax.set_title('Temporal Progression of Oxygen Consumption')
    ax.text(0.02, 0.98, f'Decreasing: {(consumption_changes < 0).sum()}\nIncreasing: {(consumption_changes > 0).sum()}',
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: Feature coverage
ax = axes[1, 0]
feature_coverage = features_df.groupby('drug').size()
ax.hist(feature_coverage, bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Wells per Drug')
ax.set_ylabel('Number of Drugs')
ax.set_title('Well Coverage by Drug')
ax.axvline(feature_coverage.median(), color='red', linestyle='--',
           label=f'Median: {feature_coverage.median():.0f} wells')
ax.legend()

# Plot 4: CV distribution
ax = axes[1, 1]
if 'cv_o2_mean_mean' in drug_features.columns:
    cv_values = drug_features['cv_o2_mean_mean'].dropna()
    ax.hist(cv_values, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(cv_values.median(), color='red', linestyle='--',
               label=f'Median CV: {cv_values.median():.3f}')
    ax.set_xlabel('Mean Coefficient of Variation')
    ax.set_ylabel('Number of Drugs')
    ax.set_title('Oxygen Measurement Variability')
    ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / "event_aware_features_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📊 Summary figure saved to: {fig_dir / 'event_aware_features_summary.png'}")

# Feature statistics
print(f"\n📈 FEATURE STATISTICS:")
if 'consumption_rate_mean_mean' in drug_features.columns:
    print(f"   Consumption rate: {drug_features['consumption_rate_mean_mean'].median():.3f} ± {drug_features['consumption_rate_mean_mean'].std():.3f} %O₂/hour")
if 'cv_o2_mean_mean' in drug_features.columns:
    print(f"   Average CV: {drug_features['cv_o2_mean_mean'].mean():.3f}")
if 'n_segments_mean' in drug_features.columns:
    print(f"   Segments per well: {drug_features['n_segments_mean'].mean():.1f}")
if 'total_monitored_hours_mean' in drug_features.columns:
    print(f"   Monitored hours: {drug_features['total_monitored_hours_mean'].mean():.1f}")

print("\n✅ Event-aware feature extraction complete!")

conn.close()

def main():
    """Main entry point for event-aware feature extraction"""
    # This script is designed to be run directly
    pass

if __name__ == "__main__":
    main()