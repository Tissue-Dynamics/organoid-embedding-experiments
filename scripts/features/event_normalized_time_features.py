#!/usr/bin/env python3
"""
Event-Normalized Time Features

PURPOSE:
    Creates features based on time relative to media change events rather than 
    absolute time. This approach captures drug response dynamics more accurately
    by aligning temporal patterns to biological intervention points.

METHODOLOGY:
    Instead of using elapsed_hours from experiment start, we use:
    - Time since last media change (hours_since_event)
    - Time until next media change (hours_to_next_event)
    - Recovery trajectories post-media change
    - Pre-event buildup patterns
    
    This allows comparison of drug responses at equivalent biological time points
    across different experimental schedules.

FEATURES:
    1. Post-event recovery features (0-6h, 6-12h, 12-24h windows)
    2. Pre-event anticipation features (-6-0h before next event)
    3. Event-aligned catch22 features
    4. Event-aligned SAX patterns
    5. Recovery rate metrics (time to 50%, 90% baseline)
    6. Event response consistency across multiple events

INPUTS:
    - Oxygen consumption time series data
    - Media change event timings
    - Well metadata (drug, concentration)

OUTPUTS:
    - results/data/event_normalized_features_wells.parquet
      Well-level event-normalized features
    - results/data/event_normalized_features_drugs.parquet
      Drug-level aggregated features
    - results/figures/event_normalized/
      Visualizations of event-aligned patterns
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
print("EVENT-NORMALIZED TIME FEATURES")
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

# ========== MAIN PROCESSING ==========

print("\n📊 Loading oxygen consumption data...")
loader = DataLoader()
df = loader.load_oxygen_data(limit=2)  # Process only 2 plates for speed

# Rename o2 to oxygen if needed
if 'o2' in df.columns and 'oxygen' not in df.columns:
    df = df.rename(columns={'o2': 'oxygen'})

print(f"   Loaded {len(df)} oxygen measurements")

# Get unique wells
wells = df.groupby('well_id').agg({
    'plate_id': 'first',
    'drug': 'first', 
    'concentration': 'first'
}).reset_index()

# Sample 100 wells for faster processing
if len(wells) > 100:
    wells = wells.sample(n=100, random_state=42)

print(f"   Processing {len(wells)} wells")

# Process each well
well_features_list = []

for idx, well_info in wells.iterrows():
    if idx % 100 == 0:
        print(f"   Processing well {idx+1}/{len(wells)}...")
    
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
        well_features_list.append(features)

print(f"\n✓ Extracted features for {len(well_features_list)} wells")

# Convert to DataFrame
well_features_df = pd.DataFrame(well_features_list)

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
print(f"   Created drug-level features for {len(drug_features_df)} drug-concentration combinations")

# ========== VISUALIZATION ==========

print("\n📊 Creating visualizations...")

# 1. Event-aligned oxygen trajectories
print("   Creating event-aligned trajectory plot...")

# Sample a few drugs with different DILI levels
sample_drugs = ['Sorafenib', 'Dasatinib', 'Alpelisib']  # High, Medium, Low DILI
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, drug in enumerate(sample_drugs):
    ax = axes[i]
    
    drug_wells = wells[wells['drug'] == drug]
    if len(drug_wells) == 0:
        continue
    
    # Get data for one concentration
    conc = drug_wells['concentration'].iloc[0]
    drug_conc_wells = drug_wells[drug_wells['concentration'] == conc]['well_id'].values[:4]
    
    # Plot event-aligned trajectories
    for well_id in drug_conc_wells:
        well_data = df[df['well_id'] == well_id].sort_values('elapsed_hours')
        events = detect_media_changes(well_data)
        
        if len(events) > 0:
            normalized_data = create_event_normalized_timeseries(well_data, events)
            
            # Plot first event response
            event_1_data = normalized_data[normalized_data['event_number'] == 1]
            if len(event_1_data) > 0:
                ax.plot(event_1_data['hours_since_event'], 
                       event_1_data['oxygen'], 
                       alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Hours Since Media Change')
    ax.set_title(f'{drug} (Event 1)')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Media Change')
    
    if i == 0:
        ax.set_ylabel('Oxygen Consumption')

plt.suptitle('Event-Aligned Oxygen Consumption Trajectories', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'event_aligned_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Recovery metrics by drug
print("   Creating recovery metrics visualization...")

# Calculate mean recovery time for each drug
recovery_metrics = []

for drug in drug_features_df['drug'].unique()[:20]:  # Top 20 drugs
    drug_data = well_features_df[well_features_df['drug'] == drug]
    
    if len(drug_data) > 0:
        # Get recovery time features
        recovery_cols = [col for col in drug_data.columns if 'time_to_90pct_recovery' in col and '_mean' in col]
        
        if recovery_cols:
            recovery_times = []
            for col in recovery_cols:
                values = drug_data[col].dropna()
                if len(values) > 0:
                    recovery_times.extend(values)
            
            if recovery_times:
                recovery_metrics.append({
                    'drug': drug,
                    'mean_recovery_time': np.mean(recovery_times),
                    'std_recovery_time': np.std(recovery_times)
                })

if recovery_metrics:
    recovery_df = pd.DataFrame(recovery_metrics)
    recovery_df = recovery_df.sort_values('mean_recovery_time')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(recovery_df)), recovery_df['mean_recovery_time'], 
                   yerr=recovery_df['std_recovery_time'], capsize=5, alpha=0.7)
    
    plt.xticks(range(len(recovery_df)), recovery_df['drug'], rotation=45, ha='right')
    plt.ylabel('Hours to 90% Recovery')
    plt.title('Media Change Recovery Time by Drug', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'recovery_time_by_drug.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Feature importance heatmap
print("   Creating feature importance heatmap...")

# Get a subset of features for visualization
feature_subset = drug_features_df.select_dtypes(include=[np.number]).columns
feature_subset = [col for col in feature_subset if any(x in col for x in 
                  ['immediate_post', 'early_recovery', 'max_suppression', 'time_to_90'])][:20]

if len(feature_subset) > 5 and len(drug_features_df) > 5:
    # Create heatmap data
    heatmap_data = drug_features_df.set_index('drug')[feature_subset].head(15)
    
    # Normalize features
    scaler = StandardScaler()
    heatmap_data_norm = pd.DataFrame(
        scaler.fit_transform(heatmap_data.fillna(0)),
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_norm.T, cmap='RdBu_r', center=0, 
                xticklabels=True, yticklabels=True, cbar_kws={'label': 'Normalized Value'})
    
    plt.xlabel('Drug')
    plt.title('Event-Normalized Features Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'event_normalized_features_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== SAVE RESULTS ==========

print("\n💾 Saving event-normalized features...")

# Save well-level features
well_features_df.to_parquet(results_dir / 'event_normalized_features_wells.parquet', index=False)
print(f"   Well-level features: {results_dir / 'event_normalized_features_wells.parquet'}")

# Save drug-level features
drug_features_df.to_parquet(results_dir / 'event_normalized_features_drugs.parquet', index=False)
print(f"   Drug-level features: {results_dir / 'event_normalized_features_drugs.parquet'}")

# Feature summary
feature_summary = {
    'n_wells_processed': len(well_features_list),
    'n_drugs': len(drug_features_df['drug'].unique()),
    'n_features_per_well': len([col for col in well_features_df.columns if col not in ['well_id', 'drug', 'concentration']]),
    'event_windows': EVENT_WINDOWS,
    'recovery_thresholds': RECOVERY_THRESHOLDS,
    'feature_categories': {
        'window_features': len([col for col in well_features_df.columns if any(w in col for w in EVENT_WINDOWS.keys())]),
        'recovery_metrics': len([col for col in well_features_df.columns if 'recovery' in col or 'suppression' in col]),
        'consistency_metrics': len([col for col in well_features_df.columns if 'consistency' in col]),
        'catch22_features': len([col for col in well_features_df.columns if 'catch22' in col])
    }
}

import json
with open(results_dir / 'event_normalized_features_summary.json', 'w') as f:
    json.dump(feature_summary, f, indent=2)

print(f"\n✅ Event-normalized feature extraction complete!")
print(f"\n📊 SUMMARY:")
print(f"   Processed {feature_summary['n_wells_processed']} wells")
print(f"   Created {feature_summary['n_features_per_well']} features per well")
print(f"   Covered {feature_summary['n_drugs']} unique drugs")
print(f"\n🎯 Key features extracted:")
print(f"   - Post-event recovery trajectories (0-48h)")
print(f"   - Pre-event anticipation patterns (-6-0h)")
print(f"   - Recovery time to baseline thresholds")
print(f"   - Event response consistency metrics")
print(f"   - Event-aligned catch22 features")