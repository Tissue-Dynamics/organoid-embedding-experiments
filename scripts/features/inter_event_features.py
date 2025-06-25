#!/usr/bin/env python3
"""
Inter-Event Period Feature Extraction

PURPOSE:
    Extracts features from periods between media changes to capture how drug
    responses evolve over sequential events. This implements Step 5.1 of the
    Advanced Feature Engineering Plan.

METHODOLOGY:
    - Segments each well's time series by detected media change events
    - Extracts catch22 and SAX features for each inter-event period
    - Calculates stability metrics (CV, trend) relative to baseline
    - Measures response magnitude compared to pre-treatment baseline
    - Tracks feature evolution across sequential media changes
    - Builds SAX pattern evolution maps

INPUTS:
    - Database connection for oxygen consumption data
    - Existing media change detection results
    - Baseline feature extractions

OUTPUTS:
    - results/data/inter_event_features_wells.parquet
      Features for each inter-event period per well
    - results/data/inter_event_features_drugs.parquet  
      Aggregated features by drug/concentration
    - results/data/sax_pattern_evolution.parquet
      SAX pattern changes across events
    - results/figures/inter_event_features/
      Visualizations of temporal feature evolution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
from scipy import stats
import pycatch22
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from collections import Counter

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "inter_event_features"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INTER-EVENT PERIOD FEATURE EXTRACTION")
print("=" * 80)

# Configuration
BASELINE_HOURS = 48  # Baseline period for comparison
MIN_PERIOD_POINTS = 15  # Minimum points needed for valid inter-event period
SAX_CONFIGS = [
    {'n_symbols': 4, 'alphabet_size': 3, 'name': 'coarse'},
    {'n_symbols': 8, 'alphabet_size': 4, 'name': 'medium'},
    {'n_symbols': 16, 'alphabet_size': 6, 'name': 'fine'}
]

# Initialize data loader
loader = DataLoader()

print("\n📊 Loading data...")
# Load oxygen consumption data (limit for development)
df = loader.load_oxygen_data(limit=5)  # Start with 5 plates for faster processing
print(f"   Loaded {len(df):,} measurements from {df['well_id'].nunique():,} wells")

# Rename o2 column for consistency
df = df.rename(columns={'o2': 'oxygen'})

# Load existing media change events if available
try:
    media_events_df = pd.read_parquet(results_dir / "improved_media_change_events_expanded.parquet")
    if len(media_events_df) > 0 and 'well_id' in media_events_df.columns:
        print(f"   Loaded media change events for {media_events_df['well_id'].nunique()} wells")
        events_available = True
    else:
        print("   Media events file empty - will detect events")
        events_available = False
except FileNotFoundError:
    print("   No existing media change events found - will detect events")
    events_available = False

# ========== EVENT DETECTION FUNCTIONS ==========

def detect_media_changes_simple(well_data, window_hours=6, variance_threshold=2.0):
    """Simple media change detection based on variance spikes"""
    well_data = well_data.sort_values('elapsed_hours').copy()
    
    # Calculate rolling variance
    window_size = max(3, int(window_hours * len(well_data) / well_data['elapsed_hours'].max()))
    well_data['rolling_var'] = well_data['oxygen'].rolling(window=window_size, center=True).var()
    
    # Find baseline variance (first 48 hours)
    baseline_data = well_data[well_data['elapsed_hours'] <= BASELINE_HOURS]
    if len(baseline_data) < 10:
        return []
    
    baseline_var = baseline_data['rolling_var'].mean()
    if np.isnan(baseline_var) or baseline_var == 0:
        return []
    
    # Find variance spikes
    well_data['var_ratio'] = well_data['rolling_var'] / baseline_var
    spike_mask = well_data['var_ratio'] > variance_threshold
    
    # Group consecutive spikes and find event times
    events = []
    in_spike = False
    spike_start = None
    
    for idx, row in well_data.iterrows():
        if spike_mask[idx] and not in_spike:
            spike_start = row['elapsed_hours']
            in_spike = True
        elif not spike_mask[idx] and in_spike:
            # End of spike - record event
            events.append({
                'event_time': spike_start,
                'event_number': len(events) + 1,
                'peak_var_ratio': well_data.loc[spike_start:row['elapsed_hours'], 'var_ratio'].max()
            })
            in_spike = False
    
    return events

def extract_inter_event_periods(well_data, events):
    """Extract data segments between media change events"""
    well_data = well_data.sort_values('elapsed_hours').copy()
    periods = []
    
    if len(events) == 0:
        return periods
    
    # Add baseline as first "period"
    baseline_end = min(BASELINE_HOURS, events[0]['event_time'] - 2)  # 2h buffer before first event
    baseline_data = well_data[well_data['elapsed_hours'] <= baseline_end]
    
    if len(baseline_data) >= MIN_PERIOD_POINTS:
        periods.append({
            'period_number': 0,
            'period_type': 'baseline',
            'start_time': 0,
            'end_time': baseline_end,
            'duration_hours': baseline_end,
            'data': baseline_data,
            'n_points': len(baseline_data)
        })
    
    # Extract inter-event periods
    for i in range(len(events)):
        # Start time: 2h after current event (recovery period)
        start_time = events[i]['event_time'] + 2
        
        # End time: 2h before next event, or end of data
        if i + 1 < len(events):
            end_time = events[i + 1]['event_time'] - 2
        else:
            end_time = well_data['elapsed_hours'].max()
        
        if end_time <= start_time:
            continue
            
        period_data = well_data[
            (well_data['elapsed_hours'] >= start_time) & 
            (well_data['elapsed_hours'] <= end_time)
        ].copy()
        
        if len(period_data) >= MIN_PERIOD_POINTS:
            periods.append({
                'period_number': i + 1,
                'period_type': 'inter_event',
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': end_time - start_time,
                'data': period_data,
                'n_points': len(period_data),
                'post_event_number': events[i]['event_number']
            })
    
    return periods

# ========== FEATURE EXTRACTION FUNCTIONS ==========

def extract_period_catch22_features(period_data):
    """Extract catch22 features for a period"""
    oxygen_values = period_data['oxygen'].values
    
    if len(oxygen_values) < 10:
        return {}
    
    try:
        features = pycatch22.catch22_all(oxygen_values)
        feature_dict = {}
        
        for i, name in enumerate(features['names']):
            feature_dict[f'catch22_{name}'] = features['values'][i]
            
        return feature_dict
    except:
        return {}

def extract_period_sax_features(period_data, config):
    """Extract SAX features for a period"""
    oxygen_values = period_data['oxygen'].values
    
    if len(oxygen_values) < config['n_symbols']:
        return {}
    
    try:
        # Simple SAX implementation to avoid numpy compatibility issues
        # Normalize data
        if np.std(oxygen_values) == 0:
            return {}
            
        normalized = (oxygen_values - np.mean(oxygen_values)) / np.std(oxygen_values)
        
        # Simple binning approach
        n_bins = config['alphabet_size']
        bin_edges = np.linspace(normalized.min(), normalized.max(), n_bins + 1)
        bin_indices = np.digitize(normalized, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Convert to alphabet
        alphabet = [chr(ord('a') + i) for i in range(n_bins)]
        sax_string = ''.join([alphabet[idx] for idx in bin_indices])
        
        # Calculate features
        features = {}
        level = config['name']
        
        # Symbol frequencies
        symbol_counts = Counter(sax_string)
        total_symbols = len(sax_string)
        
        for symbol in alphabet:
            freq = symbol_counts.get(symbol, 0) / total_symbols
            features[f'sax_{level}_symbol_{symbol}_freq'] = freq
        
        # Entropy
        probs = np.array([symbol_counts.get(s, 0) for s in alphabet]) / total_symbols
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
        features[f'sax_{level}_entropy'] = entropy
        
        # Transitions
        transitions = []
        for i in range(len(sax_string) - 1):
            transitions.append(f'{sax_string[i]}{sax_string[i+1]}')
        
        if transitions:
            transition_counts = Counter(transitions)
            most_common = transition_counts.most_common(1)[0] if transition_counts else ('aa', 0)
            features[f'sax_{level}_dominant_transition'] = most_common[1] / len(transitions)
            features[f'sax_{level}_n_unique_transitions'] = len(transition_counts)
        
        # Pattern complexity
        features[f'sax_{level}_pattern_length'] = len(sax_string)
        features[f'sax_{level}_unique_symbols'] = len(symbol_counts)
        
        return features
        
    except Exception as e:
        print(f"SAX extraction error: {e}")
        return {}

def extract_period_stability_features(period_data, baseline_stats=None):
    """Extract stability and trend features for a period"""
    oxygen_values = period_data['oxygen'].values
    
    if len(oxygen_values) < 5:
        return {}
    
    features = {}
    
    # Basic statistics
    features['mean_oxygen'] = np.mean(oxygen_values)
    features['std_oxygen'] = np.std(oxygen_values)
    features['cv_oxygen'] = np.std(oxygen_values) / np.mean(oxygen_values) if np.mean(oxygen_values) != 0 else np.nan
    features['min_oxygen'] = np.min(oxygen_values)
    features['max_oxygen'] = np.max(oxygen_values)
    features['range_oxygen'] = np.max(oxygen_values) - np.min(oxygen_values)
    
    # Trend analysis
    if len(oxygen_values) > 2:
        time_values = np.arange(len(oxygen_values))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_values, oxygen_values)
            features['trend_slope'] = slope
            features['trend_r_squared'] = r_value ** 2
            features['trend_p_value'] = p_value
            features['trend_significant'] = p_value < 0.05
        except:
            features['trend_slope'] = 0
            features['trend_r_squared'] = 0
            features['trend_p_value'] = 1
            features['trend_significant'] = False
    
    # Baseline comparison (if available)
    if baseline_stats is not None:
        baseline_mean = baseline_stats.get('mean_oxygen', np.nan)
        baseline_std = baseline_stats.get('std_oxygen', np.nan)
        
        if not np.isnan(baseline_mean):
            features['baseline_deviation'] = (features['mean_oxygen'] - baseline_mean) / baseline_mean
            features['baseline_fold_change'] = features['mean_oxygen'] / baseline_mean
        
        if not np.isnan(baseline_std) and baseline_std > 0:
            features['baseline_cv_ratio'] = features['cv_oxygen'] / (baseline_std / baseline_mean)
    
    # Variability patterns
    if len(oxygen_values) > 10:
        # Rolling CV to detect stability changes
        window_size = min(len(oxygen_values) // 3, 10)
        rolling_means = np.convolve(oxygen_values, np.ones(window_size)/window_size, mode='valid')
        rolling_stds = np.array([np.std(oxygen_values[i:i+window_size]) for i in range(len(oxygen_values) - window_size + 1)])
        rolling_cvs = rolling_stds / rolling_means
        rolling_cvs = rolling_cvs[~np.isnan(rolling_cvs)]
        
        if len(rolling_cvs) > 0:
            features['stability_mean_cv'] = np.mean(rolling_cvs)
            features['stability_cv_trend'] = np.corrcoef(np.arange(len(rolling_cvs)), rolling_cvs)[0, 1] if len(rolling_cvs) > 1 else 0
    
    return features

# ========== MAIN PROCESSING ==========

print("\n🔄 Loading well metadata...")
well_metadata = loader.load_well_metadata()
print(f"   Loaded metadata for {len(well_metadata)} wells")

# The oxygen data already has drug info - no need to merge
df_with_drugs = df.copy()

print(f"   Data columns: {df_with_drugs.columns.tolist()}")
print(f"   Wells with drug info: {df_with_drugs['well_id'].nunique()}")
print(f"   Unique drugs: {df_with_drugs['drug'].nunique()}")

print("\n🔄 Processing wells for inter-event features...")

all_period_features = []
well_ids = df_with_drugs['well_id'].unique()

for well_id in tqdm(well_ids, desc="Processing wells"):
    well_data = df_with_drugs[df_with_drugs['well_id'] == well_id].copy()
    
    if len(well_data) < 50:  # Skip wells with too little data
        continue
    
    # Get drug and concentration info from merged data
    drug = well_data['drug'].iloc[0]
    concentration = well_data['concentration'].iloc[0]
    
    # Detect media changes (or use existing if available)
    if events_available:
        existing_events = media_events_df[media_events_df['well_id'] == well_id]
        events = []
        for _, event in existing_events.iterrows():
            events.append({
                'event_time': event['event_time_hours'],
                'event_number': event['event_number']
            })
    else:
        events = detect_media_changes_simple(well_data)
    
    if len(events) == 0:
        continue
    
    # Extract inter-event periods
    periods = extract_inter_event_periods(well_data, events)
    
    if len(periods) == 0:
        continue
    
    # Get baseline statistics for comparison
    baseline_period = next((p for p in periods if p['period_type'] == 'baseline'), None)
    baseline_stats = None
    
    if baseline_period is not None:
        baseline_stability = extract_period_stability_features(baseline_period['data'])
        baseline_stats = baseline_stability
    
    # Extract features for each period
    for period in periods:
        period_features = {
            'well_id': well_id,
            'drug': drug,
            'concentration': concentration,
            'period_number': period['period_number'],
            'period_type': period['period_type'],
            'start_time': period['start_time'],
            'end_time': period['end_time'],
            'duration_hours': period['duration_hours'],
            'n_points': period['n_points']
        }
        
        # Add post-event info for inter-event periods
        if 'post_event_number' in period:
            period_features['post_event_number'] = period['post_event_number']
        
        # Extract catch22 features
        catch22_features = extract_period_catch22_features(period['data'])
        period_features.update(catch22_features)
        
        # Extract SAX features at multiple levels
        for config in SAX_CONFIGS:
            sax_features = extract_period_sax_features(period['data'], config)
            period_features.update(sax_features)
        
        # Extract stability features
        stability_features = extract_period_stability_features(period['data'], baseline_stats)
        period_features.update(stability_features)
        
        all_period_features.append(period_features)

# Convert to DataFrame
inter_event_features_df = pd.DataFrame(all_period_features)

if len(inter_event_features_df) > 0:
    print(f"\n📊 INTER-EVENT FEATURE EXTRACTION RESULTS:")
    print(f"   Total periods processed: {len(inter_event_features_df)}")
    print(f"   Wells with features: {inter_event_features_df['well_id'].nunique()}")
    print(f"   Drugs: {inter_event_features_df['drug'].nunique()}")
    print(f"   Period types: {inter_event_features_df['period_type'].value_counts().to_dict()}")
    
    # Analyze feature coverage
    feature_cols = [col for col in inter_event_features_df.columns 
                   if col not in ['well_id', 'drug', 'concentration', 'period_number', 
                                 'period_type', 'start_time', 'end_time', 'duration_hours', 
                                 'n_points', 'post_event_number']]
    print(f"   Features extracted: {len(feature_cols)}")
    
    # Count non-null values for key feature types
    catch22_cols = [col for col in feature_cols if 'catch22' in col]
    sax_cols = [col for col in feature_cols if 'sax' in col]
    stability_cols = [col for col in feature_cols if col in ['mean_oxygen', 'cv_oxygen', 'trend_slope']]
    
    print(f"   catch22 features: {len(catch22_cols)}")
    print(f"   SAX features: {len(sax_cols)}")
    print(f"   Stability features: {len(stability_cols)}")
    
    # Save results
    print(f"\n💾 Saving inter-event features...")
    inter_event_features_df.to_parquet(results_dir / 'inter_event_features_wells.parquet', index=False)
    print(f"   Well-level features: {results_dir / 'inter_event_features_wells.parquet'}")
    
    # Create drug-level aggregated features
    print(f"\n🔄 Creating drug-level aggregated features...")
    
    # Aggregate by drug, concentration, and period type
    drug_features = []
    
    for (drug, conc, period_type), group in inter_event_features_df.groupby(['drug', 'concentration', 'period_type']):
        if len(group) == 0:
            continue
            
        drug_feature_row = {
            'drug': drug,
            'concentration': conc,
            'period_type': period_type,
            'n_wells': len(group),
            'mean_duration_hours': group['duration_hours'].mean(),
            'mean_n_points': group['n_points'].mean()
        }
        
        # Aggregate numerical features
        for col in feature_cols:
            if col in group.columns:
                numeric_vals = pd.to_numeric(group[col], errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    drug_feature_row[f'{col}_mean'] = numeric_vals.mean()
                    drug_feature_row[f'{col}_std'] = numeric_vals.std()
                    drug_feature_row[f'{col}_cv'] = numeric_vals.std() / numeric_vals.mean() if numeric_vals.mean() != 0 else np.nan
        
        drug_features.append(drug_feature_row)
    
    drug_features_df = pd.DataFrame(drug_features)
    drug_features_df.to_parquet(results_dir / 'inter_event_features_drugs.parquet', index=False)
    print(f"   Drug-level features: {results_dir / 'inter_event_features_drugs.parquet'}")
    
    # Create SAX pattern evolution analysis
    print(f"\n🔄 Analyzing SAX pattern evolution...")
    
    sax_evolution = []
    
    for (well_id, drug, conc), well_group in inter_event_features_df.groupby(['well_id', 'drug', 'concentration']):
        well_periods = well_group.sort_values('period_number')
        
        if len(well_periods) < 2:
            continue
        
        for config in SAX_CONFIGS:
            level = config['name']
            alphabet = [chr(ord('a') + i) for i in range(config['alphabet_size'])]
            
            # Track symbol frequency changes across periods
            for symbol in alphabet:
                freq_col = f'sax_{level}_symbol_{symbol}_freq'
                if freq_col in well_periods.columns:
                    symbol_freqs = well_periods[freq_col].values
                    
                    # Calculate trend across periods
                    if len(symbol_freqs) > 1 and not np.all(np.isnan(symbol_freqs)):
                        valid_freqs = symbol_freqs[~np.isnan(symbol_freqs)]
                        if len(valid_freqs) > 1:
                            period_nums = np.arange(len(valid_freqs))
                            slope, _, r_value, p_value, _ = stats.linregress(period_nums, valid_freqs)
                            
                            sax_evolution.append({
                                'well_id': well_id,
                                'drug': drug,
                                'concentration': conc,
                                'sax_level': level,
                                'symbol': symbol,
                                'freq_trend_slope': slope,
                                'freq_trend_r2': r_value ** 2,
                                'freq_trend_p_value': p_value,
                                'n_periods': len(valid_freqs),
                                'initial_freq': valid_freqs[0],
                                'final_freq': valid_freqs[-1],
                                'freq_change': valid_freqs[-1] - valid_freqs[0]
                            })
    
    if sax_evolution:
        sax_evolution_df = pd.DataFrame(sax_evolution)
        sax_evolution_df.to_parquet(results_dir / 'sax_pattern_evolution.parquet', index=False)
        print(f"   SAX evolution: {results_dir / 'sax_pattern_evolution.parquet'}")
        print(f"   Pattern evolution records: {len(sax_evolution_df)}")
    
    print(f"\n✅ Inter-event feature extraction complete!")
    print(f"\n🎯 Next steps:")
    print(f"   1. Analyze progressive effect patterns (Step 5.2)")
    print(f"   2. Build event-normalized time features (Step 5.3)")
    print(f"   3. Apply Hill curve fitting to inter-event features")
    print(f"   4. Create hierarchical feature architecture (Step 6)")

else:
    print("\n❌ No inter-event features extracted!")
    print("   Check media change detection and data quality")