#!/usr/bin/env python3
"""
Hierarchical Feature Architecture

PURPOSE:
    Creates the unified, comprehensive feature vector that combines ALL extracted 
    features into a structured, pharmacologically-grounded embedding for drug
    characterization. This implements Step 6 of the Advanced Feature Engineering Plan.

METHODOLOGY:
    Combines features from all extraction pipelines:
    - Baseline features (catch22, SAX, stability metrics)
    - Multi-timescale catch22 features (24h, 48h, 96h)
    - Hierarchical SAX features (coarse, medium, fine)
    - Hill curve parameters (EC50, Emax, Hill slopes) for all feature types
    - Inter-event period features
    - Progressive effect indicators
    - Quality assessment flags

ARCHITECTURE:
    [baseline_features | multiscale_catch22 | hierarchical_SAX | 
     hill_parameters | inter_event_features | progressive_indicators | 
     quality_flags]

INPUTS:
    - All feature datasets from previous extraction steps
    - Hill curve parameter files
    - Quality assessment results

OUTPUTS:
    - results/data/hierarchical_drug_embeddings.parquet
      Complete drug embedding matrix
    - results/data/hierarchical_feature_metadata.json
      Feature names, types, and interpretability info
    - results/figures/hierarchical_features/
      Comprehensive visualizations of the unified feature space
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import sys

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "hierarchical_features"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HIERARCHICAL FEATURE ARCHITECTURE")
print("=" * 80)

# ========== FEATURE LOADING AND VALIDATION ==========

def load_feature_datasets():
    """Load all available feature datasets"""
    datasets = {}
    
    # 1. Multi-scale catch22 features
    try:
        datasets['multiscale_catch22'] = pd.read_parquet(results_dir / "multiscale_catch22_drugs.parquet")
        print(f"   ✓ Multi-scale catch22: {datasets['multiscale_catch22'].shape}")
    except FileNotFoundError:
        print("   ✗ Multi-scale catch22 features not found")
        datasets['multiscale_catch22'] = pd.DataFrame()
    
    # 2. Hierarchical SAX features
    try:
        datasets['hierarchical_sax'] = pd.read_parquet(results_dir / "hierarchical_sax_features_drugs.parquet")
        print(f"   ✓ Hierarchical SAX: {datasets['hierarchical_sax'].shape}")
    except FileNotFoundError:
        print("   ✗ Hierarchical SAX features not found")
        datasets['hierarchical_sax'] = pd.DataFrame()
    
    # 3. Event-aware features
    try:
        datasets['event_aware'] = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")
        print(f"   ✓ Event-aware: {datasets['event_aware'].shape}")
    except FileNotFoundError:
        print("   ✗ Event-aware features not found")
        datasets['event_aware'] = pd.DataFrame()
    
    # 4. Inter-event features
    try:
        datasets['inter_event'] = pd.read_parquet(results_dir / "inter_event_features_drugs.parquet")
        print(f"   ✓ Inter-event: {datasets['inter_event'].shape}")
    except FileNotFoundError:
        print("   ✗ Inter-event features not found")
        datasets['inter_event'] = pd.DataFrame()
    
    # 5. Hill curve parameters
    try:
        datasets['hill_parameters'] = pd.read_parquet(results_dir / "dose_response_hill_parameters.parquet")
        print(f"   ✓ Hill parameters: {datasets['hill_parameters'].shape}")
    except FileNotFoundError:
        print("   ✗ Hill parameters not found")
        datasets['hill_parameters'] = pd.DataFrame()
    
    # 6. Comprehensive Hill parameters (if available)
    try:
        datasets['comprehensive_hill'] = pd.read_parquet(results_dir / "comprehensive_hill_parameters.parquet")
        print(f"   ✓ Comprehensive Hill: {datasets['comprehensive_hill'].shape}")
    except FileNotFoundError:
        print("   ✗ Comprehensive Hill parameters not found")
        datasets['comprehensive_hill'] = pd.DataFrame()
    
    # 7. Quality features
    try:
        datasets['quality'] = pd.read_parquet(results_dir / "quality_aware_features.parquet")
        print(f"   ✓ Quality features: {datasets['quality'].shape}")
    except FileNotFoundError:
        print("   ✗ Quality features not found")
        datasets['quality'] = pd.DataFrame()
    
    return datasets

def extract_drug_level_features(df, prefix, drug_col='drug', conc_col='concentration'):
    """Extract drug-level features from a dataset"""
    if len(df) == 0:
        return pd.DataFrame()
    
    # Check if concentration column exists
    if conc_col not in df.columns:
        print(f"   WARNING: {conc_col} not found in {prefix} data, using drug-level aggregation")
        # Aggregate by drug only
        feature_cols = [col for col in df.columns if col not in [drug_col]]
        drug_features = df.groupby(drug_col)[feature_cols].mean().reset_index()
        drug_features.columns = [drug_col] + [f'{prefix}_{col}' for col in feature_cols]
        return drug_features
    
    # Get feature columns (exclude metadata)
    meta_cols = [drug_col, conc_col, 'n_wells', 'n_replicates', 'plate_id', 'well_id', 'period_type', 'period_number']
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    if len(feature_cols) == 0:
        print(f"   WARNING: No feature columns found in {prefix} data")
        return pd.DataFrame()
    
    # Aggregate features by drug (across all concentrations)
    drug_features = df.groupby(drug_col)[feature_cols].mean().reset_index()
    
    # Rename columns with prefix
    feature_col_mapping = {col: f'{prefix}_{col}' for col in feature_cols}
    drug_features = drug_features.rename(columns=feature_col_mapping)
    
    print(f"   {prefix}: {len(feature_cols)} features for {len(drug_features)} drugs")
    return drug_features

def extract_hill_parameters_by_drug(hill_df):
    """Extract Hill parameters organized by drug and feature type"""
    if len(hill_df) == 0:
        return pd.DataFrame()
    
    # Pivot Hill parameters by drug
    hill_features = []
    
    for drug in hill_df['drug'].unique():
        drug_data = hill_df[hill_df['drug'] == drug]
        
        drug_feature_row = {'drug': drug}
        
        # Group parameters by feature type if available
        if 'feature_type' in drug_data.columns:
            for feature_type in drug_data['feature_type'].unique():
                type_data = drug_data[drug_data['feature_type'] == feature_type]
                
                # Calculate summary statistics for each parameter
                for param in ['EC50', 'log_EC50', 'Emax', 'E0', 'n']:
                    if param in type_data.columns:
                        values = type_data[param].dropna()
                        if len(values) > 0:
                            drug_feature_row[f'hill_{feature_type}_{param}_mean'] = values.mean()
                            drug_feature_row[f'hill_{feature_type}_{param}_std'] = values.std()
                            drug_feature_row[f'hill_{feature_type}_{param}_median'] = values.median()
                
                # Quality metrics
                if 'R2' in type_data.columns:
                    r2_values = type_data['R2'].dropna()
                    if len(r2_values) > 0:
                        drug_feature_row[f'hill_{feature_type}_R2_mean'] = r2_values.mean()
                        drug_feature_row[f'hill_{feature_type}_high_quality_frac'] = (r2_values > 0.7).mean()
        else:
            # Simple aggregation if no feature_type column
            for param in ['EC50', 'log_EC50', 'Emax', 'E0', 'n', 'R2']:
                if param in drug_data.columns:
                    values = drug_data[param].dropna()
                    if len(values) > 0:
                        drug_feature_row[f'hill_{param}_mean'] = values.mean()
                        drug_feature_row[f'hill_{param}_std'] = values.std()
        
        hill_features.append(drug_feature_row)
    
    hill_features_df = pd.DataFrame(hill_features)
    print(f"   Hill parameters: {len(hill_features_df.columns)-1} features for {len(hill_features_df)} drugs")
    return hill_features_df

# ========== HIERARCHICAL FEATURE CONSTRUCTION ==========

def build_hierarchical_features(datasets):
    """Build the comprehensive hierarchical feature matrix"""
    
    print("\n🔄 Building hierarchical feature architecture...")
    
    # Extract drug-level features from each dataset
    feature_dfs = []
    
    # 1. Multi-scale catch22 features
    if len(datasets['multiscale_catch22']) > 0:
        catch22_features = extract_drug_level_features(
            datasets['multiscale_catch22'], 'catch22'
        )
        if len(catch22_features) > 0:
            feature_dfs.append(catch22_features)
    
    # 2. Hierarchical SAX features
    if len(datasets['hierarchical_sax']) > 0:
        sax_features = extract_drug_level_features(
            datasets['hierarchical_sax'], 'sax'
        )
        if len(sax_features) > 0:
            feature_dfs.append(sax_features)
    
    # 3. Event-aware features
    if len(datasets['event_aware']) > 0:
        event_features = extract_drug_level_features(
            datasets['event_aware'], 'event', drug_col='drug', conc_col='concentration'
        )
        if len(event_features) > 0:
            feature_dfs.append(event_features)
    
    # 4. Inter-event features (aggregate across period types)
    if len(datasets['inter_event']) > 0:
        # Separate baseline and inter-event features
        baseline_inter = datasets['inter_event'][datasets['inter_event']['period_type'] == 'baseline']
        inter_event_inter = datasets['inter_event'][datasets['inter_event']['period_type'] == 'inter_event']
        
        if len(baseline_inter) > 0:
            baseline_features = extract_drug_level_features(
                baseline_inter, 'baseline', drug_col='drug', conc_col='concentration'
            )
            if len(baseline_features) > 0:
                feature_dfs.append(baseline_features)
        
        if len(inter_event_inter) > 0:
            inter_features = extract_drug_level_features(
                inter_event_inter, 'inter_event', drug_col='drug', conc_col='concentration'
            )
            if len(inter_features) > 0:
                feature_dfs.append(inter_features)
    
    # 5. Hill parameters
    if len(datasets['hill_parameters']) > 0:
        hill_features = extract_hill_parameters_by_drug(datasets['hill_parameters'])
        if len(hill_features) > 0:
            feature_dfs.append(hill_features)
    
    # 6. Comprehensive Hill parameters
    if len(datasets['comprehensive_hill']) > 0:
        comp_hill_features = extract_hill_parameters_by_drug(datasets['comprehensive_hill'])
        if len(comp_hill_features) > 0:
            feature_dfs.append(comp_hill_features)
    
    # Merge all feature datasets
    if len(feature_dfs) == 0:
        print("   ERROR: No feature datasets available for hierarchical construction")
        return pd.DataFrame(), {}
    
    print(f"\n🔗 Merging {len(feature_dfs)} feature datasets...")
    
    # Start with the first dataset
    hierarchical_features = feature_dfs[0].copy()
    
    # Merge subsequent datasets
    for i, df in enumerate(feature_dfs[1:], 1):
        print(f"   Merging dataset {i+1}: {df.shape}")
        hierarchical_features = hierarchical_features.merge(df, on='drug', how='outer')
    
    print(f"\n✓ Final hierarchical features: {hierarchical_features.shape}")
    print(f"   Drugs: {len(hierarchical_features)}")
    print(f"   Features: {len(hierarchical_features.columns) - 1}")
    
    # Create feature metadata
    feature_metadata = create_feature_metadata(hierarchical_features)
    
    return hierarchical_features, feature_metadata

def create_feature_metadata(features_df):
    """Create comprehensive metadata for all features"""
    metadata = {
        'total_features': len(features_df.columns) - 1,
        'total_drugs': len(features_df),
        'feature_categories': {},
        'feature_descriptions': {},
        'feature_types': {}
    }
    
    # Categorize features by type
    feature_cols = [col for col in features_df.columns if col != 'drug']
    
    categories = {
        'catch22': [col for col in feature_cols if 'catch22' in col],
        'sax': [col for col in feature_cols if 'sax' in col],
        'baseline': [col for col in feature_cols if 'baseline' in col],
        'inter_event': [col for col in feature_cols if 'inter_event' in col],
        'hill_parameters': [col for col in feature_cols if 'hill' in col],
        'quality': [col for col in feature_cols if any(q in col for q in ['quality', 'R2', 'success'])],
        'event_aware': [col for col in feature_cols if 'event' in col and 'inter_event' not in col and 'baseline' not in col]
    }
    
    # Remove overlaps (assign to most specific category)
    assigned_features = set()
    for category, features in categories.items():
        categories[category] = [f for f in features if f not in assigned_features]
        assigned_features.update(categories[category])
    
    # Add uncategorized features
    uncategorized = [col for col in feature_cols if col not in assigned_features]
    if uncategorized:
        categories['other'] = uncategorized
    
    metadata['feature_categories'] = {k: v for k, v in categories.items() if v}
    
    # Add feature counts by category
    for category, features in metadata['feature_categories'].items():
        metadata[f'{category}_count'] = len(features)
    
    return metadata

# ========== QUALITY CONTROL AND PREPROCESSING ==========

def preprocess_hierarchical_features(features_df, metadata):
    """Apply quality control and preprocessing to hierarchical features"""
    
    print("\n🔧 Preprocessing hierarchical features...")
    
    # Remove features with too many missing values
    feature_cols = [col for col in features_df.columns if col != 'drug']
    missing_threshold = 0.8  # Remove features missing in >80% of drugs (more lenient)
    
    features_to_keep = []
    for col in feature_cols:
        missing_frac = features_df[col].isna().mean()
        if missing_frac <= missing_threshold:
            features_to_keep.append(col)
        else:
            print(f"   Removing {col}: {missing_frac:.2f} missing")
    
    processed_df = features_df[['drug'] + features_to_keep].copy()
    print(f"   Kept {len(features_to_keep)} / {len(feature_cols)} features")
    
    # If no features pass the threshold, use a more selective approach
    if len(features_to_keep) == 0:
        print("   No features passed missing data threshold - using selective approach")
        
        # Focus on most complete feature categories
        priority_prefixes = ['hill_', 'event_', 'sax_', 'catch22_']
        
        for prefix in priority_prefixes:
            prefix_features = [col for col in feature_cols if col.startswith(prefix)]
            for col in prefix_features:
                missing_frac = features_df[col].isna().mean()
                if missing_frac <= 0.9:  # Even more lenient for priority features
                    features_to_keep.append(col)
        
        if len(features_to_keep) > 0:
            processed_df = features_df[['drug'] + features_to_keep].copy()
            print(f"   Selective approach: kept {len(features_to_keep)} features")
        else:
            print("   ERROR: No usable features found!")
            return features_df[['drug']].copy(), metadata
    
    # Handle remaining missing values with median imputation
    for col in features_to_keep:
        if processed_df[col].isna().any() or np.isinf(processed_df[col]).any():
            # Replace infinite values with NaN first
            processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
            
            median_val = processed_df[col].median()
            if not np.isnan(median_val):
                processed_df[col] = processed_df[col].fillna(median_val)
                if processed_df[col].isna().any():
                    print(f"   Imputed {col} with median: {median_val:.3f}")
            else:
                # If median is NaN, use 0 as fallback
                processed_df[col] = processed_df[col].fillna(0)
                print(f"   Imputed {col} with 0 (median was NaN)")
    
    # Remove features with zero variance
    numeric_features = features_to_keep
    zero_var_features = []
    
    for col in numeric_features:
        if processed_df[col].std() == 0:
            zero_var_features.append(col)
    
    if zero_var_features:
        processed_df = processed_df.drop(columns=zero_var_features)
        print(f"   Removed {len(zero_var_features)} zero-variance features")
    
    # Update metadata
    final_features = [col for col in processed_df.columns if col != 'drug']
    metadata['processed_features'] = len(final_features)
    metadata['removed_features'] = len(feature_cols) - len(final_features)
    
    print(f"   Final processed features: {len(final_features)}")
    
    return processed_df, metadata

# ========== DIMENSIONALITY REDUCTION AND VISUALIZATION ==========

def create_dimensionality_reduction_embeddings(features_df):
    """Create PCA and UMAP embeddings of the hierarchical features"""
    
    print("\n📊 Creating dimensionality reduction embeddings...")
    
    # Prepare data
    drug_names = features_df['drug'].values
    feature_matrix = features_df.drop(columns=['drug']).values
    feature_names = [col for col in features_df.columns if col != 'drug']
    
    # Final check for infinite values
    if np.isinf(feature_matrix).any():
        print("   WARNING: Infinite values detected, replacing with median")
        for i in range(feature_matrix.shape[1]):
            col_data = feature_matrix[:, i]
            if np.isinf(col_data).any():
                median_val = np.nanmedian(col_data[~np.isinf(col_data)])
                if np.isnan(median_val):
                    median_val = 0
                feature_matrix[np.isinf(col_data), i] = median_val
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    embeddings = {}
    
    # PCA
    pca = PCA(n_components=min(50, len(feature_names)))
    pca_embeddings = pca.fit_transform(feature_matrix_scaled)
    
    embeddings['pca'] = {
        'embeddings': pca_embeddings,
        'explained_variance': pca.explained_variance_ratio_,
        'components': pca.components_,
        'feature_names': feature_names
    }
    
    print(f"   PCA: {pca_embeddings.shape[1]} components, {pca.explained_variance_ratio_[:5].sum():.2f} variance in top 5")
    
    # UMAP
    if len(drug_names) > 10:  # Need sufficient samples for UMAP
        try:
            umap_reducer = umap.UMAP(n_neighbors=min(15, len(drug_names)-1), 
                                   n_components=2, random_state=42, min_dist=0.1)
            umap_embeddings = umap_reducer.fit_transform(feature_matrix_scaled)
            
            embeddings['umap'] = {
                'embeddings': umap_embeddings,
                'reducer': umap_reducer
            }
            print(f"   UMAP: 2D embedding created")
        except Exception as e:
            print(f"   UMAP failed: {e}")
    
    # t-SNE (if not too many samples)
    if len(drug_names) <= 1000:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(drug_names)-1))
            tsne_embeddings = tsne.fit_transform(feature_matrix_scaled)
            
            embeddings['tsne'] = {
                'embeddings': tsne_embeddings
            }
            print(f"   t-SNE: 2D embedding created")
        except Exception as e:
            print(f"   t-SNE failed: {e}")
    
    return embeddings, drug_names, scaler

# ========== MAIN EXECUTION ==========

print("\n📊 Loading feature datasets...")
datasets = load_feature_datasets()

# Build hierarchical features
hierarchical_features, metadata = build_hierarchical_features(datasets)

if len(hierarchical_features) == 0:
    print("\n❌ No hierarchical features could be constructed!")
    exit(1)

# Preprocess features
processed_features, updated_metadata = preprocess_hierarchical_features(hierarchical_features, metadata)

# Create embeddings
embeddings, drug_names, scaler = create_dimensionality_reduction_embeddings(processed_features)

# Save results
print("\n💾 Saving hierarchical feature results...")

# Save feature matrix
processed_features.to_parquet(results_dir / 'hierarchical_drug_embeddings.parquet', index=False)
print(f"   Hierarchical embeddings: {results_dir / 'hierarchical_drug_embeddings.parquet'}")

# Save metadata
with open(results_dir / 'hierarchical_feature_metadata.json', 'w') as f:
    json.dump(updated_metadata, f, indent=2)
print(f"   Feature metadata: {results_dir / 'hierarchical_feature_metadata.json'}")

# Save embeddings
import joblib
joblib.dump({
    'embeddings': embeddings,
    'drug_names': drug_names,
    'scaler': scaler,
    'feature_names': [col for col in processed_features.columns if col != 'drug']
}, results_dir / 'hierarchical_embeddings_results.joblib')
print(f"   Embedding results: {results_dir / 'hierarchical_embeddings_results.joblib'}")

print(f"\n✅ Hierarchical feature architecture complete!")
print(f"\n📊 FINAL SUMMARY:")
print(f"   Total drugs: {len(processed_features)}")
print(f"   Total features: {len(processed_features.columns) - 1}")
print(f"   Feature categories: {len(updated_metadata['feature_categories'])}")

for category, features in updated_metadata['feature_categories'].items():
    print(f"      {category}: {len(features)} features")

print(f"\n🎯 Next steps:")
print(f"   1. Create comprehensive visualizations")
print(f"   2. Apply to DILI correlation analysis")
print(f"   3. Compare with previous embedding approaches")
print(f"   4. Perform drug clustering and mechanism discovery")