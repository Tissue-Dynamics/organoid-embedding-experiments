#!/usr/bin/env python3
"""
Multi-Timescale catch22 Feature Extraction

PURPOSE:
    Extracts 22 canonical time series features (catch22) at multiple temporal
    resolutions to capture both short-term fluctuations and long-term trends
    in oxygen consumption data. Different timescales reveal different aspects
    of drug response dynamics.

METHODOLOGY:
    - Extracts catch22 features using rolling windows at 24h, 48h, and 96h scales
    - Uses 50% window overlap for smooth temporal evolution tracking
    - Aggregates features per well: mean, std, min, max, range, and temporal trend
    - Further aggregates to drug-concentration level for analysis
    - Tracks media change events within each window

INPUTS:
    - Database connection via DATABASE_URL environment variable
    - Queries raw oxygen consumption data from raw_data table
    - Optional: results/data/improved_media_change_events.parquet
      If available, marks media change timepoints in windows

OUTPUTS:
    - results/data/multiscale_catch22_windows.parquet
      Window-level features for all timescales (most detailed)
    - results/data/multiscale_catch22_wells.parquet
      Well-level aggregated features by timescale
    - results/data/multiscale_catch22_drugs.parquet
      Drug-concentration level features combining all timescales

REQUIREMENTS:
    - numpy, pandas, pycatch22, duckdb, tqdm
    - Database connection with oxygen consumption data
    - Minimum 10 timepoints per window for catch22 calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pycatch22
from tqdm import tqdm
import warnings
from datetime import timedelta
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)
fig_dir = project_root / "results" / "figures" / "multiscale_catch22"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MULTI-TIMESCALE CATCH22 FEATURE EXTRACTION")
print("=" * 80)

def extract_catch22_features(time_series):
    """
    Extract catch22 features from a time series
    Returns dict with feature names and values
    """
    if len(time_series) < 10:  # Need minimum points
        return None
    
    # Remove NaN values
    clean_ts = time_series[~np.isnan(time_series)]
    if len(clean_ts) < 10:
        return None
    
    try:
        # Extract features
        result = pycatch22.catch22_all(clean_ts)
        
        # Create feature dictionary
        features = {}
        for name, value in zip(result['names'], result['values']):
            # Handle any inf or nan values
            if np.isfinite(value):
                features[name] = value
            else:
                features[name] = 0.0
        
        return features
    
    except Exception as e:
        print(f"Error extracting catch22 features: {e}")
        return None

def extract_rolling_window_features(df, window_hours, overlap_fraction=0.5):
    """
    Extract catch22 features using rolling windows
    
    Args:
        df: DataFrame with 'elapsed_hours' and 'o2' columns
        window_hours: Size of rolling window in hours
        overlap_fraction: Fraction of window to overlap (0.5 = 50% overlap)
    
    Returns:
        List of dicts with window info and features
    """
    # Sort by time
    df = df.sort_values('elapsed_hours').copy()
    
    # Calculate step size
    step_hours = window_hours * (1 - overlap_fraction)
    
    # Extract windows
    window_features = []
    start_hour = 0
    
    while start_hour + window_hours <= df['elapsed_hours'].max():
        # Get data in window
        mask = (df['elapsed_hours'] >= start_hour) & (df['elapsed_hours'] < start_hour + window_hours)
        window_data = df[mask]
        
        if len(window_data) >= 10:  # Minimum points for catch22
            # Extract features
            features = extract_catch22_features(window_data['o2'].values)
            
            if features is not None:
                # Add window metadata
                window_info = {
                    'window_start_hours': start_hour,
                    'window_end_hours': start_hour + window_hours,
                    'window_center_hours': start_hour + window_hours/2,
                    'window_size_hours': window_hours,
                    'n_points': len(window_data),
                    'o2_mean': window_data['o2'].mean(),
                    'o2_std': window_data['o2'].std(),
                    'o2_min': window_data['o2'].min(),
                    'o2_max': window_data['o2'].max()
                }
                
                # Check for media changes in window
                if 'is_media_change' in window_data.columns:
                    window_info['n_media_changes'] = window_data['is_media_change'].sum()
                else:
                    window_info['n_media_changes'] = 0
                
                # Combine metadata and features
                window_info.update(features)
                window_features.append(window_info)
        
        # Move to next window
        start_hour += step_hours
    
    return window_features

def visualize_rolling_windows(well_data, well_id, drug, concentration, save_path):
    """
    Visualize time series with rolling windows at different timescales
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Multi-Timescale Rolling Windows - {drug} ({concentration} μM)\nWell: {well_id}', 
                 fontsize=16, fontweight='bold')
    
    # Original time series
    ax = axes[0]
    ax.plot(well_data['elapsed_hours'], well_data['o2'], 'b-', alpha=0.7, linewidth=1)
    ax.set_ylabel('O₂ (%)')
    ax.set_title('Original Time Series')
    ax.grid(True, alpha=0.3)
    
    # Mark media changes if available
    if 'is_media_change' in well_data.columns:
        media_times = well_data[well_data['is_media_change']]['elapsed_hours']
        for t in media_times:
            ax.axvline(t, color='red', alpha=0.5, linestyle='--', linewidth=1)
    
    # Show windows for each timescale
    colors = ['red', 'green', 'purple']
    timescales = {'24h': 24, '48h': 48, '96h': 96}
    
    for idx, (scale_name, window_hours) in enumerate(timescales.items()):
        ax = axes[idx + 1]
        
        # Plot time series
        ax.plot(well_data['elapsed_hours'], well_data['o2'], 'b-', alpha=0.3, linewidth=1)
        
        # Show windows
        window_features = extract_rolling_window_features(well_data, window_hours)
        
        for i, window in enumerate(window_features[:10]):  # Show first 10 windows
            start = window['window_start_hours']
            end = window['window_end_hours']
            
            # Draw window rectangle
            ax.axvspan(start, end, alpha=0.2, color=colors[idx], 
                      label=f'{scale_name} window' if i == 0 else '')
            
            # Add window center line
            center = window['window_center_hours']
            ax.axvline(center, color=colors[idx], alpha=0.5, linestyle=':', linewidth=1)
        
        ax.set_ylabel('O₂ (%)')
        ax.set_title(f'{scale_name} Windows (50% overlap)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Elapsed Hours')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_feature_evolution(features_df, drug, save_path):
    """
    Visualize how catch22 features evolve over time at different timescales
    """
    # Select important catch22 features to visualize
    key_features = ['DN_HistogramMode_5', 'DN_HistogramMode_10', 
                    'CO_f1ecac', 'CO_FirstMin_ac', 
                    'SP_Summaries_welch_rect_area_5_1', 
                    'SB_BinaryStats_mean_longstretch1']
    
    drug_data = features_df[features_df['drug'] == drug]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Feature Evolution Over Time - {drug}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features[:6]):
        ax = axes[idx]
        
        for scale_name, color in zip(['24h', '48h', '96h'], ['red', 'green', 'purple']):
            scale_data = drug_data[drug_data['timescale'] == scale_name]
            
            if len(scale_data) > 0 and feature in scale_data.columns:
                # Group by concentration
                for conc in sorted(scale_data['concentration'].unique()):
                    conc_data = scale_data[scale_data['concentration'] == conc]
                    
                    ax.scatter(conc_data['window_center_hours'], conc_data[feature], 
                             alpha=0.6, s=30, color=color, 
                             label=f'{scale_name} - {conc:.1f}μM' if idx == 0 else '')
                    
                    # Add trend line
                    if len(conc_data) > 3:
                        z = np.polyfit(conc_data['window_center_hours'], conc_data[feature], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(conc_data['window_center_hours'].min(), 
                                            conc_data['window_center_hours'].max(), 100)
                        ax.plot(x_trend, p(x_trend), color=color, alpha=0.3, linewidth=2)
        
        ax.set_xlabel('Window Center (hours)')
        ax.set_ylabel(feature[:20])
        ax.set_title(feature)
        ax.grid(True, alpha=0.3)
    
    # Add legend to first subplot
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_timescale_comparison(agg_features_df, save_path):
    """
    Compare feature distributions across different timescales
    """
    # Select key features
    key_features = ['DN_HistogramMode_5', 'CO_f1ecac', 'SP_Summaries_welch_rect_area_5_1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Feature Distribution Across Timescales', fontsize=16, fontweight='bold')
    
    for idx, base_feature in enumerate(key_features):
        ax = axes[idx]
        
        # Prepare data for violin plot
        plot_data = []
        for scale in ['24h', '48h', '96h']:
            scale_data = agg_features_df[agg_features_df['timescale'] == scale]
            feature_col = f'{base_feature}_mean'
            
            if feature_col in scale_data.columns:
                values = scale_data[feature_col].dropna()
                for val in values:
                    plot_data.append({'Timescale': scale, 'Value': val})
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(plot_df) > 0:
            sns.violinplot(data=plot_df, x='Timescale', y='Value', ax=ax, 
                          palette=['red', 'green', 'purple'])
            ax.set_title(base_feature[:30])
            ax.set_ylabel('Feature Value')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_feature_correlation_matrix(drug_features_df, save_path):
    """
    Create correlation heatmap of features across timescales
    """
    # Get numeric feature columns
    feature_cols = [col for col in drug_features_df.columns 
                   if col not in ['drug', 'concentration', 'n_wells']]
    
    if len(feature_cols) > 50:  # Limit to top 50 features by variance
        variances = drug_features_df[feature_cols].var()
        feature_cols = variances.nlargest(50).index.tolist()
    
    # Calculate correlation matrix
    corr_matrix = drug_features_df[feature_cols].corr()
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Create custom colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Matrix Across Timescales', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_drug_embeddings(drug_features_df, save_path):
    """
    Create UMAP and t-SNE visualizations of drugs using multi-timescale features
    """
    # Get feature columns
    feature_cols = [col for col in drug_features_df.columns 
                   if col not in ['drug', 'concentration', 'n_wells']]
    
    if len(feature_cols) == 0:
        print("   Warning: No features found for embedding visualization")
        return
    
    # Prepare data matrix
    X = drug_features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create embeddings
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Drug Embeddings using Multi-Timescale catch22 Features', 
                 fontsize=16, fontweight='bold')
    
    # PCA
    ax = axes[0]
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(X_scaled)
    
    scatter = ax.scatter(pca_embedding[:, 0], pca_embedding[:, 1], 
                        c=drug_features_df['concentration'], cmap='viridis', 
                        s=100, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA Projection')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Concentration (μM)')
    
    # t-SNE
    ax = axes[1]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    tsne_embedding = tsne.fit_transform(X_scaled)
    
    scatter = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                        c=drug_features_df['concentration'], cmap='viridis', 
                        s=100, alpha=0.7)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Projection')
    ax.grid(True, alpha=0.3)
    
    # UMAP
    ax = axes[2]
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)-1))
    umap_embedding = reducer.fit_transform(X_scaled)
    
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                        c=drug_features_df['concentration'], cmap='viridis', 
                        s=100, alpha=0.7)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP Projection')
    ax.grid(True, alpha=0.3)
    
    # Add some drug labels
    for i, drug in enumerate(drug_features_df['drug'].values[::5]):  # Every 5th drug
        ax.annotate(drug[:10], (umap_embedding[i*5, 0], umap_embedding[i*5, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_feature_importance_by_variance(drug_features_df, save_path):
    """
    Show which features and timescales have highest variance
    """
    # Calculate variance for each feature
    feature_cols = [col for col in drug_features_df.columns 
                   if col not in ['drug', 'concentration', 'n_wells']]
    
    variances = []
    for col in feature_cols:
        var = drug_features_df[col].var()
        if np.isfinite(var):
            # Extract timescale and feature name
            parts = col.split('_', 1)
            if parts[0] in ['24h', '48h', '96h']:
                variances.append({
                    'feature': col,
                    'timescale': parts[0],
                    'feature_name': parts[1] if len(parts) > 1 else col,
                    'variance': var
                })
    
    var_df = pd.DataFrame(variances)
    
    if len(var_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Importance by Variance', fontsize=16, fontweight='bold')
        
        # Top features overall
        ax = axes[0]
        top_features = var_df.nlargest(20, 'variance')
        
        colors = {'24h': 'red', '48h': 'green', '96h': 'purple'}
        bar_colors = [colors[ts] for ts in top_features['timescale']]
        
        bars = ax.barh(range(len(top_features)), top_features['variance'], color=bar_colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f[:40] for f in top_features['feature']], fontsize=8)
        ax.set_xlabel('Variance')
        ax.set_title('Top 20 Features by Variance')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Variance by timescale
        ax = axes[1]
        timescale_var = var_df.groupby('timescale')['variance'].agg(['mean', 'sum', 'count'])
        
        x = range(len(timescale_var))
        ax.bar(x, timescale_var['mean'], color=['red', 'green', 'purple'], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(timescale_var.index)
        ax.set_ylabel('Mean Variance')
        ax.set_title('Average Feature Variance by Timescale')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add feature counts
        for i, (idx, row) in enumerate(timescale_var.iterrows()):
            ax.text(i, row['mean'], f"n={row['count']}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# Load oxygen consumption data
print("\n📊 Loading oxygen consumption data...")
import sys
sys.path.append(str(project_root))
from src.utils.data_loader import DataLoader

loader = DataLoader()
# Load limited data for faster demonstration
df = loader.load_oxygen_data(limit=3)  # Only 3 plates for demo
print(f"   Loaded {len(df)} measurements")
print(f"   Wells: {df['well_id'].nunique()}")
print(f"   Time range: {df['elapsed_hours'].min():.1f} - {df['elapsed_hours'].max():.1f} hours")

# Load media change events if available
media_changes_file = results_dir / "improved_media_change_events.parquet"
if media_changes_file.exists():
    print("\n📊 Loading media change events...")
    media_changes = pd.read_parquet(media_changes_file)
    
    # Mark media change timepoints by plate (affects all wells on plate)
    df['is_media_change'] = False
    for _, event in media_changes.iterrows():
        mask = (df['plate_id'] == event['plate_id']) & \
               (df['elapsed_hours'] >= event['event_time_hours'] - 1) & \
               (df['elapsed_hours'] <= event['event_time_hours'] + 1)
        df.loc[mask, 'is_media_change'] = True
    
    print(f"   Marked {df['is_media_change'].sum()} timepoints as media changes")

# Define timescales
timescales = {
    '24h': 24,
    '48h': 48, 
    '96h': 96
}

# Process each well at each timescale
print(f"\n🔄 Extracting multi-timescale features...")
all_features = []
example_wells = []  # Store examples for visualization

wells = df['well_id'].unique()
for well_id in tqdm(wells, desc="Processing wells"):
    well_data = df[df['well_id'] == well_id].copy()
    
    # Get drug and concentration info
    drug = well_data['drug'].iloc[0]
    concentration = well_data['concentration'].iloc[0]
    
    # Store first few examples for visualization
    if len(example_wells) < 3 and drug not in ['Unknown', 'DMSO'] and len(well_data) > 100:
        example_wells.append({
            'well_id': well_id,
            'drug': drug,
            'concentration': concentration,
            'data': well_data
        })
    
    # Extract features at each timescale
    for scale_name, window_hours in timescales.items():
        window_features = extract_rolling_window_features(well_data, window_hours)
        
        # Add well metadata to each window
        for window in window_features:
            window['well_id'] = well_id
            window['drug'] = drug
            window['concentration'] = concentration
            window['timescale'] = scale_name
            window['timescale_hours'] = window_hours
            
        all_features.extend(window_features)

# Convert to DataFrame
features_df = pd.DataFrame(all_features)
print(f"\n📊 Extracted {len(features_df)} feature windows")
print(f"   Timescales: {features_df['timescale'].value_counts().to_dict()}")

# Get catch22 feature names
catch22_names = [col for col in features_df.columns 
                 if col not in ['well_id', 'drug', 'concentration', 'timescale', 'timescale_hours',
                               'window_start_hours', 'window_end_hours', 'window_center_hours',
                               'window_size_hours', 'n_points', 'o2_mean', 'o2_std', 'o2_min', 
                               'o2_max', 'n_media_changes']]

print(f"   catch22 features: {len(catch22_names)}")

# Aggregate features by well and timescale
print("\n🔄 Aggregating features by well...")
aggregated_features = []

for (well_id, timescale), group in features_df.groupby(['well_id', 'timescale']):
    drug = group['drug'].iloc[0]
    concentration = group['concentration'].iloc[0]
    
    agg_features = {
        'well_id': well_id,
        'drug': drug,
        'concentration': concentration,
        'timescale': timescale,
        'n_windows': len(group)
    }
    
    # Aggregate each catch22 feature
    for feature in catch22_names:
        if feature in group.columns:
            values = group[feature].dropna()
            if len(values) > 0:
                agg_features[f'{feature}_mean'] = values.mean()
                agg_features[f'{feature}_std'] = values.std()
                agg_features[f'{feature}_min'] = values.min()
                agg_features[f'{feature}_max'] = values.max()
                agg_features[f'{feature}_range'] = values.max() - values.min()
                
                # Temporal trend (linear regression slope)
                if len(values) > 1:
                    x = group['window_center_hours'].values[:len(values)]
                    coef = np.polyfit(x, values, 1)[0]
                    agg_features[f'{feature}_trend'] = coef
    
    # Add overall statistics
    agg_features['o2_global_mean'] = group['o2_mean'].mean()
    agg_features['o2_global_std'] = group['o2_std'].mean()
    agg_features['total_media_changes'] = group['n_media_changes'].sum()
    
    aggregated_features.append(agg_features)

# Convert to DataFrame
agg_features_df = pd.DataFrame(aggregated_features)
print(f"   Created {len(agg_features_df)} aggregated feature sets")

# Create drug-level features by combining timescales
print("\n🔄 Creating drug-level multi-timescale features...")
drug_features = []

for drug in agg_features_df['drug'].unique():
    drug_data = agg_features_df[agg_features_df['drug'] == drug]
    
    # Get all concentrations for this drug
    concentrations = sorted(drug_data['concentration'].unique())
    
    for conc in concentrations:
        conc_data = drug_data[drug_data['concentration'] == conc]
        
        drug_feat = {
            'drug': drug,
            'concentration': conc,
            'n_wells': conc_data['well_id'].nunique()
        }
        
        # Combine features across timescales
        for timescale in timescales.keys():
            scale_data = conc_data[conc_data['timescale'] == timescale]
            
            if len(scale_data) > 0:
                # Get feature columns for this timescale
                feature_cols = [col for col in scale_data.columns 
                               if any(catch22_name in col for catch22_name in catch22_names)]
                
                # Average across wells
                for col in feature_cols:
                    values = scale_data[col].dropna()
                    if len(values) > 0:
                        drug_feat[f'{timescale}_{col}'] = values.mean()
        
        drug_features.append(drug_feat)

drug_features_df = pd.DataFrame(drug_features)
print(f"   Created features for {len(drug_features_df)} drug-concentration pairs")

# Save results
print("\n💾 Saving results...")
features_df.to_parquet(results_dir / 'multiscale_catch22_windows.parquet', index=False)
agg_features_df.to_parquet(results_dir / 'multiscale_catch22_wells.parquet', index=False)
drug_features_df.to_parquet(results_dir / 'multiscale_catch22_drugs.parquet', index=False)

print(f"   Window-level features: {results_dir / 'multiscale_catch22_windows.parquet'}")
print(f"   Well-level features: {results_dir / 'multiscale_catch22_wells.parquet'}")
print(f"   Drug-level features: {results_dir / 'multiscale_catch22_drugs.parquet'}")

# Create summary statistics
print("\n📊 FEATURE SUMMARY:")
print(f"   Total windows extracted: {len(features_df)}")
print(f"   Wells processed: {features_df['well_id'].nunique()}")
print(f"   Drugs processed: {features_df['drug'].nunique()}")

# Show example features for one drug
example_drug = drug_features_df['drug'].iloc[0]
example_features = drug_features_df[drug_features_df['drug'] == example_drug].iloc[0]
feature_cols = [col for col in example_features.index if any(scale in col for scale in timescales.keys())]

print(f"\n🔍 Example features for {example_drug}:")
print(f"   Total multi-timescale features: {len(feature_cols)}")
print(f"   Features per timescale:")
for scale in timescales.keys():
    scale_features = [col for col in feature_cols if col.startswith(f'{scale}_')]
    print(f"      {scale}: {len(scale_features)} features")

# Feature importance preview (which timescales capture most variance)
print("\n📈 Timescale variance analysis:")
for timescale in timescales.keys():
    scale_cols = [col for col in drug_features_df.columns if col.startswith(f'{timescale}_')]
    if scale_cols:
        variances = drug_features_df[scale_cols].var()
        print(f"   {timescale}: mean variance = {variances.mean():.4f}")

# Generate visualizations
print("\n📊 Generating visualizations...")

# 1. Rolling window visualization
if len(example_wells) > 0:
    print("   Creating rolling window visualizations...")
    for i, example in enumerate(example_wells[:2]):
        save_path = fig_dir / f'rolling_windows_example_{i+1}.png'
        visualize_rolling_windows(example['data'], example['well_id'], 
                                example['drug'], example['concentration'], save_path)
        print(f"      Saved: {save_path.name}")

# 2. Feature evolution
if len(features_df) > 0:
    print("   Creating feature evolution visualization...")
    # Pick a drug with multiple concentrations
    drug_counts = features_df.groupby('drug')['concentration'].nunique()
    example_drug = drug_counts[drug_counts > 3].index[0] if any(drug_counts > 3) else features_df['drug'].iloc[0]
    visualize_feature_evolution(features_df, example_drug, fig_dir / 'feature_evolution.png')
    print(f"      Saved: feature_evolution.png")

# 3. Timescale comparison
print("   Creating timescale comparison visualization...")
visualize_timescale_comparison(agg_features_df, fig_dir / 'timescale_comparison.png')
print(f"      Saved: timescale_comparison.png")

# 4. Feature correlation matrix
print("   Creating feature correlation matrix...")
visualize_feature_correlation_matrix(drug_features_df, fig_dir / 'feature_correlation_matrix.png')
print(f"      Saved: feature_correlation_matrix.png")

# 5. Drug embeddings
print("   Creating drug embedding visualizations...")
visualize_drug_embeddings(drug_features_df, fig_dir / 'drug_embeddings.png')
print(f"      Saved: drug_embeddings.png")

# 6. Feature importance by variance
print("   Creating feature importance visualization...")
visualize_feature_importance_by_variance(drug_features_df, fig_dir / 'feature_importance.png')
print(f"      Saved: feature_importance.png")

print("\n✅ Multi-timescale catch22 extraction complete!")
print(f"   Generated {len(all_features)} window-level features")
print(f"   Generated {len(agg_features_df)} well-timescale features")
print(f"   Generated {len(drug_features_df)} drug-concentration features")
print(f"   Created 8 visualization figures in: {fig_dir}")
print(f"\n   Next steps:")
print(f"   1. Correlate with DILI using multi-timescale features")
print(f"   2. Compare predictive power of different timescales")
print(f"   3. Combine with SAX features for hybrid representation")
print(f"   4. Use for Hill curve fitting at each timescale")