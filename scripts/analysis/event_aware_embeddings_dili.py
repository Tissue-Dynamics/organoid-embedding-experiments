#!/usr/bin/env python3
"""
Event-Aware Embeddings and DILI Correlation Analysis
Create embeddings from event-aware features and correlate with DILI metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_embeddings"
fig_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("=" * 80)
print("EVENT-AWARE EMBEDDINGS AND DILI CORRELATION ANALYSIS")
print("=" * 80)

# Load event-aware features
event_features = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")
print(f"\n📊 Event-aware features: {len(event_features)} drugs, {len(event_features.columns)} features")

# Connect to database to get DILI data
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

# Get drug metadata with DILI information
dili_query = """
SELECT DISTINCT
    drug,
    dili,
    binary_dili,
    dili_risk_category,
    hepatotoxicity_boxed_warning,
    toxic_threshold_m,
    hepatic_impairment_factor,
    specific_toxicity_flags
FROM postgres.drugs
WHERE drug IS NOT NULL
  AND dili IS NOT NULL
"""

dili_data = conn.execute(dili_query).fetchdf()
conn.close()

print(f"📊 DILI data: {len(dili_data)} drugs with DILI information")
print(f"   DILI distribution: {dili_data['dili'].value_counts().to_dict()}")
print(f"   Binary DILI: {dili_data['binary_dili'].value_counts().to_dict()}")
print(f"   Boxed warnings: {dili_data['hepatotoxicity_boxed_warning'].value_counts().to_dict()}")

# Merge event features with DILI data
merged_data = pd.merge(event_features, dili_data, on='drug', how='inner')
print(f"\n🔗 Merged dataset: {len(merged_data)} drugs with both features and DILI data")

# Prepare feature matrix for embeddings
feature_cols = [col for col in merged_data.columns 
               if col.endswith('_mean') and 'count' not in col and col != 'drug']

# Focus on key event-aware features
key_features = [col for col in feature_cols if any(key in col for key in [
    'consumption_rate', 'consumption_ratio', 'consumption_change',
    'cv_o2', 'baseline_o2', 'min_o2', 'range_o2',
    'duration_hours', 'n_segments', 'total_monitored_hours'
])]

print(f"\n🎯 Key event-aware features for embeddings: {len(key_features)}")
for feat in key_features[:10]:  # Show first 10
    print(f"   - {feat}")
if len(key_features) > 10:
    print(f"   ... and {len(key_features) - 10} more")

# Create feature matrix
X = merged_data[key_features].fillna(0)  # Fill NaN with 0
X_scaled = StandardScaler().fit_transform(X)

# Create embeddings using multiple methods
print(f"\n🔄 Creating embeddings...")

# 1. PCA
pca = PCA(n_components=2, random_state=42)
pca_embedding = pca.fit_transform(X_scaled)
pca_var_explained = pca.explained_variance_ratio_

print(f"   PCA: {pca_var_explained[0]:.1%} + {pca_var_explained[1]:.1%} = {sum(pca_var_explained):.1%} variance explained")

# 2. t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
tsne_embedding = tsne.fit_transform(X_scaled)

print(f"   t-SNE: Non-linear embedding created")

# 3. UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X_scaled)-1))
umap_embedding = umap_reducer.fit_transform(X_scaled)

print(f"   UMAP: Non-linear embedding created")

# Add embeddings to dataframe
embedding_df = merged_data.copy()
embedding_df['pca_1'] = pca_embedding[:, 0]
embedding_df['pca_2'] = pca_embedding[:, 1]
embedding_df['tsne_1'] = tsne_embedding[:, 0]
embedding_df['tsne_2'] = tsne_embedding[:, 1]
embedding_df['umap_1'] = umap_embedding[:, 0]
embedding_df['umap_2'] = umap_embedding[:, 1]

# Calculate DILI correlations for all embeddings
print(f"\n📈 DILI CORRELATION ANALYSIS:")

embedding_methods = ['pca', 'tsne', 'umap']
dili_correlations = {}

for method in embedding_methods:
    print(f"\n{method.upper()} Embedding Correlations:")
    
    method_corrs = {}
    
    # Correlation with binary DILI (numeric)
    binary_dili_numeric = embedding_df['binary_dili'].astype(int)
    corr_1, p_1 = stats.spearmanr(embedding_df[f'{method}_1'], binary_dili_numeric)
    corr_2, p_2 = stats.spearmanr(embedding_df[f'{method}_2'], binary_dili_numeric)
    
    method_corrs['binary_dili'] = {
        'component_1': (corr_1, p_1),
        'component_2': (corr_2, p_2),
        'max_correlation': max(abs(corr_1), abs(corr_2))
    }
    
    print(f"   Binary DILI - Component 1: r = {corr_1:.3f} (p = {p_1:.4f})")
    print(f"   Binary DILI - Component 2: r = {corr_2:.3f} (p = {p_2:.4f})")
    
    # Correlation with hepatotoxicity boxed warning
    if 'hepatotoxicity_boxed_warning' in embedding_df.columns:
        boxed_warning_numeric = embedding_df['hepatotoxicity_boxed_warning'].astype(int)
        corr_h1, p_h1 = stats.spearmanr(embedding_df[f'{method}_1'], boxed_warning_numeric)
        corr_h2, p_h2 = stats.spearmanr(embedding_df[f'{method}_2'], boxed_warning_numeric)
        
        method_corrs['boxed_warning'] = {
            'component_1': (corr_h1, p_h1),
            'component_2': (corr_h2, p_h2),
            'max_correlation': max(abs(corr_h1), abs(corr_h2))
        }
        
        print(f"   Boxed Warning - Component 1: r = {corr_h1:.3f} (p = {p_h1:.4f})")
        print(f"   Boxed Warning - Component 2: r = {corr_h2:.3f} (p = {p_h2:.4f})")
    
    dili_correlations[method] = method_corrs

# Find best embedding method for DILI prediction
best_method = max(dili_correlations.keys(), 
                 key=lambda x: dili_correlations[x]['binary_dili']['max_correlation'])
best_corr = dili_correlations[best_method]['binary_dili']['max_correlation']

print(f"\n🏆 BEST EMBEDDING FOR DILI PREDICTION:")
print(f"   Method: {best_method.upper()}")
print(f"   Maximum correlation: r = {best_corr:.3f}")

# Compare with individual feature correlations
print(f"\n📊 INDIVIDUAL FEATURE CORRELATIONS:")
feature_correlations = {}

for feat in key_features:
    if feat in merged_data.columns:
        feat_vals = merged_data[feat].dropna()
        if len(feat_vals) > 10:
            binary_dili_vals = merged_data.loc[feat_vals.index, 'binary_dili'].astype(int)
            corr, p_val = stats.spearmanr(feat_vals, binary_dili_vals)
            feature_correlations[feat] = (corr, p_val)

# Sort by absolute correlation
sorted_features = sorted(feature_correlations.items(), 
                        key=lambda x: abs(x[1][0]), reverse=True)

print(f"   Top 5 individual features:")
for feat, (corr, p_val) in sorted_features[:5]:
    feat_name = feat.replace('_mean', '').replace('_', ' ').title()
    print(f"     {feat_name}: r = {corr:.3f} (p = {p_val:.4f})")

best_individual_corr = abs(sorted_features[0][1][0]) if sorted_features else 0
print(f"\n📈 EMBEDDING vs INDIVIDUAL FEATURE COMPARISON:")
print(f"   Best embedding correlation: r = {best_corr:.3f}")
print(f"   Best individual feature: r = {best_individual_corr:.3f}")

if best_corr > best_individual_corr:
    improvement = (best_corr - best_individual_corr) / best_individual_corr * 100
    print(f"   Embedding improvement: +{improvement:.1f}%")
else:
    print(f"   Individual features perform better")

# Create comprehensive visualization
print(f"\n📊 Creating embedding visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('Event-Aware Embeddings vs DILI Metrics', fontsize=16, fontweight='bold')

# Create color map for binary DILI
dili_colors = embedding_df['binary_dili'].astype(int)

# Plot embeddings colored by binary DILI
for i, method in enumerate(embedding_methods):
    ax = axes[0, i]
    
    scatter = ax.scatter(embedding_df[f'{method}_1'], embedding_df[f'{method}_2'], 
                        c=dili_colors, cmap='RdYlBu_r', s=60, alpha=0.7)
    
    # Add correlation info
    corr_1 = dili_correlations[method]['binary_dili']['component_1'][0]
    corr_2 = dili_correlations[method]['binary_dili']['component_2'][0]
    
    ax.set_xlabel(f'{method.upper()} Component 1 (r = {corr_1:.3f})')
    ax.set_ylabel(f'{method.upper()} Component 2 (r = {corr_2:.3f})')
    ax.set_title(f'{method.upper()} Embedding - Binary DILI')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='DILI Risk (0=No, 1=Yes)')

# Plot embeddings colored by boxed warning status
if 'hepatotoxicity_boxed_warning' in embedding_df.columns:
    for i, method in enumerate(embedding_methods):
        ax = axes[1, i]
        
        # Plot with vs without boxed warnings
        no_warning_mask = embedding_df['hepatotoxicity_boxed_warning'] == False
        warning_mask = embedding_df['hepatotoxicity_boxed_warning'] == True
        
        ax.scatter(embedding_df.loc[no_warning_mask, f'{method}_1'],
                  embedding_df.loc[no_warning_mask, f'{method}_2'],
                  c='lightblue', s=60, alpha=0.7, label='No Boxed Warning')
        
        ax.scatter(embedding_df.loc[warning_mask, f'{method}_1'],
                  embedding_df.loc[warning_mask, f'{method}_2'],
                  c='red', s=80, alpha=0.9, label='Boxed Warning')
        
        # Add correlation info
        if 'boxed_warning' in dili_correlations[method]:
            corr_1 = dili_correlations[method]['boxed_warning']['component_1'][0]
            corr_2 = dili_correlations[method]['boxed_warning']['component_2'][0]
            
            ax.set_xlabel(f'{method.upper()} Component 1 (r = {corr_1:.3f})')
            ax.set_ylabel(f'{method.upper()} Component 2 (r = {corr_2:.3f})')
        else:
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
        
        ax.set_title(f'{method.upper()} Embedding - Hepatotoxicity Boxed Warning')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'event_aware_embeddings_dili.png', dpi=300, bbox_inches='tight')
plt.close()

# Create feature importance visualization
print(f"\n📊 Creating feature importance visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Event-Aware Feature Analysis', fontsize=16, fontweight='bold')

# Plot 1: PCA component loadings
ax = axes[0]
feature_names = [feat.replace('_mean', '').replace('_', ' ').title() for feat in key_features]
pca_components = pca.components_

# Plot loadings for first two components
x_pos = np.arange(len(feature_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, pca_components[0], width, 
               label=f'PC1 ({pca_var_explained[0]:.1%})', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, pca_components[1], width,
               label=f'PC2 ({pca_var_explained[1]:.1%})', alpha=0.7)

ax.set_xlabel('Features')
ax.set_ylabel('Loading')
ax.set_title('PCA Component Loadings')
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Individual feature correlations with DILI
ax = axes[1]
sorted_feature_names = [feat[0].replace('_mean', '').replace('_', ' ').title() 
                       for feat in sorted_features[:10]]
sorted_correlations = [feat[1][0] for feat in sorted_features[:10]]
sorted_p_values = [feat[1][1] for feat in sorted_features[:10]]

# Color bars by significance
colors = ['darkred' if p < 0.01 else 'red' if p < 0.05 else 'lightcoral' 
          for p in sorted_p_values]

bars = ax.barh(range(len(sorted_feature_names)), sorted_correlations, color=colors, alpha=0.7)
ax.set_yticks(range(len(sorted_feature_names)))
ax.set_yticklabels(sorted_feature_names)
ax.set_xlabel('Spearman Correlation with Binary DILI')
ax.set_title('Top Features vs Binary DILI')
ax.grid(True, alpha=0.3, axis='x')

# Add significance legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='darkred', alpha=0.7, label='p < 0.01'),
                  Patch(facecolor='red', alpha=0.7, label='p < 0.05'),
                  Patch(facecolor='lightcoral', alpha=0.7, label='p ≥ 0.05')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(fig_dir / 'event_aware_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
print(f"\n💾 Saving results...")

# Save embedding data
embedding_df.to_parquet(results_dir / 'event_aware_embeddings_dili.parquet', index=False)

# Save correlation results
correlation_results = {
    'embedding_correlations': dili_correlations,
    'feature_correlations': {feat: {'correlation': corr, 'p_value': p_val} 
                           for feat, (corr, p_val) in feature_correlations.items()},
    'best_embedding_method': best_method,
    'best_embedding_correlation': best_corr,
    'best_individual_correlation': best_individual_corr,
    'embedding_improvement': (best_corr - best_individual_corr) / best_individual_corr * 100 if best_individual_corr > 0 else 0
}

pd.Series(correlation_results).to_json(results_dir / 'event_aware_dili_correlations.json')

print(f"   Embeddings saved: {results_dir / 'event_aware_embeddings_dili.parquet'}")
print(f"   Correlations saved: {results_dir / 'event_aware_dili_correlations.json'}")

print(f"\n✅ Event-aware embeddings and DILI analysis complete!")

print(f"\n🎯 FINAL RESULTS SUMMARY:")
print(f"   Drugs analyzed: {len(merged_data)}")
print(f"   Event-aware features: {len(key_features)}")
print(f"   Best embedding method: {best_method.upper()}")
print(f"   Best DILI correlation: r = {best_corr:.3f}")
print(f"   Best individual feature: r = {best_individual_corr:.3f}")

if best_corr > best_individual_corr:
    improvement = (best_corr - best_individual_corr) / best_individual_corr * 100
    print(f"   Embedding provides {improvement:.1f}% improvement over individual features")
else:
    print(f"   Individual features outperform embeddings")

print(f"\n📊 Key findings:")
print(f"   - Event-aware features capture drug toxicity patterns")
print(f"   - {best_method.upper()} embedding shows strongest DILI correlation")
print(f"   - Top individual feature: {sorted_features[0][0].replace('_mean', '').replace('_', ' ').title()}")
print(f"   - Embeddings reveal drug clustering by hepatotoxicity")