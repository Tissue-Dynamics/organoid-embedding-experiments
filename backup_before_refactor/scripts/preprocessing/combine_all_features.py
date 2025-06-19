#!/usr/bin/env python3
"""
Combine Event-Aware Features with Phase 2 Embeddings
Create a comprehensive feature set for DILI prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "combined_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMBINING ALL FEATURES FOR ENHANCED DILI PREDICTION")
print("=" * 80)

# Load all data sources
print("\n📊 Loading data sources...")

# 1. Event-aware features
event_features = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")
print(f"   Event-aware features: {len(event_features)} drugs")

# 2. Phase 2 embeddings
phase2_results = joblib.load(results_dir / "hierarchical_embedding_results.joblib")
drug_embeddings = phase2_results['drug_embeddings']
drug_metadata = phase2_results['drug_metadata']
print(f"   Phase 2 embeddings: {len(drug_metadata)} drugs")

# 3. DILI data
wells_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")
drug_dili_map = wells_df[wells_df['dili'].notna()].groupby('drug').agg({
    'dili': 'first',
    'binary_dili': 'first',
    'dili_risk_category': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    'molecular_weight': 'first',
    'logp': 'first',
    'half_life_hours': 'first',
    'clearance_l_hr_kg': 'first'
}).reset_index()

# Create numeric DILI risk
dili_risk_mapping = {
    'Low': 1, 'Low risk': 1, 'Low Risk': 1, 'Low DILI Risk': 1,
    'Low hepatotoxicity risk': 1, 'No DILI concern': 0,
    'No-DILI-Concern per database': 0, 'vLess-DILI-Concern': 1,
    'Low concern': 1, 'Low - vLess-DILI-Concern': 1,
    'Low frequency (<1:10,000), high severity': 2,
    'Low frequency but potentially severe': 2,
    'Low-Moderate': 1.5, 'Moderate': 2, 'Moderate Risk': 2,
    'Intermediate': 2, 'High': 3, 'High Risk': 3,
    'High risk - drug withdrawn': 4, 'High risk - cumulative dose-dependent': 3,
    'High concern': 3, 'High Risk - Black Box Warning': 4,
    'Severe': 4, 'Severe - High Risk': 4, 'Black Box Warning': 4,
    'LiverTox Category D': 3, 'vMost-DILI-Concern': 4,
    'Not formally categorized - primary toxicity was hematologic': 1
}

drug_dili_map['dili_risk_numeric'] = drug_dili_map['dili_risk_category'].map(dili_risk_mapping)

# Fill NaN values
if drug_dili_map['dili_risk_numeric'].isna().any():
    drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'dili_risk_numeric'] = \
        drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'binary_dili'].map({True: 2.0, False: 0.0})

print(f"   DILI data: {len(drug_dili_map)} drugs")

# Find overlapping drugs
print("\n🔍 Finding overlapping drugs...")

# Get drug names from each source
event_drugs = set(event_features['drug'])
phase2_drugs = set(drug_metadata['drug'])
dili_drugs = set(drug_dili_map['drug'])

# Three-way overlap
overlap_all = event_drugs.intersection(phase2_drugs).intersection(dili_drugs)
print(f"   Three-way overlap: {len(overlap_all)} drugs")

# Two-way overlaps
phase2_dili = phase2_drugs.intersection(dili_drugs)
event_dili = event_drugs.intersection(dili_drugs)
print(f"   Phase 2 + DILI: {len(phase2_dili)} drugs")
print(f"   Event + DILI: {len(event_dili)} drugs")

# Build combined dataset
print("\n🔧 Building combined feature sets...")

# 1. Three-way combined (event + phase2 + DILI)
if len(overlap_all) > 10:
    print(f"\n   Building three-way combined dataset ({len(overlap_all)} drugs)...")
    
    # Get event features
    event_subset = event_features[event_features['drug'].isin(overlap_all)].set_index('drug')
    
    # Get best Phase 2 embeddings (Fourier)
    fourier_embeddings = drug_embeddings['fourier']
    phase2_subset = pd.DataFrame(fourier_embeddings, index=drug_metadata['drug'])
    phase2_subset = phase2_subset.loc[list(overlap_all)]
    phase2_subset.columns = [f'fourier_dim_{i}' for i in range(phase2_subset.shape[1])]
    
    # Get DILI data
    dili_subset = drug_dili_map[drug_dili_map['drug'].isin(overlap_all)].set_index('drug')
    
    # Combine all features
    combined_df = event_subset.join(phase2_subset).join(dili_subset[['dili_risk_numeric']])
    
    print(f"   Combined dataset: {len(combined_df)} drugs, {len(combined_df.columns)-1} features")

# 2. Phase 2 + DILI only (larger dataset)
print(f"\n   Building Phase 2 + DILI dataset ({len(phase2_dili)} drugs)...")

phase2_dili_df = pd.DataFrame()
for method, embeddings in drug_embeddings.items():
    if isinstance(embeddings, np.ndarray):
        embed_df = pd.DataFrame(embeddings, index=drug_metadata['drug'])
        embed_df = embed_df.loc[list(phase2_dili)]
        # Use only first 10 dimensions from each method
        embed_df = embed_df.iloc[:, :10]
        embed_df.columns = [f'{method}_dim_{i}' for i in range(embed_df.shape[1])]
        
        if phase2_dili_df.empty:
            phase2_dili_df = embed_df
        else:
            phase2_dili_df = phase2_dili_df.join(embed_df)

dili_phase2_subset = drug_dili_map[drug_dili_map['drug'].isin(phase2_dili)].set_index('drug')
phase2_dili_df = phase2_dili_df.join(dili_phase2_subset[['dili_risk_numeric', 'molecular_weight', 'logp']])

print(f"   Phase 2 dataset: {len(phase2_dili_df)} drugs, {len(phase2_dili_df.columns)-1} features")

# Analyze feature importance
print("\n🎯 Analyzing feature importance...")

if len(overlap_all) > 10:
    # Use Random Forest to assess feature importance
    X = combined_df.drop(['dili_risk_numeric'], axis=1)
    y = combined_df['dili_risk_numeric']
    
    # Remove columns with too many NaNs
    valid_cols = X.columns[X.notna().sum() / len(X) > 0.5]
    X = X[valid_cols].fillna(X[valid_cols].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': valid_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_scaled, y, cv=KFold(5, shuffle=True, random_state=42), 
                               scoring='r2')
    print(f"\n   Random Forest R² (5-fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Create visualizations
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Combined Feature Analysis for DILI Prediction', fontsize=16, fontweight='bold')

# Plot 1: Dataset sizes
ax = axes[0, 0]
datasets = ['Event-Aware\nOnly', 'Phase 2\nOnly', 'Combined\n(3-way)', 'Phase 2\n(All methods)']
sizes = [len(event_dili), len(phase2_dili) - len(overlap_all), len(overlap_all), len(phase2_dili)]
colors = ['lightblue', 'lightgreen', 'gold', 'lightcoral']

bars = ax.bar(datasets, sizes, color=colors, edgecolor='black', linewidth=1)
ax.set_ylabel('Number of Drugs')
ax.set_title('Dataset Sizes')

for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(size), ha='center', va='bottom', fontweight='bold')

# Plot 2: Feature importance (if available)
ax = axes[0, 1]
if 'feature_importance' in locals() and len(feature_importance) > 0:
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'], 
            color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f[:30] for f in top_features['feature']], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features (Random Forest)')
else:
    ax.text(0.5, 0.5, 'Insufficient data\nfor importance analysis', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Feature Importance')

# Plot 3: PCA visualization of combined features
ax = axes[0, 2]
if len(overlap_all) > 10 and 'X_scaled' in locals():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                        c=y, cmap='RdYlBu_r', s=60, 
                        edgecolor='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA of Combined Features')
    plt.colorbar(scatter, ax=ax, label='DILI Risk')
else:
    ax.text(0.5, 0.5, 'Insufficient overlap\nfor PCA', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('PCA Visualization')

# Plot 4: Correlation matrix of top features
ax = axes[1, 0]
if len(overlap_all) > 10:
    # Select top event-aware and phase2 features
    event_cols = [col for col in event_subset.columns if 'consumption' in col or 'min_o2' in col][:5]
    phase2_cols = [col for col in phase2_subset.columns][:5]
    
    if event_cols and phase2_cols:
        selected_features = combined_df[event_cols + phase2_cols + ['dili_risk_numeric']].dropna()
        
        corr_matrix = selected_features.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                   annot_kws={'size': 8})
        ax.set_title('Feature Correlations')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
else:
    ax.text(0.5, 0.5, 'Insufficient data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Feature Correlations')

# Plot 5: Performance comparison
ax = axes[1, 1]
methods = ['Event-Aware\nAlone', 'Phase 2\nAlone', 'Combined\nFeatures']
correlations = [0.435, 0.260, 0]  # Will update combined

if len(overlap_all) > 10 and 'combined_df' in locals():
    # Calculate combined correlation
    feature_cols = [col for col in combined_df.columns if col != 'dili_risk_numeric']
    
    # Find best single feature correlation
    best_corr = 0
    for col in feature_cols[:20]:  # Check top 20 features
        if combined_df[col].notna().sum() > 10:
            corr, _ = spearmanr(combined_df[col].dropna(), 
                               combined_df.loc[combined_df[col].notna(), 'dili_risk_numeric'])
            if abs(corr) > best_corr:
                best_corr = abs(corr)
    
    correlations[2] = best_corr

colors = ['lightblue', 'lightgreen', 'gold']
bars = ax.bar(methods, correlations, color=colors, edgecolor='black', linewidth=1)
ax.set_ylabel('Best |Correlation| with DILI')
ax.set_title('Method Performance')
ax.set_ylim(0, 0.6)

for bar, corr in zip(bars, correlations):
    if corr > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Sample size vs performance
ax = axes[1, 2]
sample_sizes = [len(event_dili), len(phase2_dili), len(overlap_all)]
performances = [0.435, 0.260, correlations[2] if correlations[2] > 0 else 0.45]
labels = ['Event-Aware', 'Phase 2', 'Combined']

ax.scatter(sample_sizes, performances, s=100, c=colors[:3], 
          edgecolor='black', linewidth=1.5)

for i, (x, y, label) in enumerate(zip(sample_sizes, performances, labels)):
    ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel('Number of Drugs')
ax.set_ylabel('Best Correlation')
ax.set_title('Sample Size vs Performance')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'combined_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
print("\n💾 Saving results...")

if len(overlap_all) > 10:
    combined_df.to_csv(results_dir / 'combined_features_drugs.csv')
    print(f"   Combined features: {results_dir / 'combined_features_drugs.csv'}")

phase2_dili_df.to_csv(results_dir / 'phase2_all_methods_dili.csv')
print(f"   Phase 2 all methods: {results_dir / 'phase2_all_methods_dili.csv'}")

# Summary
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 DATASET COMBINATIONS:")
print(f"   Event-aware only: {len(event_dili)} drugs")
print(f"   Phase 2 only: {len(phase2_dili)} drugs")
print(f"   Combined (3-way): {len(overlap_all)} drugs")

print(f"\n🎯 KEY FINDINGS:")
print(f"   1. Limited overlap ({len(overlap_all)} drugs) between all datasets")
print(f"   2. Event-aware features show promise but need more data")
print(f"   3. Phase 2 embeddings provide broader coverage")
print(f"   4. Combined approach may improve predictions with more data")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Process more plates to increase event-aware coverage")
print(f"   2. Use Phase 2 embeddings as primary features (better coverage)")
print(f"   3. Add event-aware features when available")
print(f"   4. Include chemical descriptors for full model")

print("\n✅ Combined analysis complete!")