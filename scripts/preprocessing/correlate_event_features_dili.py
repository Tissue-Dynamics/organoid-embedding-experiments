#!/usr/bin/env python3
"""
Correlate Event-Aware Features with DILI Risk
Compare with Phase 2 embeddings to see improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_aware_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EVENT-AWARE FEATURES vs DILI CORRELATION ANALYSIS")
print("=" * 80)

# Load event-aware features
drug_features = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")
print(f"\n📊 Loaded event-aware features for {len(drug_features)} drugs")

# Load DILI data
wells_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")

# Create drug-level DILI mapping
drug_dili_map = wells_df[wells_df['dili'].notna()].groupby('drug').agg({
    'dili': 'first',
    'binary_dili': 'first',
    'dili_risk_category': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
}).reset_index()

# Create numeric DILI risk mapping
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

# Fill NaN values using binary DILI
if drug_dili_map['dili_risk_numeric'].isna().any():
    drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'dili_risk_numeric'] = \
        drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'binary_dili'].map({True: 2.0, False: 0.0})

print(f"\n💊 Drug DILI mapping: {len(drug_dili_map)} drugs")

# Merge features with DILI
merged_df = drug_features.merge(drug_dili_map, on='drug', how='inner')
print(f"\n🔗 Merged dataset: {len(merged_df)} drugs with both features and DILI")

# Select key features for correlation
feature_cols = [col for col in drug_features.columns if col != 'drug' and '_mean' in col and '_count' not in col]
print(f"\n📊 Analyzing {len(feature_cols)} features")

# Calculate correlations
correlations = []
for feature in feature_cols:
    if feature in merged_df.columns:
        valid_data = merged_df[[feature, 'dili_risk_numeric']].dropna()
        
        if len(valid_data) > 10:
            # Spearman correlation
            corr_s, p_s = spearmanr(valid_data[feature], valid_data['dili_risk_numeric'])
            # Pearson correlation  
            corr_p, p_p = pearsonr(valid_data[feature], valid_data['dili_risk_numeric'])
            
            correlations.append({
                'feature': feature,
                'spearman_r': corr_s,
                'spearman_p': p_s,
                'pearson_r': corr_p,
                'pearson_p': p_p,
                'n_drugs': len(valid_data)
            })

corr_df = pd.DataFrame(correlations)
corr_df['abs_spearman'] = corr_df['spearman_r'].abs()
corr_df = corr_df.sort_values('abs_spearman', ascending=False)

# Find significant correlations
significant = corr_df[corr_df['spearman_p'] < 0.05]

print(f"\n🔍 CORRELATION RESULTS:")
print(f"   Significant correlations (p<0.05): {len(significant)}")
print(f"\n   Top 10 correlations:")
for _, row in corr_df.head(10).iterrows():
    print(f"   {row['feature']}: r={row['spearman_r']:.3f}, p={row['spearman_p']:.3e}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Event-Aware Features vs DILI Risk Analysis', fontsize=16, fontweight='bold')

# Plot 1: Top correlation scatter
ax = axes[0, 0]
if len(corr_df) > 0:
    best_feature = corr_df.iloc[0]['feature']
    valid_data = merged_df[[best_feature, 'dili_risk_numeric']].dropna()
    
    scatter = ax.scatter(valid_data[best_feature], valid_data['dili_risk_numeric'], 
                        alpha=0.6, s=60, edgecolor='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(valid_data[best_feature], valid_data['dili_risk_numeric'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(valid_data[best_feature].min(), valid_data[best_feature].max(), 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel(best_feature.replace('_', ' ').title())
    ax.set_ylabel('DILI Risk Score')
    ax.set_title(f'Best Correlation: r={corr_df.iloc[0]["spearman_r"]:.3f}')
    ax.grid(True, alpha=0.3)

# Plot 2: Correlation strengths
ax = axes[0, 1]
top_features = corr_df.head(15)
colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'gray' 
          for p in top_features['spearman_p']]

ax.barh(range(len(top_features)), top_features['abs_spearman'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f.replace('_mean', '').replace('_', ' ')[:30] for f in top_features['feature']], fontsize=8)
ax.set_xlabel('|Correlation| with DILI Risk')
ax.set_title('Top Feature Correlations')
ax.axvline(0.3, color='green', linestyle='--', alpha=0.5, label='|r|=0.3')
ax.legend()

# Plot 3: Consumption rate vs DILI
ax = axes[0, 2]
if 'consumption_rate_mean_mean' in merged_df.columns:
    valid_data = merged_df[['consumption_rate_mean_mean', 'dili_risk_numeric']].dropna()
    
    # Group by DILI risk
    risk_groups = valid_data.groupby('dili_risk_numeric')['consumption_rate_mean_mean'].apply(list)
    
    ax.boxplot([values for values in risk_groups.values], 
               positions=risk_groups.index,
               widths=0.6)
    ax.set_xlabel('DILI Risk Score')
    ax.set_ylabel('Oxygen Consumption Rate (%O₂/hour)')
    ax.set_title('Consumption Rate by DILI Risk')
    ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Temporal changes vs DILI
ax = axes[1, 0]
if 'consumption_change_mean' in merged_df.columns:
    valid_data = merged_df[['consumption_change_mean', 'dili_risk_numeric']].dropna()
    
    colors = ['green' if risk <= 1 else 'orange' if risk <= 2 else 'red' 
              for risk in valid_data['dili_risk_numeric']]
    
    ax.scatter(valid_data['consumption_change_mean'], valid_data['dili_risk_numeric'],
               c=colors, alpha=0.6, s=60, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Consumption Rate Change (Late - Early)')
    ax.set_ylabel('DILI Risk Score')
    ax.set_title('Temporal Progression vs DILI Risk')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

# Plot 5: Feature importance heatmap
ax = axes[1, 1]
# Select top features by correlation
top_n = 20
top_features_df = corr_df.head(top_n).copy()
top_features_df['signed_corr'] = top_features_df['spearman_r']

# Create heatmap data
heatmap_data = top_features_df[['feature', 'signed_corr']].set_index('feature')
heatmap_data['p_value'] = top_features_df.set_index('feature')['spearman_p']

# Plot heatmap
im = ax.imshow(heatmap_data[['signed_corr']].T, aspect='auto', cmap='RdBu_r', 
               vmin=-0.5, vmax=0.5)
ax.set_yticks([0])
ax.set_yticklabels(['Correlation'])
ax.set_xticks(range(len(heatmap_data)))
ax.set_xticklabels([f.replace('_mean', '').replace('_', ' ')[:20] for f in heatmap_data.index], 
                   rotation=45, ha='right', fontsize=8)
ax.set_title('Feature Correlation Heatmap')

# Add significance markers
for i, (idx, row) in enumerate(heatmap_data.iterrows()):
    if row['p_value'] < 0.001:
        ax.text(i, 0, '***', ha='center', va='center', fontsize=8)
    elif row['p_value'] < 0.01:
        ax.text(i, 0, '**', ha='center', va='center', fontsize=8)
    elif row['p_value'] < 0.05:
        ax.text(i, 0, '*', ha='center', va='center', fontsize=8)

# Plot 6: Comparison with Phase 2
ax = axes[1, 2]
# Load Phase 2 results for comparison
comparison_data = {
    'Method': ['Phase 2\nFourier', 'Phase 2\nTSFresh', 'Phase 2\nCatch22', 
               'Event-Aware\nFeatures'],
    'Best |r|': [0.260, 0.243, 0.237, corr_df.iloc[0]['abs_spearman'] if len(corr_df) > 0 else 0],
    'N_Significant': [3, 2, 2, len(significant)]
}

comp_df = pd.DataFrame(comparison_data)
colors = ['lightblue', 'lightblue', 'lightblue', 'lightgreen']

bars = ax.bar(comp_df['Method'], comp_df['Best |r|'], color=colors, edgecolor='black', linewidth=1)
ax.set_ylabel('Best |Correlation|')
ax.set_title('Method Comparison')
ax.set_ylim(0, max(comp_df['Best |r|']) * 1.2)

# Add value labels
for bar, val, n_sig in zip(bars, comp_df['Best |r|'], comp_df['N_Significant']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}\n({n_sig} sig)', 
            ha='center', va='bottom', fontsize=10)

ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='|r|=0.3 threshold')
ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / 'event_aware_dili_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save correlation results
corr_df.to_csv(results_dir / 'event_aware_dili_correlations.csv', index=False)

# Summary statistics
print(f"\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 DATASET:")
print(f"   Event-aware features: {len(drug_features)} drugs")
print(f"   DILI data: {len(drug_dili_map)} drugs")
print(f"   Overlap: {len(merged_df)} drugs")

print(f"\n🏆 BEST CORRELATIONS:")
if len(significant) > 0:
    for _, row in significant.head(5).iterrows():
        print(f"   {row['feature']}: r={row['spearman_r']:.3f}, p={row['spearman_p']:.3e}")
else:
    print("   No significant correlations found")

print(f"\n📈 IMPROVEMENT OVER PHASE 2:")
phase2_best = 0.260  # Best Phase 2 correlation
event_best = corr_df.iloc[0]['abs_spearman'] if len(corr_df) > 0 else 0
improvement = (event_best - phase2_best) / phase2_best * 100

print(f"   Phase 2 best: r={phase2_best:.3f}")
print(f"   Event-aware best: r={event_best:.3f}")
print(f"   Improvement: {improvement:+.1f}%")

# Feature interpretation
print(f"\n💡 KEY INSIGHTS:")
if len(significant) > 0:
    # Consumption rate
    consumption_corrs = significant[significant['feature'].str.contains('consumption_rate')]
    if len(consumption_corrs) > 0:
        print(f"   • Oxygen consumption rate correlates with DILI (r={consumption_corrs.iloc[0]['spearman_r']:.3f})")
    
    # Temporal changes
    temporal_corrs = significant[significant['feature'].str.contains('consumption_change')]
    if len(temporal_corrs) > 0:
        print(f"   • Temporal changes in consumption predict DILI (r={temporal_corrs.iloc[0]['spearman_r']:.3f})")
    
    # Variability
    cv_corrs = significant[significant['feature'].str.contains('cv_')]
    if len(cv_corrs) > 0:
        print(f"   • Oxygen variability associated with DILI (r={cv_corrs.iloc[0]['spearman_r']:.3f})")

print(f"\n✅ Event-aware DILI correlation analysis complete!")
print(f"   Results saved to: {fig_dir}")
print(f"   Correlation table: {results_dir / 'event_aware_dili_correlations.csv'}")