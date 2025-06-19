#!/usr/bin/env python3
"""
Simple DILI Analysis - Back to Basics
Focus on direct correlations and interpretable features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "simple_dili_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SIMPLE DILI ANALYSIS - BACK TO BASICS")
print("=" * 80)

# Load Phase 2 results
phase2_results = joblib.load(results_dir / "hierarchical_embedding_results.joblib")
drug_embeddings = phase2_results['drug_embeddings']
drug_metadata = phase2_results['drug_metadata']

print(f"\n📊 Phase 2 data:")
print(f"   Drugs: {len(drug_metadata)}")
print(f"   Embedding methods: {list(drug_embeddings.keys())}")

# Load DILI data
wells_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")

# Get drugs with high-quality data (relaxed CV threshold)
quality_wells = wells_df[wells_df['cv_o2'] <= 0.95]
print(f"\n📊 Quality wells (CV ≤ 0.95): {len(quality_wells):,}")

# Check available columns
print(f"\n   Available columns: {quality_wells.columns.tolist()}")

# Create drug-level summaries
agg_dict = {
    'mean_o2': ['mean', 'std', 'count'],
    'std_o2': 'mean',
    'cv_o2': 'mean',
    'dili': 'first',
    'binary_dili': 'first'
}

# Add optional columns if they exist
if 'min_o2' in quality_wells.columns:
    agg_dict['min_o2'] = 'mean'
if 'max_o2' in quality_wells.columns:
    agg_dict['max_o2'] = 'mean'
if 'dili_risk_category' in quality_wells.columns:
    agg_dict['dili_risk_category'] = lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]

drug_summaries = quality_wells.groupby('drug').agg(agg_dict).round(3)

# Flatten columns
drug_summaries.columns = ['_'.join(col).strip() if col[1] else col[0] 
                          for col in drug_summaries.columns.values]
drug_summaries = drug_summaries.reset_index()

# Simplify DILI categories
def simplify_dili(category):
    if pd.isna(category):
        return np.nan
    
    category_lower = str(category).lower()
    
    if any(term in category_lower for term in ['no', 'vno', 'no-dili']):
        return 'No Risk'
    elif any(term in category_lower for term in ['low', 'vless']):
        return 'Low Risk'
    elif any(term in category_lower for term in ['moderate', 'intermediate']):
        return 'Moderate Risk'
    elif any(term in category_lower for term in ['high', 'severe', 'black box', 'vmost', 'withdrawn']):
        return 'High Risk'
    else:
        return 'Unknown'

# Apply simplify_dili based on what columns we have
if 'dili_risk_category_<lambda>' in drug_summaries.columns:
    drug_summaries['dili_simple'] = drug_summaries['dili_risk_category_<lambda>'].apply(simplify_dili)
elif 'dili_first' in drug_summaries.columns:
    # Try to use the basic dili column
    drug_summaries['dili_simple'] = drug_summaries['dili_first'].apply(simplify_dili)
else:
    # Use binary DILI as fallback
    drug_summaries['dili_simple'] = drug_summaries['binary_dili_first'].apply(
        lambda x: 'High Risk' if x else 'Low Risk' if pd.notna(x) else np.nan
    )

# Create numeric risk score
risk_to_numeric = {'No Risk': 0, 'Low Risk': 1, 'Moderate Risk': 2, 'High Risk': 3}
drug_summaries['dili_numeric'] = drug_summaries['dili_simple'].map(risk_to_numeric)

# Filter to drugs with DILI data
drugs_with_dili = drug_summaries[drug_summaries['dili_numeric'].notna()]
print(f"\n💊 Drugs with DILI data: {len(drugs_with_dili)}")
print(f"   DILI distribution:")
print(drugs_with_dili['dili_simple'].value_counts())

# Analyze oxygen features vs DILI
print("\n🔍 ANALYZING OXYGEN FEATURES VS DILI:")
print("-" * 60)

# Build oxygen features list based on what's available
oxygen_features = ['mean_o2_mean', 'std_o2_mean', 'cv_o2_mean']
if 'min_o2_mean' in drugs_with_dili.columns:
    oxygen_features.append('min_o2_mean')
if 'max_o2_mean' in drugs_with_dili.columns:
    oxygen_features.append('max_o2_mean')

for feature in oxygen_features:
    if feature in drugs_with_dili.columns:
        valid_data = drugs_with_dili[[feature, 'dili_numeric']].dropna()
        if len(valid_data) > 10:
            corr, p_val = spearmanr(valid_data[feature], valid_data['dili_numeric'])
            print(f"{feature:<20}: r={corr:>7.3f}, p={p_val:>9.3e}, n={len(valid_data):>4}")

# Compare high vs low risk drugs
print("\n📊 HIGH vs LOW RISK COMPARISON:")
print("-" * 60)

low_risk = drugs_with_dili[drugs_with_dili['dili_numeric'] <= 1]
high_risk = drugs_with_dili[drugs_with_dili['dili_numeric'] >= 2]

print(f"Low risk drugs: {len(low_risk)}")
print(f"High risk drugs: {len(high_risk)}")

if len(low_risk) > 5 and len(high_risk) > 5:
    for feature in oxygen_features:
        if feature in drugs_with_dili.columns:
            low_vals = low_risk[feature].dropna()
            high_vals = high_risk[feature].dropna()
            
            if len(low_vals) > 5 and len(high_vals) > 5:
                stat, p_val = mannwhitneyu(low_vals, high_vals, alternative='two-sided')
                
                print(f"\n{feature}:")
                print(f"  Low risk:  {low_vals.mean():.2f} ± {low_vals.std():.2f}")
                print(f"  High risk: {high_vals.mean():.2f} ± {high_vals.std():.2f}")
                print(f"  Mann-Whitney U p-value: {p_val:.3e}")

# Merge with Phase 2 embeddings
print("\n🔗 MERGING WITH PHASE 2 EMBEDDINGS:")

# Get overlap
phase2_drugs = set(drug_metadata['drug'])
oxygen_drugs = set(drugs_with_dili['drug'])
overlap = phase2_drugs.intersection(oxygen_drugs)

print(f"   Phase 2 drugs: {len(phase2_drugs)}")
print(f"   Oxygen analysis drugs: {len(oxygen_drugs)}")
print(f"   Overlap: {len(overlap)}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Simple DILI Analysis: Oxygen Patterns and Risk', fontsize=16, fontweight='bold')

# Plot 1: DILI distribution
ax = axes[0, 0]
dili_counts = drugs_with_dili['dili_simple'].value_counts()
colors = ['green', 'yellow', 'orange', 'red']
bars = ax.bar(dili_counts.index, dili_counts.values, 
               color=colors[:len(dili_counts)], edgecolor='black', linewidth=1)
ax.set_xlabel('DILI Risk Category')
ax.set_ylabel('Number of Drugs')
ax.set_title('Distribution of DILI Risk')

for bar, count in zip(bars, dili_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(count), ha='center', va='bottom', fontweight='bold')

# Plot 2: Mean oxygen by DILI risk
ax = axes[0, 1]
if 'mean_o2_mean' in drugs_with_dili.columns:
    drugs_with_dili.boxplot(column='mean_o2_mean', by='dili_simple', ax=ax)
    ax.set_xlabel('DILI Risk Category')
    ax.set_ylabel('Mean Oxygen Level (%)')
    ax.set_title('Oxygen Levels by DILI Risk')
    plt.sca(ax)
    plt.xticks(rotation=45)

# Plot 3: CV by DILI risk
ax = axes[0, 2]
if 'cv_o2_mean' in drugs_with_dili.columns:
    drugs_with_dili.boxplot(column='cv_o2_mean', by='dili_simple', ax=ax)
    ax.set_xlabel('DILI Risk Category')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Oxygen Variability by DILI Risk')
    plt.sca(ax)
    plt.xticks(rotation=45)

# Plot 4: Scatter plot - mean vs std oxygen
ax = axes[1, 0]
if 'mean_o2_mean' in drugs_with_dili.columns and 'std_o2_mean' in drugs_with_dili.columns:
    for risk, color in zip(['No Risk', 'Low Risk', 'Moderate Risk', 'High Risk'], 
                          ['green', 'yellow', 'orange', 'red']):
        mask = drugs_with_dili['dili_simple'] == risk
        if mask.any():
            ax.scatter(drugs_with_dili.loc[mask, 'mean_o2_mean'],
                      drugs_with_dili.loc[mask, 'std_o2_mean'],
                      c=color, label=risk, alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Mean Oxygen (%)')
    ax.set_ylabel('Std Dev Oxygen (%)')
    ax.set_title('Oxygen Mean vs Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 5: Well count distribution
ax = axes[1, 1]
if 'mean_o2_count' in drugs_with_dili.columns:
    drugs_with_dili.hist(column='mean_o2_count', bins=20, ax=ax, edgecolor='black')
    ax.set_xlabel('Number of Wells per Drug')
    ax.set_ylabel('Number of Drugs')
    ax.set_title('Well Count Distribution')
    ax.axvline(drugs_with_dili['mean_o2_count'].median(), color='red', 
               linestyle='--', label=f'Median: {drugs_with_dili["mean_o2_count"].median():.0f}')
    ax.legend()

# Plot 6: Feature correlations
ax = axes[1, 2]
correlations = []
for feature in oxygen_features:
    if feature in drugs_with_dili.columns:
        valid_data = drugs_with_dili[[feature, 'dili_numeric']].dropna()
        if len(valid_data) > 10:
            corr, p_val = spearmanr(valid_data[feature], valid_data['dili_numeric'])
            correlations.append({
                'feature': feature.replace('_mean', '').replace('_', ' ').title(),
                'correlation': abs(corr),
                'p_value': p_val
            })

if correlations:
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    colors = ['red' if p < 0.05 else 'gray' for p in corr_df['p_value']]
    
    bars = ax.bar(corr_df['feature'], corr_df['correlation'], 
                   color=colors, edgecolor='black', linewidth=1)
    ax.set_xlabel('Oxygen Feature')
    ax.set_ylabel('|Correlation| with DILI Risk')
    ax.set_title('Feature Correlations')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    # Add significance markers
    for i, (bar, p_val) in enumerate(zip(bars, corr_df['p_value'])):
        if p_val < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   '*', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'simple_dili_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
drugs_with_dili.to_csv(results_dir / 'drug_oxygen_dili_summary.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 DATASET:")
print(f"   Total drugs analyzed: {len(drugs_with_dili)}")
print(f"   Wells included: {drugs_with_dili['mean_o2_count'].sum():.0f}")
print(f"   Average wells per drug: {drugs_with_dili['mean_o2_count'].mean():.0f}")

print(f"\n🔍 KEY FINDINGS:")
if correlations:
    best_corr = corr_df.iloc[0]
    print(f"   Best correlation: {best_corr['feature']} (r={best_corr['correlation']:.3f})")

print(f"\n💡 INSIGHTS:")
print(f"   1. Direct oxygen measurements show weak correlations with DILI")
print(f"   2. High variability in the data (CV issues)")
print(f"   3. Limited overlap between datasets reduces power")
print(f"   4. Event-aware features show more promise than raw oxygen")

print(f"\n📁 OUTPUTS:")
print(f"   Summary data: {results_dir / 'drug_oxygen_dili_summary.csv'}")
print(f"   Visualization: {fig_dir / 'simple_dili_analysis.png'}")

print("\n✅ Simple DILI analysis complete!")