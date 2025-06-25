#!/usr/bin/env python3
"""
Event-Normalized Features DILI Analysis

PURPOSE:
    Analyzes how event-normalized time features correlate with DILI risk.
    These features capture drug response dynamics relative to media change events
    rather than absolute time, potentially revealing toxicity patterns.

APPROACH:
    - Load event-normalized features (recovery times, event consistency, etc.)
    - Correlate with DILI classifications
    - Compare predictive power to previous approaches
    - Identify which event-related patterns best predict toxicity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import sys

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_normalized_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EVENT-NORMALIZED FEATURES DILI ANALYSIS")
print("=" * 80)

# Load DILI data
def load_dili_data():
    """Load DILI classification data"""
    
    print("\n📊 Loading DILI classification data...")
    
    dili_data = {
        # High DILI concern drugs (severity 4-5)
        'Amiodarone': {'DILI_severity': 4, 'DILI_binary': 1},
        'Busulfan': {'DILI_severity': 4, 'DILI_binary': 1},
        'Imatinib': {'DILI_severity': 4, 'DILI_binary': 1},
        'Lapatinib': {'DILI_severity': 4, 'DILI_binary': 1},
        'Pazopanib': {'DILI_severity': 4, 'DILI_binary': 1},
        'Regorafenib': {'DILI_severity': 4, 'DILI_binary': 1},
        'Sorafenib': {'DILI_severity': 4, 'DILI_binary': 1},
        'Sunitinib': {'DILI_severity': 4, 'DILI_binary': 1},
        'Trametinib': {'DILI_severity': 4, 'DILI_binary': 1},
        
        # Moderate DILI concern drugs (severity 3)
        'Anastrozole': {'DILI_severity': 3, 'DILI_binary': 1},
        'Axitinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Cabozantinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Dabrafenib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Erlotinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Gefitinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Lenvatinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Nilotinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Osimertinib': {'DILI_severity': 3, 'DILI_binary': 1},
        'Vemurafenib': {'DILI_severity': 3, 'DILI_binary': 1},
        
        # Lower DILI concern drugs (severity 2)
        'Alectinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Binimetinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Bortezomib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Ceritinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Crizotinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Dasatinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Everolimus': {'DILI_severity': 2, 'DILI_binary': 1},
        'Ibrutinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Ponatinib': {'DILI_severity': 2, 'DILI_binary': 1},
        'Ruxolitinib': {'DILI_severity': 2, 'DILI_binary': 1},
        
        # Minimal/No DILI concern drugs (severity 1)
        'Alpelisib': {'DILI_severity': 1, 'DILI_binary': 0},
        'Ambrisentan': {'DILI_severity': 1, 'DILI_binary': 0},
        'Buspirone': {'DILI_severity': 1, 'DILI_binary': 0},
        'Dexamethasone': {'DILI_severity': 1, 'DILI_binary': 0},
        'Fulvestrant': {'DILI_severity': 1, 'DILI_binary': 0},
        'Letrozole': {'DILI_severity': 1, 'DILI_binary': 0},
        'Palbociclib': {'DILI_severity': 1, 'DILI_binary': 0},
        'Ribociclib': {'DILI_severity': 1, 'DILI_binary': 0},
        'Trastuzumab': {'DILI_severity': 1, 'DILI_binary': 0},
        'Zoledronic acid': {'DILI_severity': 1, 'DILI_binary': 0}
    }
    
    dili_df = pd.DataFrame.from_dict(dili_data, orient='index').reset_index()
    dili_df.columns = ['drug', 'DILI_severity', 'DILI_binary']
    
    print(f"   Loaded DILI data for {len(dili_df)} drugs")
    
    return dili_df

# Load event-normalized features
print("\n📊 Loading event-normalized features...")

try:
    # Try final version first
    if (results_dir / 'event_normalized_features_drugs_final.parquet').exists():
        drug_features_df = pd.read_parquet(results_dir / 'event_normalized_features_drugs_final.parquet')
        print(f"   ✓ Loaded final features for {len(drug_features_df)} drugs")
    else:
        drug_features_df = pd.read_parquet(results_dir / 'event_normalized_features_drugs_optimized.parquet')
        print(f"   ✓ Loaded optimized features for {len(drug_features_df)} drugs")
    
    # Load summary if available
    try:
        with open(results_dir / 'event_normalized_summary_optimized.json', 'r') as f:
            feature_summary = json.load(f)
    except:
        feature_summary = {}
    
except FileNotFoundError:
    print("   ✗ Event-normalized features not found! Run event_normalized_time_features.py first")
    exit(1)

# Load DILI data
dili_df = load_dili_data()

# Merge with DILI data
print("\n🔍 Analyzing DILI correlations...")
merged_df = drug_features_df.merge(dili_df, on='drug', how='inner')

if len(merged_df) == 0:
    print("   No drugs overlap between features and DILI data!")
    exit(1)

print(f"   Analyzing {len(merged_df)} drugs with DILI data")

# Get feature columns
exclude_cols = ['drug', 'concentration', 'n_wells', 'n_concentrations', 'DILI_severity', 'DILI_binary']
feature_cols = [col for col in merged_df.columns if col not in exclude_cols]

# Categorize features
feature_categories = {
    'recovery_time': [col for col in feature_cols if 'time_to_' in col and 'recovery' in col],
    'recovery_rate': [col for col in feature_cols if 'recovery_rate' in col],
    'suppression': [col for col in feature_cols if 'suppression' in col],
    'immediate_post': [col for col in feature_cols if 'immediate_post' in col],
    'early_post': [col for col in feature_cols if 'early_post' in col],
    'late_post': [col for col in feature_cols if 'late_post' in col],
    'pre_event': [col for col in feature_cols if 'pre_event' in col],
    'consistency': [col for col in feature_cols if 'consistency' in col],
    'catch22': [col for col in feature_cols if 'catch22' in col]
}

# Calculate correlations by category
print("\n📊 Feature category correlations with DILI:")
category_results = []

for category, features in feature_categories.items():
    if not features:
        continue
    
    correlations = []
    for feat in features:
        if feat in merged_df.columns:
            valid_data = merged_df[[feat, 'DILI_severity']].dropna()
            if len(valid_data) >= 5:
                r, p = pearsonr(valid_data[feat], valid_data['DILI_severity'])
                correlations.append({'feature': feat, 'r': r, 'p': p, 'abs_r': abs(r)})
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        category_results.append({
            'category': category,
            'n_features': len(corr_df),
            'mean_abs_r': corr_df['abs_r'].mean(),
            'max_abs_r': corr_df['abs_r'].max(),
            'top_feature': corr_df.loc[corr_df['abs_r'].idxmax(), 'feature'],
            'top_r': corr_df.loc[corr_df['abs_r'].idxmax(), 'r']
        })

category_df = pd.DataFrame(category_results).sort_values('mean_abs_r', ascending=False)

print("\nCategory performance:")
for _, row in category_df.iterrows():
    print(f"   {row['category']}: mean |r|={row['mean_abs_r']:.3f}, max |r|={row['max_abs_r']:.3f}")

# Find top individual features
all_correlations = []
for feat in feature_cols:
    if feat in merged_df.columns:
        valid_data = merged_df[[feat, 'DILI_severity']].dropna()
        if len(valid_data) >= 5:
            r, p = pearsonr(valid_data[feat], valid_data['DILI_severity'])
            all_correlations.append({
                'feature': feat,
                'r': r,
                'p': p,
                'abs_r': abs(r),
                'n': len(valid_data)
            })

correlations_df = pd.DataFrame(all_correlations).sort_values('abs_r', ascending=False)

print(f"\n🎯 Top 10 event-normalized features for DILI prediction:")
for _, row in correlations_df.head(10).iterrows():
    print(f"   {row['feature']}: r={row['r']:.3f} (p={row['p']:.3f})")

# Create visualizations
print("\n📊 Creating visualizations...")

# 1. Top features bar plot
plt.figure(figsize=(12, 8))
top_features = correlations_df.head(20)
colors = ['darkred' if r > 0 else 'darkblue' for r in top_features['r']]
bars = plt.barh(range(len(top_features)), top_features['r'], color=colors, alpha=0.7)

plt.yticks(range(len(top_features)), 
           [f.replace('_', ' ')[:50] + '...' if len(f) > 50 else f.replace('_', ' ') 
            for f in top_features['feature']])
plt.xlabel('Pearson Correlation with DILI Severity')
plt.title('Top 20 Event-Normalized Features: DILI Correlation', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(fig_dir / 'top_event_normalized_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Category comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(category_df)), category_df['mean_abs_r'], color='skyblue', alpha=0.7)
plt.xticks(range(len(category_df)), category_df['category'], rotation=45, ha='right')
plt.ylabel('Mean |Pearson r| with DILI')
plt.title('Event-Normalized Feature Categories: DILI Prediction Performance', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add max correlations
for i, (idx, row) in enumerate(category_df.iterrows()):
    plt.text(i, row['mean_abs_r'] + 0.005, f"max={row['max_abs_r']:.2f}", 
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / 'event_normalized_category_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
print("\n💾 Saving analysis results...")

correlations_df.to_parquet(results_dir / 'event_normalized_dili_correlations.parquet', index=False)
category_df.to_parquet(results_dir / 'event_normalized_dili_categories.parquet', index=False)

# Summary
summary = {
    'n_drugs_analyzed': merged_df['drug'].nunique(),
    'n_drug_conc_combinations': len(merged_df),
    'top_correlation': float(correlations_df.iloc[0]['r']),
    'top_feature': correlations_df.iloc[0]['feature'],
    'top_p_value': float(correlations_df.iloc[0]['p']),
    'category_performance': category_df.to_dict('records'),
    'comparison': {
        'phase2_embeddings': 0.260,
        'event_aware_features': 0.435,
        'post_treatment_features': 0.522,
        'event_normalized_best': float(correlations_df.iloc[0]['abs_r'])
    }
}

with open(results_dir / 'event_normalized_dili_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Event-normalized DILI analysis complete!")
print(f"\n📊 KEY RESULTS:")
print(f"   Top correlation: {correlations_df.iloc[0]['feature']}")
print(f"                   r = {correlations_df.iloc[0]['r']:.3f}")
print(f"   Best category: {category_df.iloc[0]['category']}")
print(f"                 mean |r| = {category_df.iloc[0]['mean_abs_r']:.3f}")

# Compare to previous approaches
print(f"\n📈 Performance comparison:")
print(f"   Phase 2 embeddings: r = 0.260")
print(f"   Event-aware features: r = 0.435")
print(f"   Post-treatment features: r = 0.522")
print(f"   Event-normalized (current): r = {correlations_df.iloc[0]['abs_r']:.3f}")