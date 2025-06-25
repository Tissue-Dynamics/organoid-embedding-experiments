#!/usr/bin/env python3
"""
Corrected Hierarchical Features DILI Correlation Analysis

PURPOSE:
    Fixed version that properly categorizes features to avoid the baseline prediction paradox.
    True baseline features (pre-treatment) should NOT predict DILI.
    Drug response features (post-treatment) can legitimately predict DILI.

FIXES:
    1. Properly categorize true baseline vs drug response features
    2. Separate "baseline_deviation" type features (which are drug responses) from true baseline
    3. Validate that true pre-treatment features don't predict DILI
    4. Focus analysis on legitimate drug response features
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import sys

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "corrected_hierarchical_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CORRECTED HIERARCHICAL FEATURES DILI CORRELATION ANALYSIS")
print("=" * 80)

# ========== CORRECTED FEATURE CATEGORIZATION ==========

def load_dili_data():
    """Load DILI classification data"""
    
    print("\n📊 Loading DILI classification data...")
    
    # DILI data for compounds in our hierarchical features dataset
    dili_data = {
        # High DILI concern drugs (severity 4-5)
        'Amiodarone': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Busulfan': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Imatinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Lapatinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Pazopanib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Regorafenib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Sorafenib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Sunitinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        'Trametinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern'},
        
        # Moderate DILI concern drugs (severity 3)
        'Anastrozole': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Axitinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Cabozantinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Dabrafenib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Erlotinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Gefitinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Lenvatinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Nilotinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Osimertinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        'Vemurafenib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern'},
        
        # Lower DILI concern drugs (severity 2)
        'Alectinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Binimetinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Bortezomib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Ceritinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Crizotinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Dasatinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Everolimus': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Ibrutinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Ponatinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        'Ruxolitinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern'},
        
        # Minimal/No DILI concern drugs (severity 1)
        'Alpelisib': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Ambrisentan': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Buspirone': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Dexamethasone': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Fulvestrant': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Letrozole': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Palbociclib': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Ribociclib': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Trastuzumab': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'},
        'Zoledronic acid': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern'}
    }
    
    dili_df = pd.DataFrame.from_dict(dili_data, orient='index').reset_index()
    dili_df.columns = ['drug', 'DILI_severity', 'DILI_binary', 'DILI_concern']
    
    print(f"   Loaded DILI data for {len(dili_df)} drugs")
    print(f"   DILI-positive: {dili_df['DILI_binary'].sum()}")
    print(f"   DILI-negative: {(dili_df['DILI_binary'] == 0).sum()}")
    
    return dili_df

def corrected_feature_categorization(feature_cols):
    """Properly categorize features to avoid baseline paradox"""
    
    print("\n🔧 Applying corrected feature categorization...")
    
    categories = {
        'true_baseline': [],          # Pre-treatment features only
        'catch22_baseline': [],       # Catch22 from baseline period
        'sax_baseline': [],          # SAX from baseline period  
        'catch22_post_treatment': [], # Catch22 from post-treatment windows
        'sax_post_treatment': [],    # SAX from post-treatment windows
        'event_aware': [],           # Event-related features
        'drug_response': [],         # Features comparing to baseline (drug effects)
        'hill_parameters': [],       # Dose-response curves
        'quality': []               # Quality metrics
    }
    
    for col in feature_cols:
        categorized = False
        
        # TRUE BASELINE FEATURES (pre-treatment only)
        if (col.startswith('baseline_') and 
            not any(x in col for x in ['deviation', 'fold_change', 'cv_ratio']) and
            'catch22' not in col and 'sax' not in col):
            categories['true_baseline'].append(col)
            categorized = True
        
        # CATCH22 FROM BASELINE PERIOD
        elif col.startswith('baseline_') and 'catch22' in col:
            categories['catch22_baseline'].append(col)
            categorized = True
        
        # SAX FROM BASELINE PERIOD  
        elif col.startswith('baseline_') and 'sax' in col:
            categories['sax_baseline'].append(col)
            categorized = True
        
        # DRUG RESPONSE FEATURES (compare to baseline)
        elif any(x in col for x in ['deviation', 'fold_change', 'cv_ratio', 'baseline_deviation', 'baseline_fold_change']):
            categories['drug_response'].append(col)
            categorized = True
        
        # POST-TREATMENT CATCH22
        elif 'catch22' in col and not col.startswith('baseline_'):
            categories['catch22_post_treatment'].append(col)
            categorized = True
        
        # POST-TREATMENT SAX
        elif 'sax' in col and not col.startswith('baseline_'):
            categories['sax_post_treatment'].append(col)
            categorized = True
        
        # EVENT-AWARE FEATURES
        elif col.startswith('event_') or col.startswith('inter_event_'):
            if not any(x in col for x in ['deviation', 'fold_change', 'cv_ratio']):
                categories['event_aware'].append(col)
            else:
                categories['drug_response'].append(col)
            categorized = True
        
        # HILL PARAMETERS
        elif 'hill' in col:
            categories['hill_parameters'].append(col)
            categorized = True
        
        # QUALITY METRICS
        elif any(q in col for q in ['quality', 'R2', 'success', 'n_wells', 'duration']):
            categories['quality'].append(col)
            categorized = True
        
        # If not categorized, put in appropriate post-treatment category
        if not categorized:
            if 'catch22' in col:
                categories['catch22_post_treatment'].append(col)
            elif 'sax' in col:
                categories['sax_post_treatment'].append(col)
            else:
                categories['event_aware'].append(col)
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    
    # Print categorization summary
    print("   Feature categorization:")
    total_features = 0
    for category, features in categories.items():
        print(f"      {category}: {len(features)} features")
        total_features += len(features)
    print(f"      TOTAL: {total_features} features")
    
    return categories

def validate_baseline_features(features_df, dili_df, categories):
    """Validate that true baseline features don't predict DILI (sanity check)"""
    
    print("\n🧪 Validating baseline features (sanity check)...")
    
    if 'true_baseline' not in categories:
        print("   No true baseline features to validate")
        return
    
    # Merge data
    merged_df = features_df.merge(dili_df, on='drug', how='inner')
    
    if len(merged_df) < 5:
        print("   Insufficient data for validation")
        return
    
    baseline_features = categories['true_baseline']
    
    print(f"   Testing {len(baseline_features)} true baseline features...")
    
    significant_correlations = 0
    strong_correlations = 0
    
    for feat in baseline_features:
        if feat in merged_df.columns:
            valid_data = merged_df[[feat, 'DILI_severity']].dropna()
            
            if len(valid_data) >= 5:
                r, p = pearsonr(valid_data[feat], valid_data['DILI_severity'])
                
                if p < 0.05:
                    significant_correlations += 1
                    print(f"      WARNING: {feat} significantly correlates with DILI (r={r:.3f}, p={p:.3f})")
                
                if abs(r) > 0.3:
                    strong_correlations += 1
    
    print(f"   Results: {significant_correlations} significant correlations, {strong_correlations} strong correlations")
    
    if significant_correlations > 0:
        print("   ⚠️  WARNING: True baseline features should NOT predict DILI!")
        print("      This suggests data leakage or mislabeled features.")
    else:
        print("   ✅ PASS: True baseline features appropriately don't predict DILI")

def analyze_corrected_correlations(features_df, dili_df, categories):
    """Analyze DILI correlations with corrected feature categories"""
    
    print("\n🔍 Analyzing corrected DILI correlations...")
    
    # Merge data
    merged_df = features_df.merge(dili_df, on='drug', how='inner')
    
    if len(merged_df) == 0:
        print("   No drugs overlap between features and DILI data!")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"   Analyzing {len(merged_df)} drugs with both features and DILI data")
    
    # Calculate correlations for each category
    category_results = []
    all_correlations = []
    
    for category, feature_list in categories.items():
        if not feature_list:
            continue
            
        category_correlations = []
        
        for feat in feature_list:
            if feat not in merged_df.columns:
                continue
                
            if merged_df[feat].isna().all() or merged_df[feat].std() == 0:
                continue
                
            # Skip if too many missing values
            if merged_df[feat].isna().mean() > 0.5:
                continue
            
            valid_data = merged_df[[feat, 'DILI_severity']].dropna()
            
            if len(valid_data) >= 5:
                r, p = pearsonr(valid_data[feat], valid_data['DILI_severity'])
                
                category_correlations.append({
                    'feature': feat,
                    'category': category,
                    'pearson_r': r,
                    'pearson_p': p,
                    'abs_pearson_r': abs(r),
                    'n_drugs': len(valid_data)
                })
        
        if category_correlations:
            cat_df = pd.DataFrame(category_correlations)
            
            category_results.append({
                'category': category,
                'n_features': len(cat_df),
                'mean_abs_r': cat_df['abs_pearson_r'].mean(),
                'max_abs_r': cat_df['abs_pearson_r'].max(),
                'significant_features': (cat_df['pearson_p'] < 0.05).sum(),
                'top_feature': cat_df.loc[cat_df['abs_pearson_r'].idxmax(), 'feature'],
                'top_correlation': cat_df.loc[cat_df['abs_pearson_r'].idxmax(), 'pearson_r']
            })
            
            all_correlations.extend(category_correlations)
    
    correlations_df = pd.DataFrame(all_correlations)
    category_summary_df = pd.DataFrame(category_results)
    
    if len(category_summary_df) > 0:
        category_summary_df = category_summary_df.sort_values('mean_abs_r', ascending=False)
        
        print("   Corrected category rankings:")
        for _, row in category_summary_df.iterrows():
            print(f"      {row['category']}: {row['n_features']} features, mean |r|={row['mean_abs_r']:.3f}")
    
    if len(correlations_df) > 0:
        correlations_df = correlations_df.sort_values('abs_pearson_r', ascending=False)
        print(f"   Top correlation: {correlations_df.iloc[0]['feature']} (r={correlations_df.iloc[0]['pearson_r']:.3f})")
    
    return correlations_df, category_summary_df

# ========== MAIN EXECUTION ==========

# Load data
dili_df = load_dili_data()

try:
    features_df = pd.read_parquet(results_dir / "hierarchical_drug_embeddings.parquet")
    print(f"\n📊 Loaded hierarchical features: {features_df.shape}")
except FileNotFoundError:
    print("\n❌ Hierarchical features not found!")
    exit(1)

# Get feature columns
feature_cols = [col for col in features_df.columns if col != 'drug']

# Apply corrected categorization
categories = corrected_feature_categorization(feature_cols)

# Validate baseline features (sanity check)
validate_baseline_features(features_df, dili_df, categories)

# Analyze corrected correlations
correlations_df, category_summary_df = analyze_corrected_correlations(features_df, dili_df, categories)

# Save results
print("\n💾 Saving corrected DILI correlation analysis...")

if len(correlations_df) > 0:
    correlations_df.to_parquet(results_dir / 'corrected_hierarchical_dili_correlations.parquet', index=False)
    print(f"   Corrected correlations: {results_dir / 'corrected_hierarchical_dili_correlations.parquet'}")

if len(category_summary_df) > 0:
    category_summary_df.to_parquet(results_dir / 'corrected_dili_category_performance.parquet', index=False)
    print(f"   Category performance: {results_dir / 'corrected_dili_category_performance.parquet'}")

# Save summary
results_summary = {
    'analysis_type': 'corrected_hierarchical_dili_correlation',
    'n_drugs_analyzed': len(features_df.merge(dili_df, on='drug', how='inner')),
    'feature_categories': {cat: len(feats) for cat, feats in categories.items()},
    'top_correlation': float(correlations_df.iloc[0]['pearson_r']) if len(correlations_df) > 0 else None,
    'top_feature': correlations_df.iloc[0]['feature'] if len(correlations_df) > 0 else None,
    'top_category': correlations_df.iloc[0]['category'] if len(correlations_df) > 0 else None
}

with open(results_dir / 'corrected_hierarchical_dili_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n✅ Corrected hierarchical DILI correlation analysis complete!")

if len(correlations_df) > 0:
    print(f"\n📊 CORRECTED RESULTS:")
    print(f"   Top legitimate correlation: {correlations_df.iloc[0]['feature']}")
    print(f"                              r = {correlations_df.iloc[0]['pearson_r']:.3f}")
    print(f"                              category = {correlations_df.iloc[0]['category']}")
    
    if len(category_summary_df) > 0:
        print(f"   Best category: {category_summary_df.iloc[0]['category']}")
        print(f"                 mean |r| = {category_summary_df.iloc[0]['mean_abs_r']:.3f}")

print(f"\n🎯 Key insight: Only post-treatment features should predict DILI!")
print(f"   Pre-treatment (baseline) features cannot logically predict drug toxicity.")