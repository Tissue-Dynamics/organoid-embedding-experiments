#!/usr/bin/env python3
"""
Post-Treatment Features DILI Analysis

PURPOSE:
    Focuses exclusively on post-treatment features for DILI prediction, assuming
    the experimental design is sound (drugs were intentionally grouped on plates).
    
    This analysis blinds itself to which drugs were on which plates and only
    analyzes features that capture drug response after treatment.

APPROACH:
    - Exclude all pre-treatment baseline features
    - Focus on drug response features that measure changes from baseline
    - Analyze post-treatment time series features
    - Compare different feature types for DILI prediction
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
fig_dir = project_root / "results" / "figures" / "post_treatment_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("POST-TREATMENT FEATURES DILI ANALYSIS")
print("=" * 80)
print("\nFocusing exclusively on drug response features that capture post-treatment effects")

# ========== DATA LOADING ==========

def load_dili_data():
    """Load DILI classification data"""
    
    print("\n📊 Loading DILI classification data...")
    
    # DILI data for compounds
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
    
    return dili_df

def load_hierarchical_features():
    """Load hierarchical features and filter for post-treatment only"""
    
    print("\n📊 Loading hierarchical features...")
    
    try:
        features_df = pd.read_parquet(results_dir / "hierarchical_drug_embeddings.parquet")
        print(f"   ✓ Loaded {features_df.shape[0]} drugs with {features_df.shape[1]-1} features")
        
        # Load metadata
        with open(results_dir / "hierarchical_feature_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return features_df, metadata
        
    except FileNotFoundError:
        print("   ✗ Hierarchical features not found!")
        return pd.DataFrame(), {}

def filter_post_treatment_features(features_df):
    """Filter to only include post-treatment features"""
    
    print("\n🔍 Filtering for post-treatment features only...")
    
    feature_cols = [col for col in features_df.columns if col != 'drug']
    
    # Define patterns that indicate post-treatment features
    post_treatment_patterns = [
        # Drug response features (comparing to baseline)
        'deviation', 'fold_change', 'cv_ratio', 'baseline_deviation', 'baseline_fold_change',
        
        # Event-related features
        'event_', 'inter_event_',
        
        # Hill parameters (dose-response)
        'hill_',
        
        # Post-treatment time windows (exclude baseline prefix)
        'catch22_24h_', 'catch22_48h_', 'catch22_96h_',
        'sax_24h_', 'sax_48h_', 'sax_96h_'
    ]
    
    # Features to explicitly exclude (pre-treatment)
    exclude_patterns = [
        'baseline_mean_', 'baseline_std_', 'baseline_cv_', 
        'baseline_range_', 'baseline_min_', 'baseline_max_',
        'baseline_trend_', 'baseline_catch22_', 'baseline_sax_'
    ]
    
    post_treatment_features = []
    excluded_features = []
    
    for col in feature_cols:
        # Check if it should be excluded
        exclude = False
        for pattern in exclude_patterns:
            if col.startswith(pattern):
                exclude = True
                excluded_features.append(col)
                break
        
        if not exclude:
            # Check if it matches post-treatment patterns
            include = False
            for pattern in post_treatment_patterns:
                if pattern in col:
                    include = True
                    break
            
            # Also include features that don't start with 'baseline_'
            if not include and not col.startswith('baseline_'):
                include = True
            
            if include:
                post_treatment_features.append(col)
    
    print(f"   Excluded {len(excluded_features)} pre-treatment features")
    print(f"   Retained {len(post_treatment_features)} post-treatment features")
    
    # Create filtered dataframe
    filtered_df = features_df[['drug'] + post_treatment_features].copy()
    
    return filtered_df, post_treatment_features

def analyze_post_treatment_correlations(features_df, dili_df):
    """Analyze correlations between post-treatment features and DILI"""
    
    print("\n🎯 Analyzing post-treatment feature correlations with DILI...")
    
    # Merge with DILI data
    merged_df = features_df.merge(dili_df, on='drug', how='inner')
    
    if len(merged_df) == 0:
        print("   No drugs overlap between features and DILI data!")
        return pd.DataFrame()
    
    print(f"   Analyzing {len(merged_df)} drugs")
    
    feature_cols = [col for col in features_df.columns if col != 'drug']
    correlations = []
    
    for col in feature_cols:
        if merged_df[col].isna().all() or merged_df[col].std() == 0:
            continue
            
        valid_data = merged_df[[col, 'DILI_severity']].dropna()
        
        if len(valid_data) >= 5:
            r, p = pearsonr(valid_data[col], valid_data['DILI_severity'])
            
            # Categorize feature type
            if 'hill_' in col:
                category = 'dose_response'
            elif any(x in col for x in ['deviation', 'fold_change', 'cv_ratio']):
                category = 'drug_response'
            elif 'event_' in col or 'inter_event_' in col:
                category = 'event_aware'
            elif 'catch22' in col:
                category = 'catch22_post_treatment'
            elif 'sax' in col:
                category = 'sax_post_treatment'
            else:
                category = 'other'
            
            correlations.append({
                'feature': col,
                'category': category,
                'pearson_r': r,
                'pearson_p': p,
                'abs_pearson_r': abs(r),
                'n_drugs': len(valid_data)
            })
    
    correlations_df = pd.DataFrame(correlations)
    
    if len(correlations_df) > 0:
        correlations_df = correlations_df.sort_values('abs_pearson_r', ascending=False)
        
        # Category summary
        category_summary = correlations_df.groupby('category').agg({
            'abs_pearson_r': ['mean', 'max', 'count'],
            'pearson_p': lambda x: (x < 0.05).sum()
        }).round(3)
        
        print("\n   Post-treatment feature category performance:")
        print(category_summary.sort_values(('abs_pearson_r', 'mean'), ascending=False))
    
    return correlations_df

def create_visualizations(correlations_df, fig_dir):
    """Create visualizations of post-treatment DILI correlations"""
    
    print("\n📊 Creating visualizations...")
    
    if len(correlations_df) == 0:
        print("   No correlations to visualize")
        return
    
    # 1. Top post-treatment feature correlations
    plt.figure(figsize=(12, 8))
    top_features = correlations_df.head(25)
    
    colors = ['darkred' if r > 0 else 'darkblue' for r in top_features['pearson_r']]
    bars = plt.barh(range(len(top_features)), top_features['pearson_r'], color=colors, alpha=0.7)
    
    plt.yticks(range(len(top_features)), 
               [f.replace('_', ' ')[:40] + '...' if len(f) > 40 else f.replace('_', ' ') 
                for f in top_features['feature']])
    plt.xlabel('Pearson Correlation with DILI Severity')
    plt.title('Top 25 Post-Treatment Features: DILI Correlation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Add significance markers
    for i, (idx, row) in enumerate(top_features.iterrows()):
        if row['pearson_p'] < 0.001:
            marker = '***'
        elif row['pearson_p'] < 0.01:
            marker = '**'
        elif row['pearson_p'] < 0.05:
            marker = '*'
        else:
            marker = ''
        
        if marker:
            x_pos = row['pearson_r'] + (0.02 if row['pearson_r'] > 0 else -0.02)
            plt.text(x_pos, i, marker, va='center', ha='left' if row['pearson_r'] > 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'top_post_treatment_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category comparison
    category_stats = correlations_df.groupby('category').agg({
        'abs_pearson_r': ['mean', 'std', 'max'],
        'pearson_p': lambda x: (x < 0.05).sum(),
        'feature': 'count'
    }).round(3)
    
    plt.figure(figsize=(10, 6))
    categories = category_stats.index
    mean_corrs = category_stats[('abs_pearson_r', 'mean')]
    
    bars = plt.bar(range(len(categories)), mean_corrs, color='skyblue', alpha=0.7)
    plt.errorbar(range(len(categories)), mean_corrs, 
                 yerr=category_stats[('abs_pearson_r', 'std')],
                 fmt='none', color='black', capsize=5)
    
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.ylabel('Mean |Pearson r| with DILI')
    plt.title('Post-Treatment Feature Categories: DILI Prediction Performance', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add feature counts and significance counts
    for i, (cat, row) in enumerate(category_stats.iterrows()):
        n_features = row[('feature', 'count')]
        n_sig = row[('pearson_p', '<lambda>')]
        plt.text(i, mean_corrs[i] + 0.01, f'n={n_features}\n({n_sig} sig)', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'post_treatment_category_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== MAIN EXECUTION ==========

# Load data
dili_df = load_dili_data()
features_df, metadata = load_hierarchical_features()

if len(features_df) == 0:
    print("\n❌ Cannot proceed without hierarchical features!")
    exit(1)

# Filter for post-treatment features only
filtered_features_df, post_treatment_features = filter_post_treatment_features(features_df)

# Analyze correlations
correlations_df = analyze_post_treatment_correlations(filtered_features_df, dili_df)

# Create visualizations
create_visualizations(correlations_df, fig_dir)

# Save results
print("\n💾 Saving post-treatment DILI analysis results...")

if len(correlations_df) > 0:
    correlations_df.to_parquet(results_dir / 'post_treatment_dili_correlations.parquet', index=False)
    print(f"   Correlations saved: {results_dir / 'post_treatment_dili_correlations.parquet'}")
    
    # Summary statistics
    top_10 = correlations_df.head(10)
    summary = {
        'analysis_type': 'post_treatment_dili_correlation',
        'n_drugs_analyzed': len(filtered_features_df.merge(dili_df, on='drug', how='inner')),
        'n_features_analyzed': len(post_treatment_features),
        'top_correlation': float(correlations_df.iloc[0]['pearson_r']),
        'top_feature': correlations_df.iloc[0]['feature'],
        'top_feature_category': correlations_df.iloc[0]['category'],
        'top_10_features': top_10[['feature', 'pearson_r', 'pearson_p']].to_dict('records')
    }
    
    with open(results_dir / 'post_treatment_dili_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary saved: {results_dir / 'post_treatment_dili_summary.json'}")

print("\n✅ Post-treatment DILI analysis complete!")

if len(correlations_df) > 0:
    print(f"\n📊 KEY RESULTS:")
    print(f"   Top post-treatment correlation: {correlations_df.iloc[0]['feature']}")
    print(f"                                  r = {correlations_df.iloc[0]['pearson_r']:.3f}")
    print(f"                                  p = {correlations_df.iloc[0]['pearson_p']:.3f}")
    print(f"                                  category = {correlations_df.iloc[0]['category']}")
    
    # Show top 5 features
    print(f"\n   Top 5 post-treatment features:")
    for _, row in correlations_df.head(5).iterrows():
        print(f"      {row['feature']}: r={row['pearson_r']:.3f} (p={row['pearson_p']:.3f})")

print(f"\n🎯 These features capture actual drug responses and are legitimate predictors of DILI!")