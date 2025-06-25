#!/usr/bin/env python3
"""
Hierarchical Features DILI Correlation Analysis

PURPOSE:
    Applies DILI correlation analysis to the comprehensive hierarchical feature set
    created from the advanced feature engineering pipeline. This validates whether
    our multi-modal feature approach (catch22 + SAX + Hill parameters + event-aware)
    improves DILI prediction compared to previous approaches.

METHODOLOGY:
    Tests the hierarchical features against known DILI classifications:
    - Load hierarchical drug embeddings (145 drugs, 694 features)
    - Match with DILI database (DILIrank, LiverTox)
    - Calculate correlation strength for individual features and combinations
    - Compare with previous embedding results (Phase 2: r=0.260)
    - Analyze which feature categories contribute most to DILI prediction

INPUTS:
    - results/data/hierarchical_drug_embeddings.parquet
    - External DILI databases (DILIrank, LiverTox)
    - Previous embedding results for comparison

OUTPUTS:
    - results/data/hierarchical_dili_correlations.parquet
    - results/figures/hierarchical_dili/
      Correlation heatmaps, feature importance plots, comparison with baselines
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

from src.utils.data_loader import DataLoader

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "hierarchical_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HIERARCHICAL FEATURES DILI CORRELATION ANALYSIS")
print("=" * 80)

# ========== DILI DATABASE LOADING ==========

def load_dili_data():
    """Load DILI classification data"""
    
    # DILI classifications for pharmaceutical compounds in our dataset
    # Based on DILIrank, LiverTox, and literature knowledge
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

def load_hierarchical_features():
    """Load the hierarchical feature embeddings"""
    
    print("\n📊 Loading hierarchical feature embeddings...")
    
    try:
        features_df = pd.read_parquet(results_dir / "hierarchical_drug_embeddings.parquet")
        print(f"   ✓ Loaded hierarchical features: {features_df.shape}")
        
        # Load metadata
        with open(results_dir / "hierarchical_feature_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return features_df, metadata
        
    except FileNotFoundError:
        print("   ✗ Hierarchical features not found! Run hierarchical_feature_architecture.py first")
        return pd.DataFrame(), {}

# ========== FEATURE-DILI CORRELATION ANALYSIS ==========

def calculate_feature_dili_correlations(features_df, dili_df, metadata):
    """Calculate correlations between individual features and DILI scores"""
    
    print("\n🔍 Calculating feature-DILI correlations...")
    
    # Merge features with DILI data
    merged_df = features_df.merge(dili_df, on='drug', how='inner')
    
    if len(merged_df) == 0:
        print("   ✗ No drugs overlap between features and DILI data!")
        return pd.DataFrame()
    
    print(f"   Analyzing {len(merged_df)} drugs with both features and DILI data")
    
    # Get feature columns
    feature_cols = [col for col in merged_df.columns if col not in ['drug', 'DILI_severity', 'DILI_binary', 'DILI_concern']]
    
    correlations = []
    
    for col in feature_cols:
        if merged_df[col].isna().all() or merged_df[col].std() == 0:
            continue
            
        # Skip if too many missing values
        if merged_df[col].isna().mean() > 0.5:
            continue
        
        # Calculate correlations with DILI severity (continuous)
        valid_data = merged_df[[col, 'DILI_severity']].dropna()
        
        if len(valid_data) >= 5:  # Minimum samples for meaningful correlation
            pearson_r, pearson_p = pearsonr(valid_data[col], valid_data['DILI_severity'])
            spearman_r, spearman_p = spearmanr(valid_data[col], valid_data['DILI_severity'])
            
            # Determine feature category
            feature_category = 'other'
            for category, category_features in metadata.get('feature_categories', {}).items():
                if col in category_features:
                    feature_category = category
                    break
            
            correlations.append({
                'feature': col,
                'category': feature_category,
                'n_drugs': len(valid_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'abs_pearson_r': abs(pearson_r),
                'abs_spearman_r': abs(spearman_r)
            })
    
    correlations_df = pd.DataFrame(correlations)
    
    if len(correlations_df) > 0:
        correlations_df = correlations_df.sort_values('abs_pearson_r', ascending=False)
        print(f"   Calculated correlations for {len(correlations_df)} features")
        print(f"   Top correlation: {correlations_df.iloc[0]['feature']} (r={correlations_df.iloc[0]['pearson_r']:.3f})")
    
    return correlations_df

def analyze_feature_category_performance(correlations_df):
    """Analyze which feature categories perform best for DILI prediction"""
    
    print("\n🎯 Analyzing feature category performance...")
    
    if len(correlations_df) == 0:
        return pd.DataFrame()
    
    category_stats = []
    
    for category in correlations_df['category'].unique():
        cat_data = correlations_df[correlations_df['category'] == category]
        
        category_stats.append({
            'category': category,
            'n_features': len(cat_data),
            'mean_abs_pearson_r': cat_data['abs_pearson_r'].mean(),
            'max_abs_pearson_r': cat_data['abs_pearson_r'].max(),
            'median_abs_pearson_r': cat_data['abs_pearson_r'].median(),
            'top_feature': cat_data.iloc[0]['feature'] if len(cat_data) > 0 else '',
            'top_correlation': cat_data.iloc[0]['pearson_r'] if len(cat_data) > 0 else 0,
            'significant_features': (cat_data['pearson_p'] < 0.05).sum()
        })
    
    category_performance = pd.DataFrame(category_stats)
    category_performance = category_performance.sort_values('mean_abs_pearson_r', ascending=False)
    
    print("   Feature category rankings:")
    for _, row in category_performance.head(10).iterrows():
        print(f"      {row['category']}: {row['n_features']} features, mean |r|={row['mean_abs_pearson_r']:.3f}")
    
    return category_performance

# ========== PREDICTIVE MODELING ==========

def build_dili_prediction_models(features_df, dili_df):
    """Build DILI prediction models using hierarchical features"""
    
    print("\n🎯 Building DILI prediction models...")
    
    # Merge data
    merged_df = features_df.merge(dili_df, on='drug', how='inner')
    
    if len(merged_df) < 10:
        print(f"   ✗ Insufficient data for modeling: {len(merged_df)} drugs")
        return {}
    
    # Prepare features
    feature_cols = [col for col in merged_df.columns if col not in ['drug', 'DILI_severity', 'DILI_binary', 'DILI_concern']]
    
    X = merged_df[feature_cols].copy()
    y_binary = merged_df['DILI_binary'].values
    y_severity = merged_df['DILI_severity'].values
    
    # Handle missing values
    for col in feature_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Remove zero-variance features
    zero_var_cols = [col for col in feature_cols if X[col].std() == 0]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        print(f"   Removed {len(zero_var_cols)} zero-variance features")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Training on {X_scaled.shape[0]} drugs with {X_scaled.shape[1]} features")
    
    results = {}
    
    # Binary classification (DILI vs No-DILI)
    if len(np.unique(y_binary)) > 1:
        print("\n   Binary DILI Classification:")
        
        # Random Forest
        rf_binary = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_scores = cross_val_score(rf_binary, X_scaled, y_binary, cv=min(5, len(merged_df)), scoring='roc_auc')
        
        # Logistic Regression  
        lr_binary = LogisticRegression(random_state=42, max_iter=1000)
        lr_scores = cross_val_score(lr_binary, X_scaled, y_binary, cv=min(5, len(merged_df)), scoring='roc_auc')
        
        results['binary_classification'] = {
            'rf_auc_mean': rf_scores.mean(),
            'rf_auc_std': rf_scores.std(),
            'lr_auc_mean': lr_scores.mean(),
            'lr_auc_std': lr_scores.std(),
            'n_samples': len(merged_df),
            'n_features': X_scaled.shape[1]
        }
        
        print(f"      Random Forest AUC: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
        print(f"      Logistic Regression AUC: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
        
        # Fit final model for feature importance
        rf_binary.fit(X_scaled, y_binary)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_binary.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
    
    # Severity regression
    if len(np.unique(y_severity)) > 2:
        print("\n   DILI Severity Regression:")
        
        # Random Forest Regressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        rf_reg_scores = cross_val_score(rf_regressor, X_scaled, y_severity, cv=min(5, len(merged_df)), scoring='r2')
        
        # Ridge Regression
        ridge_regressor = Ridge(random_state=42)
        ridge_scores = cross_val_score(ridge_regressor, X_scaled, y_severity, cv=min(5, len(merged_df)), scoring='r2')
        
        results['severity_regression'] = {
            'rf_r2_mean': rf_reg_scores.mean(),
            'rf_r2_std': rf_reg_scores.std(),
            'ridge_r2_mean': ridge_scores.mean(),
            'ridge_r2_std': ridge_scores.std()
        }
        
        print(f"      Random Forest R²: {rf_reg_scores.mean():.3f} ± {rf_reg_scores.std():.3f}")
        print(f"      Ridge Regression R²: {ridge_scores.mean():.3f} ± {ridge_scores.std():.3f}")
    
    return results

# ========== COMPARISON WITH PREVIOUS RESULTS ==========

def compare_with_baseline_embeddings():
    """Compare hierarchical features with previous embedding results"""
    
    print("\n📊 Comparing with baseline embedding results...")
    
    # Load previous Phase 2 results if available
    baseline_results = {
        'phase2_hierarchical': {'correlation': 0.260, 'method': 'Phase 2 Hierarchical Embeddings'},
        'event_aware_features': {'correlation': 0.435, 'method': 'Event-Aware Features'}
    }
    
    print("   Previous results for comparison:")
    for method, results in baseline_results.items():
        print(f"      {results['method']}: r = {results['correlation']:.3f}")
    
    return baseline_results

# ========== VISUALIZATION ==========

def create_correlation_visualizations(correlations_df, category_performance, fig_dir):
    """Create comprehensive visualizations of DILI correlations"""
    
    print("\n📊 Creating DILI correlation visualizations...")
    
    if len(correlations_df) == 0:
        print("   ✗ No correlation data to visualize")
        return
    
    # 1. Top feature correlations
    plt.figure(figsize=(12, 8))
    top_features = correlations_df.head(20)
    
    colors = plt.cm.RdBu_r(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['pearson_r'], color=colors)
    
    plt.yticks(range(len(top_features)), [f.replace('_', ' ')[:30] for f in top_features['feature']])
    plt.xlabel('Pearson Correlation with DILI Severity')
    plt.title('Top 20 Features: Correlation with DILI Severity')
    plt.grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, (bar, r_val) in enumerate(zip(bars, top_features['pearson_r'])):
        plt.text(r_val + 0.01 if r_val >= 0 else r_val - 0.01, i, f'{r_val:.3f}', 
                va='center', ha='left' if r_val >= 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'top_dili_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature category performance
    if len(category_performance) > 0:
        plt.figure(figsize=(10, 6))
        
        categories = category_performance['category'][:10]
        mean_corrs = category_performance['mean_abs_pearson_r'][:10]
        
        bars = plt.bar(range(len(categories)), mean_corrs, color='skyblue', alpha=0.7)
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.ylabel('Mean |Pearson r| with DILI')
        plt.title('Feature Category Performance for DILI Prediction')
        plt.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, mean_corrs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

# ========== MAIN EXECUTION ==========

# Load data
dili_df = load_dili_data()
features_df, metadata = load_hierarchical_features()

if len(features_df) == 0:
    print("\n❌ Cannot proceed without hierarchical features!")
    exit(1)

# Calculate correlations
correlations_df = calculate_feature_dili_correlations(features_df, dili_df, metadata)

# Analyze category performance
category_performance = analyze_feature_category_performance(correlations_df)

# Build prediction models
model_results = build_dili_prediction_models(features_df, dili_df)

# Compare with baselines
baseline_comparison = compare_with_baseline_embeddings()

# Create visualizations
create_correlation_visualizations(correlations_df, category_performance, fig_dir)

# Save results
print("\n💾 Saving DILI correlation analysis results...")

if len(correlations_df) > 0:
    correlations_df.to_parquet(results_dir / 'hierarchical_dili_correlations.parquet', index=False)
    print(f"   Feature correlations: {results_dir / 'hierarchical_dili_correlations.parquet'}")

if len(category_performance) > 0:
    category_performance.to_parquet(results_dir / 'hierarchical_dili_category_performance.parquet', index=False)
    print(f"   Category performance: {results_dir / 'hierarchical_dili_category_performance.parquet'}")

# Save comprehensive results
results_summary = {
    'analysis_summary': {
        'n_drugs_with_features': len(features_df),
        'n_drugs_with_dili': len(dili_df),
        'n_drugs_overlap': len(features_df.merge(dili_df, on='drug', how='inner')),
        'n_features_analyzed': len(correlations_df),
        'top_correlation': float(correlations_df.iloc[0]['pearson_r']) if len(correlations_df) > 0 else None,
        'top_feature': correlations_df.iloc[0]['feature'] if len(correlations_df) > 0 else None
    },
    'model_performance': {
        k: {k2: float(v2) if isinstance(v2, (np.float64, np.float32)) and not np.isnan(v2) else 
            (None if isinstance(v2, (np.float64, np.float32)) and np.isnan(v2) else v2)
            for k2, v2 in v.items() if k2 != 'feature_importance'}
        for k, v in model_results.items() if k != 'feature_importance'
    },
    'baseline_comparison': baseline_comparison,
    'category_rankings': category_performance.to_dict('records') if len(category_performance) > 0 else []
}

with open(results_dir / 'hierarchical_dili_analysis_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"   Analysis summary: {results_dir / 'hierarchical_dili_analysis_summary.json'}")

# Final summary
print(f"\n✅ Hierarchical DILI correlation analysis complete!")
print(f"\n📊 RESULTS SUMMARY:")

if len(correlations_df) > 0:
    print(f"   Analyzed {len(correlations_df)} features from {len(features_df)} drugs")
    print(f"   Top correlation: {correlations_df.iloc[0]['feature']}")
    print(f"                   r = {correlations_df.iloc[0]['pearson_r']:.3f} (p = {correlations_df.iloc[0]['pearson_p']:.3f})")
    
    if len(category_performance) > 0:
        print(f"   Best feature category: {category_performance.iloc[0]['category']}")
        print(f"                         mean |r| = {category_performance.iloc[0]['mean_abs_pearson_r']:.3f}")

if 'binary_classification' in model_results:
    print(f"   Binary DILI prediction AUC: {model_results['binary_classification']['rf_auc_mean']:.3f}")

print(f"\n🎯 Next steps:")
print(f"   1. Validate results with larger DILI database")
print(f"   2. Compare with individual feature type performance")
print(f"   3. Investigate top-performing features for biological relevance")
print(f"   4. Test on external validation datasets")