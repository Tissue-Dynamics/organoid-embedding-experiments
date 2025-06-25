#!/usr/bin/env python3
"""
Comprehensive DILI Visualization and Modeling

PURPOSE:
    Create comprehensive visualizations and machine learning models for the expanded 
    DILI dataset with corrected likelihood mapping (A=most dangerous, E=least dangerous).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "comprehensive_dili_final"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("COMPREHENSIVE DILI VISUALIZATION AND MODELING")
print("=" * 80)

# Load the results from the efficient analysis
print("\n📊 Loading comprehensive DILI dataset...")
final_df = pd.read_parquet(results_dir / 'efficient_comprehensive_dili_dataset.parquet')
correlations = pd.read_csv(results_dir / 'efficient_comprehensive_correlations.csv')

print(f"   Loaded {len(final_df)} drugs with {final_df.shape[1]} features")
print(f"   DILI distribution:")
print(f"     Binary positive: {final_df['dili_binary'].sum()}")
print(f"     Binary negative: {(final_df['dili_binary'] == False).sum()}")
print(f"     Likelihood A (most dangerous): {(final_df['dili_likelihood'] == 'A').sum()}")
print(f"     Likelihood E (least dangerous): {(final_df['dili_likelihood'] == 'E').sum()}")

# ========== VISUALIZATION 1: CORRELATION OVERVIEW ==========

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Top correlations bar plot
ax1 = axes[0, 0]
top_15 = correlations.head(15)

# Color by significance and direction
colors = []
for _, row in top_15.iterrows():
    if row['p'] < 0.001:
        colors.append('darkred' if row['r'] > 0 else 'darkblue')
    elif row['p'] < 0.01:
        colors.append('red' if row['r'] > 0 else 'blue')
    elif row['p'] < 0.05:
        colors.append('lightcoral' if row['r'] > 0 else 'lightblue')
    else:
        colors.append('gray')

bars = ax1.barh(range(len(top_15)), top_15['r'], color=colors, alpha=0.8)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([f.replace('_', ' ')[:25] for f in top_15['feature']], fontsize=10)
ax1.set_xlabel('Pearson Correlation with DILI Likelihood', fontsize=12)
ax1.set_title('Top 15 Feature Correlations\n(Darker = More Significant)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(0, color='black', linewidth=0.5)

# Add significance markers
for i, (_, row) in enumerate(top_15.iterrows()):
    if row['p'] < 0.001:
        marker = '***'
    elif row['p'] < 0.01:
        marker = '**'
    elif row['p'] < 0.05:
        marker = '*'
    else:
        marker = ''
    
    if marker:
        x_pos = row['r'] + (0.02 if row['r'] > 0 else -0.02)
        ax1.text(x_pos, i, marker, va='center', ha='left' if row['r'] > 0 else 'right', fontsize=10)

# 2. P-value vs correlation scatter
ax2 = axes[0, 1]
scatter = ax2.scatter(correlations['abs_r'], -np.log10(correlations['p'] + 1e-10), 
                     c=correlations['r'], cmap='RdBu_r', alpha=0.7, s=50)
ax2.set_xlabel('Absolute Correlation |r|', fontsize=12)
ax2.set_ylabel('-log10(p-value)', fontsize=12)
ax2.set_title('Feature Significance vs Correlation Strength', fontsize=13, fontweight='bold')
ax2.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
ax2.axhline(-np.log10(0.01), color='darkred', linestyle='--', alpha=0.7, label='p=0.01')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Correlation (r)')

# 3. DILI distribution by likelihood
ax3 = axes[1, 0]
likelihood_counts = final_df['dili_likelihood'].value_counts()
likelihood_order = ['A', 'B', 'C', 'D', 'E', 'E*']
colors_likelihood = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green']

plot_counts = []
plot_colors = []
plot_labels = []
for i, likelihood in enumerate(likelihood_order):
    if likelihood in likelihood_counts.index:
        plot_counts.append(likelihood_counts[likelihood])
        plot_colors.append(colors_likelihood[i])
        plot_labels.append(f'{likelihood} ({likelihood_counts[likelihood]})')

bars = ax3.bar(range(len(plot_counts)), plot_counts, color=plot_colors, alpha=0.8)
ax3.set_xticks(range(len(plot_counts)))
ax3.set_xticklabels([l.split(' ')[0] for l in plot_labels])
ax3.set_ylabel('Number of Drugs', fontsize=12)
ax3.set_xlabel('DILI Likelihood Category', fontsize=12)
ax3.set_title('Drug Distribution by DILI Likelihood\n(A=Most Dangerous, E=Least Dangerous)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for bar, count in zip(bars, plot_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Feature type breakdown
ax4 = axes[1, 1]

# Categorize features by type
feature_types = {
    'Statistical': ['mean', 'std', 'min', 'max', 'range', 'cv', 'skew', 'kurt'],
    'Phase-based': ['baseline', 'immediate', 'early', 'late'],
    'Dynamic': ['rolling'],
    'Event': ['event', 'n_events'],
    'Trend': ['slope', 'trend']
}

feature_cols = [col for col in final_df.columns if col not in ['drug', 'n_wells', 'dili_binary', 'dili_likelihood', 'dili_severity', 'dili_category', 'dili_likelihood_numeric']]

type_counts = {}
for feat_type, keywords in feature_types.items():
    count = sum(1 for col in feature_cols if any(keyword in col for keyword in keywords))
    type_counts[feat_type] = count

# Add 'Other' category
other_count = len(feature_cols) - sum(type_counts.values())
if other_count > 0:
    type_counts['Other'] = other_count

pie = ax4.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
ax4.set_title(f'Feature Type Distribution\n(Total: {len(feature_cols)} features)', fontsize=13, fontweight='bold')

plt.suptitle('Comprehensive DILI Analysis Overview', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'comprehensive_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== VISUALIZATION 2: TOP FEATURE ANALYSIS ==========

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Get top 4 features
top_4_features = correlations.head(4)

for i, (_, row) in enumerate(top_4_features.iterrows()):
    ax = axes[i//2, i%2]
    feature = row['feature']
    correlation = row['r']
    p_value = row['p']
    
    # Create scatter plot with likelihood categories
    plot_data = final_df[[feature, 'dili_likelihood', 'dili_likelihood_numeric', 'drug']].dropna()
    
    # Color mapping for likelihood
    likelihood_colors = {'A': 'darkred', 'B': 'red', 'C': 'orange', 'D': 'yellow', 'E': 'lightgreen', 'E*': 'green'}
    colors = [likelihood_colors.get(likelihood, 'gray') for likelihood in plot_data['dili_likelihood']]
    
    scatter = ax.scatter(plot_data['dili_likelihood_numeric'], plot_data[feature], 
                        c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=1)
    
    # Add trend line
    z = np.polyfit(plot_data['dili_likelihood_numeric'], plot_data[feature], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(plot_data['dili_likelihood_numeric'].min(), plot_data['dili_likelihood_numeric'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # Customize plot
    ax.set_xlabel('DILI Likelihood (Numeric)', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' '), fontsize=12)
    
    # Create custom x-axis labels
    unique_numeric = sorted(plot_data['dili_likelihood_numeric'].unique())
    reverse_mapping = {6: 'A', 5: 'B', 4: 'C', 3: 'D', 2: 'E', 1: 'E*'}
    ax.set_xticks(unique_numeric)
    ax.set_xticklabels([reverse_mapping.get(x, str(x)) for x in unique_numeric])
    
    ax.set_title(f'{feature.replace("_", " ")}\nr = {correlation:.3f}, p = {p_value:.3f}', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Top 4 DILI Predictive Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'top_features_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== MACHINE LEARNING MODELS ==========

print("\n🤖 Training machine learning models...")

# Prepare feature matrix
exclude_cols = ['drug', 'n_wells', 'dili_binary', 'dili_likelihood', 'dili_severity', 'dili_category', 'dili_likelihood_numeric']
feature_cols = [col for col in final_df.columns if col not in exclude_cols]

X = final_df[feature_cols].fillna(final_df[feature_cols].median())
# Remove zero variance features
zero_var_mask = X.std() > 0
X = X.loc[:, zero_var_mask]
print(f"   Using {X.shape[1]} features after removing zero variance")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define targets
y_binary = final_df['dili_binary'].astype(int).values
y_likelihood = final_df['dili_likelihood_numeric'].values

# ========== BINARY CLASSIFICATION ==========

print("   Training binary classification models...")

# Feature selection for binary classification
selector_binary = SelectKBest(f_classif, k=min(15, X.shape[1]))
X_binary = selector_binary.fit_transform(X_scaled, y_binary)

binary_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
binary_results = {}

for model_name, model in binary_models.items():
    try:
        cv_scores = cross_val_score(model, X_binary, y_binary, cv=cv, scoring='roc_auc')
        y_proba = cross_val_predict(model, X_binary, y_binary, cv=cv, method='predict_proba')
        auroc = roc_auc_score(y_binary, y_proba[:, 1])
        
        binary_results[model_name] = {
            'auroc': auroc,
            'cv_scores': cv_scores,
            'mean_cv': cv_scores.mean(),
            'std_cv': cv_scores.std()
        }
        print(f"     {model_name}: AUROC = {auroc:.3f} ± {cv_scores.std():.3f}")
    except Exception as e:
        print(f"     {model_name}: Failed - {e}")

# ========== LIKELIHOOD REGRESSION ==========

print("   Training likelihood prediction models...")

# Feature selection for likelihood
selector_likelihood = SelectKBest(f_regression, k=min(15, X.shape[1]))
X_likelihood = selector_likelihood.fit_transform(X_scaled, y_likelihood)

likelihood_models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=1.0)
}

likelihood_results = {}

for model_name, model in likelihood_models.items():
    try:
        cv_scores = cross_val_score(model, X_likelihood, y_likelihood, cv=5, scoring='r2')
        
        likelihood_results[model_name] = {
            'r2': cv_scores.mean(),
            'cv_scores': cv_scores,
            'std_cv': cv_scores.std()
        }
        print(f"     {model_name}: R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    except Exception as e:
        print(f"     {model_name}: Failed - {e}")

# ========== VISUALIZATION 3: MODEL PERFORMANCE ==========

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Binary classification performance
if binary_results:
    ax1 = axes[0]
    model_names = list(binary_results.keys())
    aurocs = [binary_results[m]['auroc'] for m in model_names]
    auroc_stds = [binary_results[m]['std_cv'] for m in model_names]
    
    bars = ax1.bar(range(len(model_names)), aurocs, yerr=auroc_stds, capsize=10, 
                   alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title('Binary DILI Classification Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, auroc, std in zip(bars, aurocs, auroc_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{auroc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Likelihood regression performance
if likelihood_results:
    ax2 = axes[1]
    model_names_lik = list(likelihood_results.keys())
    r2s = [likelihood_results[m]['r2'] for m in model_names_lik]
    r2_stds = [likelihood_results[m]['std_cv'] for m in model_names_lik]
    
    bars = ax2.bar(range(len(model_names_lik)), r2s, yerr=r2_stds, capsize=10, 
                   alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=1)
    ax2.set_xticks(range(len(model_names_lik)))
    ax2.set_xticklabels(model_names_lik, rotation=45, ha='right')
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('DILI Likelihood Prediction Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, r2, std in zip(bars, r2s, r2_stds):
        height = bar.get_height()
        label_y = height + 0.02 if height >= 0 else height - 0.02
        ax2.text(bar.get_x() + bar.get_width()/2, label_y, 
                f'{r2:.3f}±{std:.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== VISUALIZATION 4: ROC CURVES ==========

if binary_results:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, model in binary_models.items():
        if model_name in binary_results:
            try:
                y_proba = cross_val_predict(model, X_binary, y_binary, cv=cv, method='predict_proba')
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, 1])
                auroc = binary_results[model_name]['auroc']
                
                ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUROC = {auroc:.3f})')
            except:
                continue
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Binary DILI Classification', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== SAVE COMPREHENSIVE RESULTS ==========

print("\n💾 Saving comprehensive results...")

# Compile comprehensive results
comprehensive_results = {
    'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_summary': {
        'n_drugs': len(final_df),
        'n_features': len(feature_cols),
        'n_wells_processed': int(final_df['n_wells'].sum()),
        'dili_distribution': {
            'binary_positive': int(final_df['dili_binary'].sum()),
            'binary_negative': int((final_df['dili_binary'] == False).sum()),
            'likelihood_distribution': final_df['dili_likelihood'].value_counts().to_dict()
        }
    },
    'top_correlations': correlations.head(10).to_dict('records'),
    'model_performance': {
        'binary_classification': binary_results if binary_results else {},
        'likelihood_regression': likelihood_results if likelihood_results else {}
    },
    'best_performers': {
        'binary_auroc': max(binary_results.items(), key=lambda x: x[1]['auroc']) if binary_results else None,
        'likelihood_r2': max(likelihood_results.items(), key=lambda x: x[1]['r2']) if likelihood_results else None
    }
}

# Save results
with open(results_dir / 'comprehensive_dili_final_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2, default=str)

print("\n✅ Comprehensive DILI analysis complete!")
print(f"\n📊 FINAL SUMMARY:")
print(f"   Drugs analyzed: {len(final_df)}")
print(f"   Features extracted: {len(feature_cols)}")
print(f"   Wells processed: {int(final_df['n_wells'].sum()):,}")

if correlations is not None and len(correlations) > 0:
    best_feature = correlations.iloc[0]
    print(f"\n🎯 Best predictor: {best_feature['feature']} (r={best_feature['r']:.3f}, p={best_feature['p']:.3f})")

if binary_results:
    best_binary = max(binary_results.items(), key=lambda x: x[1]['auroc'])
    print(f"   Best binary model: {best_binary[0]} (AUROC={best_binary[1]['auroc']:.3f})")

if likelihood_results:
    best_likelihood = max(likelihood_results.items(), key=lambda x: x[1]['r2'])
    print(f"   Best likelihood model: {best_likelihood[0]} (R²={best_likelihood[1]['r2']:.3f})")

print(f"\nFigures saved to: {fig_dir}")
print(f"Results saved to: {results_dir}")