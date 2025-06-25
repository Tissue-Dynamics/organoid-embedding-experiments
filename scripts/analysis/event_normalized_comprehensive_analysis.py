#!/usr/bin/env python3
"""
Comprehensive Event-Normalized DILI Analysis

PURPOSE:
    Detailed analysis of event-normalized features including:
    - Correlation visualizations
    - Machine learning models with AUROC
    - Feature importance analysis
    - Cross-validation performance
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import sys

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_normalized_comprehensive"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("COMPREHENSIVE EVENT-NORMALIZED DILI ANALYSIS")
print("=" * 80)

# Load DILI data
def load_dili_data():
    """Load DILI classification data with likelihood categories"""
    
    dili_data = {
        # Most DILI Concern (E*/E - severity 4-5)
        'Amiodarone': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E*'},
        'Busulfan': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        'Imatinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        'Lapatinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        'Pazopanib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        'Regorafenib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E*'},
        'Sorafenib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        'Sunitinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        'Trametinib': {'DILI_severity': 4, 'DILI_binary': 1, 'DILI_concern': 'Most-DILI-Concern', 'DILI_likelihood': 'E'},
        
        # DILI Concern (D - severity 3)
        'Anastrozole': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Axitinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Cabozantinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Dabrafenib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Erlotinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Gefitinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Lenvatinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Nilotinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Osimertinib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        'Vemurafenib': {'DILI_severity': 3, 'DILI_binary': 1, 'DILI_concern': 'DILI-Concern', 'DILI_likelihood': 'D'},
        
        # Less DILI Concern (C - severity 2)
        'Alectinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Binimetinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Bortezomib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Ceritinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Crizotinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Dasatinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Everolimus': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Ibrutinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Ponatinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        'Ruxolitinib': {'DILI_severity': 2, 'DILI_binary': 1, 'DILI_concern': 'Less-DILI-Concern', 'DILI_likelihood': 'C'},
        
        # No DILI Concern (A/B - severity 1)
        'Alpelisib': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'B'},
        'Ambrisentan': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'A'},
        'Buspirone': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'A'},
        'Dexamethasone': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'A'},
        'Fulvestrant': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'B'},
        'Letrozole': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'A'},
        'Palbociclib': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'B'},
        'Ribociclib': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'B'},
        'Trastuzumab': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'A'},
        'Zoledronic acid': {'DILI_severity': 1, 'DILI_binary': 0, 'DILI_concern': 'No-DILI-Concern', 'DILI_likelihood': 'A'}
    }
    
    dili_df = pd.DataFrame.from_dict(dili_data, orient='index').reset_index()
    dili_df.columns = ['drug', 'DILI_severity', 'DILI_binary', 'DILI_concern', 'DILI_likelihood']
    
    return dili_df

# Load data
print("\n📊 Loading data...")
drug_features_df = pd.read_parquet(results_dir / 'event_normalized_features_drugs_final.parquet')
dili_df = load_dili_data()

# Merge
merged_df = drug_features_df.merge(dili_df, on='drug', how='inner')
print(f"   Analyzing {len(merged_df)} drugs with both features and DILI data")
print(f"   DILI positive: {merged_df['DILI_binary'].sum()}")
print(f"   DILI negative: {(merged_df['DILI_binary'] == 0).sum()}")

# Get feature columns
exclude_cols = ['drug', 'n_wells', 'DILI_severity', 'DILI_binary', 'DILI_concern', 'DILI_likelihood']
feature_cols = [col for col in merged_df.columns if col not in exclude_cols]

# Create numeric encoding for DILI likelihood (A=1, B=2, C=3, D=4, E=5, E*=6)
likelihood_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'E*': 6}
merged_df['DILI_likelihood_numeric'] = merged_df['DILI_likelihood'].map(likelihood_mapping)

# ========== 1. CORRELATION ANALYSIS ==========

print("\n📊 Creating correlation visualizations...")

# Calculate all correlations
correlations = []
for feat in feature_cols:
    valid_data = merged_df[[feat, 'DILI_severity']].dropna()
    if len(valid_data) >= 5:
        r, p = pearsonr(valid_data[feat], valid_data['DILI_severity'])
        correlations.append({
            'feature': feat,
            'r': r,
            'p': p,
            'abs_r': abs(r),
            'significant': p < 0.05
        })

corr_df = pd.DataFrame(correlations).sort_values('abs_r', ascending=False)

# Figure 1: Top feature correlations with significance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Subplot 1: Bar plot of top correlations
top_n = 20
top_features = corr_df.head(top_n)

colors = ['darkred' if row['significant'] else 'lightcoral' if row['r'] > 0 
          else 'darkblue' if row['significant'] else 'lightblue' 
          for _, row in top_features.iterrows()]

bars = ax1.barh(range(len(top_features)), top_features['r'], color=colors, alpha=0.8)

# Add significance stars
for i, (_, row) in enumerate(top_features.iterrows()):
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
        ax1.text(x_pos, i, marker, va='center', ha='left' if row['r'] > 0 else 'right', fontsize=12)

ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels([f.replace('_', ' ')[:30] for f in top_features['feature']])
ax1.set_xlabel('Pearson Correlation with DILI Severity', fontsize=12)
ax1.set_title('Top 20 Event-Normalized Features: DILI Correlation', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(0, color='black', linewidth=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkred', label='Positive (p<0.05)'),
    Patch(facecolor='lightcoral', label='Positive (ns)'),
    Patch(facecolor='darkblue', label='Negative (p<0.05)'),
    Patch(facecolor='lightblue', label='Negative (ns)')
]
ax1.legend(handles=legend_elements, loc='lower right')

# Subplot 2: P-value volcano plot
ax2.scatter(corr_df['r'], -np.log10(corr_df['p']), 
           c=['red' if r > 0 else 'blue' for r in corr_df['r']], 
           alpha=0.6, s=50)

# Add significance line
ax2.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
ax2.axhline(-np.log10(0.01), color='gray', linestyle=':', alpha=0.5, label='p=0.01')

# Label top features
for _, row in corr_df.head(5).iterrows():
    ax2.annotate(row['feature'].replace('_', ' ')[:20], 
                (row['r'], -np.log10(row['p'])),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.set_xlabel('Correlation Coefficient (r)', fontsize=12)
ax2.set_ylabel('-log10(p-value)', fontsize=12)
ax2.set_title('Volcano Plot: Statistical Significance vs Effect Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(fig_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 2. FEATURE PATTERNS BY DILI SEVERITY ==========

# Figure 2: Feature patterns across DILI severity levels
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Select top 4 features
top_4_features = corr_df.head(4)['feature'].tolist()

for i, feature in enumerate(top_4_features):
    ax = axes[i]
    
    # Prepare data
    plot_data = merged_df[[feature, 'DILI_severity', 'drug']].dropna()
    
    # Box plot with points
    box_parts = ax.boxplot([plot_data[plot_data['DILI_severity'] == sev][feature].values 
                            for sev in sorted(plot_data['DILI_severity'].unique())],
                          positions=sorted(plot_data['DILI_severity'].unique()),
                          widths=0.6, patch_artist=True, showfliers=False)
    
    # Color boxes by severity
    colors_sev = ['green', 'yellow', 'orange', 'red']
    for patch, color in zip(box_parts['boxes'], colors_sev):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add individual points
    for sev in sorted(plot_data['DILI_severity'].unique()):
        sev_data = plot_data[plot_data['DILI_severity'] == sev]
        y = sev_data[feature].values
        x = np.random.normal(sev, 0.1, size=len(y))
        ax.scatter(x, y, alpha=0.8, s=50, color='black')
        
        # Label some points
        if len(sev_data) <= 5:
            for j, (_, row) in enumerate(sev_data.iterrows()):
                ax.annotate(row['drug'][:3], (x[j], y[j]), 
                           xytext=(2, 2), textcoords='offset points', fontsize=6)
    
    ax.set_xlabel('DILI Severity', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' '), fontsize=12)
    ax.set_title(f'{feature.replace("_", " ")}\nr = {corr_df[corr_df["feature"]==feature]["r"].iloc[0]:.3f}, p = {corr_df[corr_df["feature"]==feature]["p"].iloc[0]:.3f}',
                fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Top Features vs DILI Severity', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'feature_patterns_by_severity.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 3. MACHINE LEARNING MODELS ==========

print("\n🤖 Training machine learning models...")

# Prepare data for ML
X = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
y_binary = merged_df['DILI_binary'].values
y_severity = merged_df['DILI_severity'].values

# Remove features with zero variance
zero_var_mask = X.std() > 0
X = X.loc[:, zero_var_mask]
print(f"   Using {X.shape[1]} features after removing zero variance")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection - select top K features
k_features = min(20, X.shape[1])
selector = SelectKBest(f_classif, k=k_features)
X_selected = selector.fit_transform(X_scaled, y_binary)
selected_features = X.columns[selector.get_support()].tolist()
print(f"   Selected top {k_features} features based on ANOVA F-value")

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Figure 3: ROC curves and AUROC comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

results = {}

for i, (model_name, model) in enumerate(models.items()):
    ax = axes[i]
    
    # Get cross-validated predictions
    y_proba = cross_val_predict(model, X_selected, y_binary, cv=cv, method='predict_proba')
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_binary, y_proba[:, 1])
    auroc = roc_auc_score(y_binary, y_proba[:, 1])
    
    # Store results
    results[model_name] = {
        'auroc': auroc,
        'cv_scores': cross_val_score(model, X_selected, y_binary, cv=cv, scoring='roc_auc'),
        'y_proba': y_proba[:, 1]
    }
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name}\nAUROC = {auroc:.3f} (±{results[model_name]["cv_scores"].std():.3f})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('DILI Classification Performance: ROC Curves', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Model comparison and feature importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: AUROC comparison
model_names = list(results.keys())
aurocs = [results[m]['auroc'] for m in model_names]
auroc_stds = [results[m]['cv_scores'].std() for m in model_names]

bars = ax1.bar(range(len(model_names)), aurocs, yerr=auroc_stds, capsize=10, alpha=0.7)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.set_ylabel('AUROC', fontsize=12)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1)

# Add values on bars
for bar, auroc, std in zip(bars, aurocs, auroc_stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{auroc:.3f}\n±{std:.3f}', ha='center', va='bottom')

# Subplot 2: Feature importance from Random Forest
rf_model = models['Random Forest']
rf_model.fit(X_selected, y_binary)

# Get feature importances
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': importances
}).sort_values('importance', ascending=False).head(15)

bars = ax2.barh(range(len(importance_df)), importance_df['importance'], alpha=0.7)
ax2.set_yticks(range(len(importance_df)))
ax2.set_yticklabels([f.replace('_', ' ')[:30] for f in importance_df['feature']])
ax2.set_xlabel('Feature Importance', fontsize=12)
ax2.set_title('Top 15 Features: Random Forest Importance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 4. HEATMAP OF FEATURE CORRELATIONS ==========

# Figure 5: Feature correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Select top features for heatmap
top_features_for_heatmap = corr_df.head(15)['feature'].tolist()
heatmap_data = merged_df[top_features_for_heatmap + ['DILI_severity']].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(heatmap_data), k=1)

# Plot heatmap
sns.heatmap(heatmap_data, mask=mask, cmap='RdBu_r', center=0, 
            annot=True, fmt='.2f', square=True, linewidths=1,
            cbar_kws={'label': 'Correlation'})

ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 5. SUMMARY REPORT ==========

print("\n📊 Creating summary report...")

# Generate summary
summary_results = {
    'data_summary': {
        'n_drugs': len(merged_df),
        'n_features': len(feature_cols),
        'n_dili_positive': int(merged_df['DILI_binary'].sum()),
        'n_dili_negative': int((merged_df['DILI_binary'] == 0).sum())
    },
    'top_correlations': corr_df.head(10)[['feature', 'r', 'p']].to_dict('records'),
    'model_performance': {
        model: {
            'auroc': float(results[model]['auroc']),
            'auroc_std': float(results[model]['cv_scores'].std()),
            'cv_scores': results[model]['cv_scores'].tolist()
        }
        for model in results
    },
    'best_model': max(results.items(), key=lambda x: x[1]['auroc'])[0],
    'best_auroc': float(max(results.values(), key=lambda x: x['auroc'])['auroc']),
    'selected_features': selected_features,
    'feature_importance': importance_df.to_dict('records')
}

# Save results
with open(results_dir / 'event_normalized_comprehensive_results.json', 'w') as f:
    json.dump(summary_results, f, indent=2)

# Save predictions
predictions_df = merged_df[['drug', 'DILI_binary', 'DILI_severity']].copy()
for model_name, model_results in results.items():
    predictions_df[f'{model_name}_proba'] = model_results['y_proba']

predictions_df.to_csv(results_dir / 'event_normalized_predictions.csv', index=False)

print("\n✅ Comprehensive analysis complete!")
print(f"\n📊 SUMMARY:")
print(f"   Top correlation: {corr_df.iloc[0]['feature']} (r={corr_df.iloc[0]['r']:.3f})")
print(f"   Best model: {summary_results['best_model']} (AUROC={summary_results['best_auroc']:.3f})")
print(f"   Figures saved to: {fig_dir}")
print(f"   Results saved to: {results_dir}")