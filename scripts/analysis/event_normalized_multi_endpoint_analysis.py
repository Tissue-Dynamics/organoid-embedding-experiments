#!/usr/bin/env python3
"""
Comprehensive Event-Normalized DILI Multi-Endpoint Analysis

PURPOSE:
    Analyzes event-normalized features against multiple DILI endpoints:
    - DILI Severity (1-4)
    - DILI Binary (0/1)
    - DILI Likelihood (A, B, C, D, E, E*)
    
    Provides correlation analysis, visualizations, and ML models for each endpoint.
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

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "event_normalized_multi_endpoint"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("COMPREHENSIVE EVENT-NORMALIZED MULTI-ENDPOINT DILI ANALYSIS")
print("=" * 80)

# Load DILI data with all endpoints
def load_dili_data():
    """Load DILI classification data with all endpoints"""
    
    dili_data = {
        # Most DILI Concern (E*/E - severity 4)
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

# Get feature columns
exclude_cols = ['drug', 'n_wells', 'DILI_severity', 'DILI_binary', 'DILI_concern', 'DILI_likelihood']
feature_cols = [col for col in merged_df.columns if col not in exclude_cols]

# Create numeric encoding for DILI likelihood (A=1, B=2, C=3, D=4, E=5, E*=6)
likelihood_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'E*': 6}
merged_df['DILI_likelihood_numeric'] = merged_df['DILI_likelihood'].map(likelihood_mapping)

print(f"   DILI Binary: {merged_df['DILI_binary'].sum()} positive, {(merged_df['DILI_binary'] == 0).sum()} negative")
print(f"   DILI Likelihood distribution:")
for likelihood in ['A', 'B', 'C', 'D', 'E', 'E*']:
    count = (merged_df['DILI_likelihood'] == likelihood).sum()
    print(f"     {likelihood}: {count} drugs")

# ========== MULTI-ENDPOINT CORRELATION ANALYSIS ==========

print("\n📊 Calculating correlations across all DILI endpoints...")

# Define endpoints
endpoints = {
    'DILI_severity': {'name': 'DILI Severity (1-4)', 'type': 'continuous'},
    'DILI_binary': {'name': 'DILI Binary (0/1)', 'type': 'binary'},
    'DILI_likelihood_numeric': {'name': 'DILI Likelihood (A-E*)', 'type': 'ordinal'}
}

# Calculate correlations for each endpoint
all_correlations = {}
for endpoint, info in endpoints.items():
    correlations = []
    for feat in feature_cols:
        valid_data = merged_df[[feat, endpoint]].dropna()
        if len(valid_data) >= 5:
            r, p = pearsonr(valid_data[feat], valid_data[endpoint])
            correlations.append({
                'feature': feat,
                'r': r,
                'p': p,
                'abs_r': abs(r),
                'significant': p < 0.05,
                'endpoint': endpoint
            })
    
    all_correlations[endpoint] = pd.DataFrame(correlations).sort_values('abs_r', ascending=False)

# ========== VISUALIZATION 1: MULTI-ENDPOINT CORRELATION COMPARISON ==========

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for i, (endpoint, info) in enumerate(endpoints.items()):
    ax = axes[i]
    corr_df = all_correlations[endpoint]
    top_features = corr_df.head(15)
    
    colors = ['darkred' if row['significant'] and row['r'] > 0 else 'lightcoral' if row['r'] > 0 
              else 'darkblue' if row['significant'] else 'lightblue' 
              for _, row in top_features.iterrows()]
    
    bars = ax.barh(range(len(top_features)), top_features['r'], color=colors, alpha=0.8)
    
    # Add significance stars
    for j, (_, row) in enumerate(top_features.iterrows()):
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
            ax.text(x_pos, j, marker, va='center', ha='left' if row['r'] > 0 else 'right', fontsize=10)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace('_', ' ')[:25] for f in top_features['feature']], fontsize=10)
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_title(f'{info["name"]}\nTop r = {top_features.iloc[0]["r"]:.3f}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5)

plt.suptitle('Event-Normalized Features: Multi-Endpoint DILI Correlations', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'multi_endpoint_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== VISUALIZATION 2: LIKELIHOOD CATEGORY ANALYSIS ==========

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Select top 4 features from likelihood analysis
top_4_features = all_correlations['DILI_likelihood_numeric'].head(4)['feature'].tolist()

for i, feature in enumerate(top_4_features):
    ax = axes[i]
    
    # Prepare data for likelihood categories
    plot_data = merged_df[[feature, 'DILI_likelihood', 'drug']].dropna()
    
    # Order categories
    likelihood_order = ['A', 'B', 'C', 'D', 'E', 'E*']
    colors_likelihood = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
    
    # Box plot
    box_data = []
    positions = []
    box_colors = []
    
    for j, likelihood in enumerate(likelihood_order):
        data = plot_data[plot_data['DILI_likelihood'] == likelihood][feature].values
        if len(data) > 0:
            box_data.append(data)
            positions.append(j)
            box_colors.append(colors_likelihood[j])
    
    if box_data:
        box_parts = ax.boxplot(box_data, positions=positions, widths=0.6, 
                              patch_artist=True, showfliers=False)
        
        for patch, color in zip(box_parts['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for j, likelihood in enumerate(likelihood_order):
            if j in positions:
                sev_data = plot_data[plot_data['DILI_likelihood'] == likelihood]
                if len(sev_data) > 0:
                    y = sev_data[feature].values
                    x = np.random.normal(j, 0.05, size=len(y))
                    ax.scatter(x, y, alpha=0.8, s=50, color='black')
    
    ax.set_xticks(range(len(likelihood_order)))
    ax.set_xticklabels(likelihood_order)
    ax.set_xlabel('DILI Likelihood Category', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' '), fontsize=12)
    
    # Get correlation for this feature
    feat_corr = all_correlations['DILI_likelihood_numeric'][
        all_correlations['DILI_likelihood_numeric']['feature'] == feature
    ]['r'].iloc[0]
    feat_p = all_correlations['DILI_likelihood_numeric'][
        all_correlations['DILI_likelihood_numeric']['feature'] == feature
    ]['p'].iloc[0]
    
    ax.set_title(f'{feature.replace("_", " ")}\nr = {feat_corr:.3f}, p = {feat_p:.3f}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Top Features vs DILI Likelihood Categories', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'likelihood_category_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== MACHINE LEARNING MODELS FOR EACH ENDPOINT ==========

print("\n🤖 Training machine learning models for each endpoint...")

# Prepare data
X = merged_df[feature_cols].fillna(merged_df[feature_cols].median())
zero_var_mask = X.std() > 0
X = X.loc[:, zero_var_mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model results storage
ml_results = {}

# Binary Classification
print("   Training binary DILI classification models...")
y_binary = merged_df['DILI_binary'].values

# Feature selection for binary
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
    cv_scores = cross_val_score(model, X_binary, y_binary, cv=cv, scoring='roc_auc')
    y_proba = cross_val_predict(model, X_binary, y_binary, cv=cv, method='predict_proba')
    auroc = roc_auc_score(y_binary, y_proba[:, 1])
    
    binary_results[model_name] = {
        'auroc': auroc,
        'cv_scores': cv_scores,
        'mean_cv': cv_scores.mean(),
        'std_cv': cv_scores.std()
    }

ml_results['binary'] = binary_results

# Likelihood Classification (ordinal)
print("   Training DILI likelihood classification models...")
y_likelihood = merged_df['DILI_likelihood_numeric'].values

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
    cv_scores = cross_val_score(model, X_likelihood, y_likelihood, cv=5, scoring='r2')
    
    likelihood_results[model_name] = {
        'r2': cv_scores.mean(),
        'cv_scores': cv_scores,
        'std_cv': cv_scores.std()
    }

ml_results['likelihood'] = likelihood_results

# ========== VISUALIZATION 3: MODEL PERFORMANCE COMPARISON ==========

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Binary classification results
ax1 = axes[0]
model_names = list(binary_results.keys())
aurocs = [binary_results[m]['auroc'] for m in model_names]
auroc_stds = [binary_results[m]['std_cv'] for m in model_names]

bars = ax1.bar(range(len(model_names)), aurocs, yerr=auroc_stds, capsize=10, alpha=0.7, color='skyblue')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.set_ylabel('AUROC', fontsize=12)
ax1.set_title('Binary DILI Classification Performance', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1)

for bar, auroc, std in zip(bars, aurocs, auroc_stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{auroc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)

# Likelihood regression results
ax2 = axes[1]
model_names_lik = list(likelihood_results.keys())
r2s = [likelihood_results[m]['r2'] for m in model_names_lik]
r2_stds = [likelihood_results[m]['std_cv'] for m in model_names_lik]

bars = ax2.bar(range(len(model_names_lik)), r2s, yerr=r2_stds, capsize=10, alpha=0.7, color='lightcoral')
ax2.set_xticks(range(len(model_names_lik)))
ax2.set_xticklabels(model_names_lik, rotation=45, ha='right')
ax2.set_ylabel('R²', fontsize=12)
ax2.set_title('DILI Likelihood Prediction Performance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar, r2, std in zip(bars, r2s, r2_stds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{r2:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(fig_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== VISUALIZATION 4: CORRELATION HEATMAP ACROSS ENDPOINTS ==========

fig, ax = plt.subplots(figsize=(12, 8))

# Get top 10 features from each endpoint
top_features_set = set()
for endpoint in endpoints.keys():
    top_features_set.update(all_correlations[endpoint].head(10)['feature'].tolist())

# Create correlation matrix
correlation_matrix = []
for feature in list(top_features_set)[:15]:  # Limit to 15 for readability
    row = []
    for endpoint in endpoints.keys():
        feat_corr = all_correlations[endpoint][
            all_correlations[endpoint]['feature'] == feature
        ]
        if len(feat_corr) > 0:
            row.append(feat_corr['r'].iloc[0])
        else:
            row.append(0)
    correlation_matrix.append(row)

correlation_df = pd.DataFrame(correlation_matrix, 
                            index=[f.replace('_', ' ')[:25] for f in list(top_features_set)[:15]],
                            columns=[endpoints[e]['name'] for e in endpoints.keys()])

# Plot heatmap
sns.heatmap(correlation_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=False, linewidths=1, cbar_kws={'label': 'Pearson Correlation'})

ax.set_title('Feature Correlations Across DILI Endpoints', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'endpoint_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== SAVE COMPREHENSIVE RESULTS ==========

print("\n💾 Saving comprehensive results...")

# Summary statistics
summary_results = {
    'data_summary': {
        'n_drugs': len(merged_df),
        'n_features': len(feature_cols),
        'dili_distribution': {
            'binary': {
                'positive': int(merged_df['DILI_binary'].sum()),
                'negative': int((merged_df['DILI_binary'] == 0).sum())
            },
            'likelihood': {likelihood: int((merged_df['DILI_likelihood'] == likelihood).sum()) 
                          for likelihood in ['A', 'B', 'C', 'D', 'E', 'E*']}
        }
    },
    'correlations_by_endpoint': {
        endpoint: {
            'top_correlation': {
                'feature': all_correlations[endpoint].iloc[0]['feature'],
                'r': float(all_correlations[endpoint].iloc[0]['r']),
                'p': float(all_correlations[endpoint].iloc[0]['p'])
            },
            'top_10_features': all_correlations[endpoint].head(10)[['feature', 'r', 'p']].to_dict('records')
        }
        for endpoint in endpoints.keys()
    },
    'model_performance': {
        'binary_classification': {
            model: {
                'auroc': float(results['auroc']),
                'cv_mean': float(results['mean_cv']),
                'cv_std': float(results['std_cv'])
            }
            for model, results in binary_results.items()
        },
        'likelihood_regression': {
            model: {
                'r2': float(results['r2']),
                'cv_std': float(results['std_cv'])
            }
            for model, results in likelihood_results.items()
        }
    }
}

# Save to JSON
with open(results_dir / 'event_normalized_multi_endpoint_results.json', 'w') as f:
    json.dump(summary_results, f, indent=2)

# Save correlation tables
for endpoint in endpoints.keys():
    all_correlations[endpoint].to_csv(
        results_dir / f'event_normalized_correlations_{endpoint}.csv', 
        index=False
    )

print("\n✅ Multi-endpoint analysis complete!")
print(f"\n📊 SUMMARY RESULTS:")
print("\nTop correlations by endpoint:")
for endpoint, info in endpoints.items():
    top_corr = all_correlations[endpoint].iloc[0]
    print(f"   {info['name']}: {top_corr['feature']} (r={top_corr['r']:.3f}, p={top_corr['p']:.3f})")

print(f"\nBest model performance:")
best_binary = max(binary_results.items(), key=lambda x: x[1]['auroc'])
print(f"   Binary classification: {best_binary[0]} (AUROC={best_binary[1]['auroc']:.3f})")

best_likelihood = max(likelihood_results.items(), key=lambda x: x[1]['r2'])
print(f"   Likelihood prediction: {best_likelihood[0]} (R²={best_likelihood[1]['r2']:.3f})")

print(f"\nFigures saved to: {fig_dir}")
print(f"Results saved to: {results_dir}")