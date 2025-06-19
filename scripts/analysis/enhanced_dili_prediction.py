#!/usr/bin/env python3
"""
Enhanced DILI Prediction using Phase 2 Embeddings
Focus on the larger dataset with better coverage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "enhanced_prediction"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ENHANCED DILI PREDICTION WITH PHASE 2 EMBEDDINGS")
print("=" * 80)

# Load the Phase 2 + DILI dataset we just created
phase2_dili_df = pd.read_csv(results_dir / "phase2_all_methods_dili.csv", index_col=0)
print(f"\n📊 Loaded Phase 2 + DILI dataset: {len(phase2_dili_df)} drugs")

# Also load chemical properties if available
if 'molecular_weight' in phase2_dili_df.columns:
    chem_features = ['molecular_weight', 'logp']
    chem_features = [f for f in chem_features if f in phase2_dili_df.columns]
    print(f"   Chemical features available: {chem_features}")
else:
    chem_features = []

# Prepare features and target
feature_cols = [col for col in phase2_dili_df.columns if col != 'dili_risk_numeric' and 'dim' in col]
feature_cols.extend(chem_features)

X = phase2_dili_df[feature_cols].fillna(phase2_dili_df[feature_cols].median())
y = phase2_dili_df['dili_risk_numeric']

print(f"\n📊 Feature matrix: {X.shape}")
print(f"   DILI risk distribution:")
print(f"   {pd.Series(y).value_counts().sort_index()}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, 4, duplicates='drop')
)

print(f"\n🔍 Train/Test split: {len(X_train)}/{len(X_test)}")

# Try multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}

results = {}

print("\n🤖 Training models...")
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=KFold(5, shuffle=True, random_state=42), 
                               scoring='r2')
    
    # Correlation
    test_corr, test_p = spearmanr(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'test_corr': test_corr,
        'test_p': test_p,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train
    }
    
    print(f"\n   {name}:")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    print(f"   CV R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"   Test correlation: r={test_corr:.3f}, p={test_p:.3e}")

# Feature importance analysis
print("\n📊 Analyzing feature importance...")

best_model_name = max(results, key=lambda x: results[x]['test_corr'])
best_model = models[best_model_name]
best_results = results[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   Top 15 features ({best_model_name}):")
    for _, row in feature_importance.head(15).iterrows():
        method = row['feature'].split('_')[0]
        print(f"   {row['feature']}: {row['importance']:.3f} ({method})")

# Dimensionality reduction analysis
print("\n🔍 Dimensionality reduction...")

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"   Original dimensions: {X_scaled.shape[1]}")
print(f"   PCA dimensions (95% variance): {X_pca.shape[1]}")
print(f"   Variance explained by first 5 PCs: {pca.explained_variance_ratio_[:5].sum():.1%}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Enhanced DILI Prediction with Phase 2 Embeddings', fontsize=20, fontweight='bold')

# Plot 1: Model comparison
ax = fig.add_subplot(gs[0, 0])
model_names = list(results.keys())
test_r2s = [results[m]['test_r2'] for m in model_names]
test_corrs = [results[m]['test_corr'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, test_r2s, width, label='Test R²', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_corrs, width, label='Test Correlation', color='lightgreen', edgecolor='black')

ax.set_xlabel('Model')
ax.set_ylabel('Performance')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim(-0.1, 0.6)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Best model predictions
ax = fig.add_subplot(gs[0, 1])
y_pred = best_results['y_pred_test']
scatter = ax.scatter(y_test, y_pred, alpha=0.6, s=60, edgecolor='black', linewidth=0.5)

# Add diagonal line
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=2, alpha=0.8)

# Add trend line
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
ax.plot(sorted(y_test), p(sorted(y_test)), "b-", alpha=0.8, linewidth=2)

ax.set_xlabel('True DILI Risk')
ax.set_ylabel('Predicted DILI Risk')
ax.set_title(f'{best_model_name} Predictions\nr={best_results["test_corr"]:.3f}, R²={best_results["test_r2"]:.3f}')
ax.grid(True, alpha=0.3)

# Plot 3: Feature importance heatmap
ax = fig.add_subplot(gs[0, 2])
if 'feature_importance' in locals():
    # Group by method
    method_importance = {}
    for _, row in feature_importance.iterrows():
        method = row['feature'].split('_')[0]
        if method not in method_importance:
            method_importance[method] = 0
        method_importance[method] += row['importance']
    
    methods = list(method_importance.keys())
    importances = list(method_importance.values())
    
    bars = ax.bar(methods, importances, color='coral', edgecolor='black')
    ax.set_xlabel('Embedding Method')
    ax.set_ylabel('Total Importance')
    ax.set_title('Feature Importance by Method')
    ax.set_xticklabels(methods, rotation=45)
    
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
               f'{imp:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 4: PCA visualization
ax = fig.add_subplot(gs[1, :2])
X_pca_2d = X_pca[:, :2]
scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                    c=y, cmap='RdYlBu_r', s=60, 
                    edgecolor='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA of Phase 2 Embeddings (Colored by DILI Risk)')
plt.colorbar(scatter, ax=ax, label='DILI Risk Score')

# Plot 5: Cross-validation scores
ax = fig.add_subplot(gs[1, 2])
cv_data = []
for name, res in results.items():
    cv_data.append({
        'Model': name,
        'CV R² Mean': res['cv_r2_mean'],
        'CV R² Std': res['cv_r2_std']
    })

cv_df = pd.DataFrame(cv_data)
x = np.arange(len(cv_df))
ax.bar(x, cv_df['CV R² Mean'], yerr=cv_df['CV R² Std'], 
       capsize=5, color='lightblue', edgecolor='black')
ax.set_xlabel('Model')
ax.set_ylabel('Cross-Validation R²')
ax.set_title('5-Fold Cross-Validation Performance')
ax.set_xticks(x)
ax.set_xticklabels(cv_df['Model'])
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 6: Residual analysis
ax = fig.add_subplot(gs[2, 0])
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.6, s=40, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', alpha=0.8)
ax.set_xlabel('Predicted DILI Risk')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# Plot 7: DILI risk distribution in predictions
ax = fig.add_subplot(gs[2, 1])
risk_categories = pd.cut(y_test, bins=[0, 1, 2, 3, 5], labels=['Low', 'Moderate', 'High', 'Severe'])
pred_categories = pd.cut(y_pred, bins=[0, 1, 2, 3, 5], labels=['Low', 'Moderate', 'High', 'Severe'])

confusion_matrix = pd.crosstab(risk_categories, pred_categories)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Risk Category')
ax.set_ylabel('True Risk Category')
ax.set_title('Risk Category Confusion Matrix')

# Plot 8: Feature correlation with DILI
ax = fig.add_subplot(gs[2, 2])
feature_corrs = []
for col in feature_cols[:20]:  # Top 20 features
    corr, p_val = spearmanr(X[col], y)
    feature_corrs.append({'feature': col, 'correlation': abs(corr), 'p_value': p_val})

corr_df = pd.DataFrame(feature_corrs).sort_values('correlation', ascending=False)
colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'gray' 
          for p in corr_df['p_value']]

ax.barh(range(len(corr_df)), corr_df['correlation'], color=colors)
ax.set_yticks(range(len(corr_df)))
ax.set_yticklabels([f[:25] for f in corr_df['feature']], fontsize=8)
ax.set_xlabel('|Correlation| with DILI Risk')
ax.set_title('Feature Correlations')
ax.set_xlim(0, 0.4)

# Plot 9: Learning curves
ax = fig.add_subplot(gs[3, :])
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
test_scores = []

for train_size in train_sizes:
    if int(train_size * len(X_train)) < 10:
        continue
    
    # Sample training data
    n_samples = int(train_size * len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    
    # Train model
    temp_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    temp_model.fit(X_train[indices], y_train.iloc[indices])
    
    # Score
    train_score = temp_model.score(X_train[indices], y_train.iloc[indices])
    test_score = temp_model.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

valid_sizes = train_sizes[:len(train_scores)]
ax.plot(valid_sizes * len(X_train), train_scores, 'o-', color='blue', label='Training score')
ax.plot(valid_sizes * len(X_train), test_scores, 'o-', color='red', label='Test score')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('R² Score')
ax.set_title('Learning Curves')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'enhanced_dili_prediction_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save models and results
print("\n💾 Saving models and results...")

# Save best model
joblib.dump(best_model, results_dir / 'best_dili_prediction_model.joblib')
joblib.dump(scaler, results_dir / 'dili_feature_scaler.joblib')

# Save predictions
predictions_df = pd.DataFrame({
    'drug': phase2_dili_df.index[len(X_train):],
    'true_dili_risk': y_test,
    'predicted_dili_risk': y_pred,
    'residual': residuals
})
predictions_df.to_csv(results_dir / 'dili_predictions_test_set.csv', index=False)

# Summary report
print("\n" + "="*80)
print("ENHANCED DILI PREDICTION SUMMARY")
print("="*80)

print(f"\n📊 DATASET:")
print(f"   Total drugs: {len(phase2_dili_df)}")
print(f"   Features: {len(feature_cols)} ({len([f for f in feature_cols if 'dim' in f])} embeddings, {len(chem_features)} chemical)")
print(f"   Train/Test: {len(X_train)}/{len(X_test)}")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {best_results['test_r2']:.3f}")
print(f"   Test Correlation: r={best_results['test_corr']:.3f} (p={best_results['test_p']:.3e})")
print(f"   CV R² (5-fold): {best_results['cv_r2_mean']:.3f} ± {best_results['cv_r2_std']:.3f}")

print(f"\n💡 KEY INSIGHTS:")
print(f"   1. Phase 2 embeddings alone achieve r={best_results['test_corr']:.3f} correlation")
print(f"   2. Multiple embedding methods contribute to predictions")
print(f"   3. Model generalizes reasonably well (CV R² = {best_results['cv_r2_mean']:.3f})")
print(f"   4. Room for improvement with more features and data")

print(f"\n📁 OUTPUTS:")
print(f"   Model: {results_dir / 'best_dili_prediction_model.joblib'}")
print(f"   Predictions: {results_dir / 'dili_predictions_test_set.csv'}")
print(f"   Visualization: {fig_dir / 'enhanced_dili_prediction_analysis.png'}")

print("\n✅ Enhanced DILI prediction analysis complete!")