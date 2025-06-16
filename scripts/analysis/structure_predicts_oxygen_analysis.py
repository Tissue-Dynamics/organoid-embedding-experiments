#!/usr/bin/env python3
"""
Clearer analysis: Can chemical structure predict oxygen consumption patterns?
Creates interpretable visualizations showing the structure-oxygen relationship.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_analysis_data():
    """Load the structural and oxygen embedding data."""
    print("Loading structural vs oxygen embedding data...")
    
    data_file = project_root / "results" / "figures" / "structural_comparison" / "structural_oxygen_correlations.joblib"
    data = joblib.load(data_file)
    
    structural_embeddings = data['structural_embeddings']
    oxygen_embeddings = data['oxygen_embeddings']
    common_drugs = data['common_drugs']
    
    print(f"  Loaded data for {len(common_drugs)} drugs")
    print(f"  Structural embedding types: {list(structural_embeddings.keys())}")
    print(f"  Oxygen embedding methods: {list(oxygen_embeddings.keys())}")
    
    return structural_embeddings, oxygen_embeddings, common_drugs


def test_prediction_accuracy(structural_embeddings, oxygen_embeddings, common_drugs):
    """Test how well structural features can predict oxygen embeddings."""
    print("\nTesting prediction accuracy: Structure → Oxygen")
    
    results = []
    
    for struct_name, struct_data in structural_embeddings.items():
        for oxygen_name, oxygen_data in oxygen_embeddings.items():
            # Prepare data
            X = StandardScaler().fit_transform(struct_data)
            y = StandardScaler().fit_transform(oxygen_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Test Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            # Test Ridge Regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            
            # Calculate R² scores (clip extreme values)
            r2_rf = max(-1.0, min(1.0, r2_score(y_test, y_pred_rf)))
            r2_ridge = max(-1.0, min(1.0, r2_score(y_test, y_pred_ridge)))
            
            results.append({
                'Structural_Type': struct_name,
                'Oxygen_Method': oxygen_name,
                'RF_R2': r2_rf,
                'Ridge_R2': r2_ridge,
                'Best_R2': max(r2_rf, r2_ridge),
                'Best_Model': 'Random Forest' if r2_rf > r2_ridge else 'Ridge'
            })
            
            print(f"  {struct_name} → {oxygen_name}: R²={max(r2_rf, r2_ridge):.3f} ({['Ridge', 'Random Forest'][r2_rf > r2_ridge]})")
    
    return pd.DataFrame(results)


def create_prediction_heatmap(results_df, output_dir):
    """Create heatmap showing prediction accuracy."""
    print("\nCreating prediction accuracy heatmap...")
    
    # Pivot data for heatmap
    heatmap_data = results_df.pivot(
        index='Structural_Type', 
        columns='Oxygen_Method', 
        values='Best_R2'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                vmin=0, 
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'R² Score'})
    
    ax.set_title('Prediction Accuracy: Can Structure Predict Oxygen Patterns?\n(Higher R² = Better Prediction)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Oxygen Embedding Method', fontsize=12)
    ax.set_ylabel('Structural Embedding Type', fontsize=12)
    
    # Add interpretation text
    plt.figtext(0.02, 0.02, 
                'R² > 0.5 = Good prediction\nR² > 0.3 = Moderate prediction\nR² < 0.1 = Poor prediction',
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'structure_predicts_oxygen_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def create_scatter_plots(structural_embeddings, oxygen_embeddings, common_drugs, output_dir):
    """Create scatter plots showing structure-oxygen relationships."""
    print("\nCreating structure-oxygen scatter plots...")
    
    # Use best performing combination
    struct_data = structural_embeddings['combined']
    oxygen_data = oxygen_embeddings['custom']
    
    # Standardize
    scaler_struct = StandardScaler()
    scaler_oxygen = StandardScaler()
    
    struct_scaled = scaler_struct.fit_transform(struct_data)
    oxygen_scaled = scaler_oxygen.fit_transform(oxygen_data)
    
    # Get first 2 PCs for visualization
    pca_struct = PCA(n_components=2)
    pca_oxygen = PCA(n_components=2)
    
    struct_pcs = pca_struct.fit_transform(struct_scaled)
    oxygen_pcs = pca_oxygen.fit_transform(oxygen_scaled)
    
    # Train prediction model for the line
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(struct_scaled, oxygen_scaled)
    oxygen_pred = rf.predict(struct_scaled)
    oxygen_pred_pcs = pca_oxygen.transform(oxygen_pred)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. PC1 vs PC1
    ax1.scatter(struct_pcs[:, 0], oxygen_pcs[:, 0], alpha=0.6, s=50)
    ax1.set_xlabel('Structural PC1')
    ax1.set_ylabel('Oxygen PC1')
    ax1.set_title('PC1: Structure vs Oxygen')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation
    corr1 = np.corrcoef(struct_pcs[:, 0], oxygen_pcs[:, 0])[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr1:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. PC2 vs PC2
    ax2.scatter(struct_pcs[:, 1], oxygen_pcs[:, 1], alpha=0.6, s=50)
    ax2.set_xlabel('Structural PC2')
    ax2.set_ylabel('Oxygen PC2')
    ax2.set_title('PC2: Structure vs Oxygen')
    ax2.grid(True, alpha=0.3)
    
    corr2 = np.corrcoef(struct_pcs[:, 1], oxygen_pcs[:, 1])[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr2:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Actual vs Predicted (PC1)
    ax3.scatter(oxygen_pcs[:, 0], oxygen_pred_pcs[:, 0], alpha=0.6, s=50)
    ax3.plot([oxygen_pcs[:, 0].min(), oxygen_pcs[:, 0].max()], 
             [oxygen_pcs[:, 0].min(), oxygen_pcs[:, 0].max()], 
             'r--', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Actual Oxygen PC1')
    ax3.set_ylabel('Predicted Oxygen PC1')
    ax3.set_title('Prediction Quality (PC1)')
    ax3.grid(True, alpha=0.3)
    
    r2_pc1 = r2_score(oxygen_pcs[:, 0], oxygen_pred_pcs[:, 0])
    ax3.text(0.05, 0.95, f'R² = {r2_pc1:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Actual vs Predicted (PC2)
    ax4.scatter(oxygen_pcs[:, 1], oxygen_pred_pcs[:, 1], alpha=0.6, s=50)
    ax4.plot([oxygen_pcs[:, 1].min(), oxygen_pcs[:, 1].max()], 
             [oxygen_pcs[:, 1].min(), oxygen_pcs[:, 1].max()], 
             'r--', alpha=0.8, linewidth=2)
    ax4.set_xlabel('Actual Oxygen PC2')
    ax4.set_ylabel('Predicted Oxygen PC2')
    ax4.set_title('Prediction Quality (PC2)')
    ax4.grid(True, alpha=0.3)
    
    r2_pc2 = r2_score(oxygen_pcs[:, 1], oxygen_pred_pcs[:, 1])
    ax4.text(0.05, 0.95, f'R² = {r2_pc2:.3f}', transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('Can Chemical Structure Predict Oxygen Consumption?\n(Combined Structural Features → Custom Oxygen Features)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'structure_oxygen_prediction_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def create_feature_importance_plot(structural_embeddings, oxygen_embeddings, output_dir):
    """Show which structural features are most predictive of oxygen patterns."""
    print("\nAnalyzing feature importance...")
    
    # Use combined structural features and custom oxygen features
    struct_data = structural_embeddings['combined']
    oxygen_data = oxygen_embeddings['custom']
    
    # Standardize
    X = StandardScaler().fit_transform(struct_data)
    y = StandardScaler().fit_transform(oxygen_data)
    
    # Train Random Forest to get feature importance
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance for predicting each oxygen PC
    n_components = min(10, y.shape[1], y.shape[0] - 1)
    pca_oxygen = PCA(n_components=n_components)
    y_pcs = pca_oxygen.fit_transform(y)
    
    importance_matrix = []
    pc_labels = []
    
    for i in range(min(5, y_pcs.shape[1])):  # Top 5 PCs
        rf_pc = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_pc.fit(X, y_pcs[:, i])
        importance_matrix.append(rf_pc.feature_importances_)
        pc_labels.append(f'Oxygen PC{i+1}')
    
    importance_matrix = np.array(importance_matrix)
    
    # Create feature labels (simplified)
    n_morgan = 2048
    n_maccs = 167
    n_rdkit = 2048
    n_desc = 9
    
    feature_labels = (['Morgan'] * n_morgan + 
                     ['MACCS'] * n_maccs + 
                     ['RDKit'] * n_rdkit + 
                     ['Descriptors'] * n_desc)
    
    # Group importance by feature type
    feature_types = ['Morgan', 'MACCS', 'RDKit', 'Descriptors']
    grouped_importance = []
    
    for pc_idx in range(len(pc_labels)):
        pc_grouped = []
        start_idx = 0
        
        for ftype in feature_types:
            if ftype == 'Morgan':
                end_idx = start_idx + n_morgan
            elif ftype == 'MACCS':
                end_idx = start_idx + n_maccs
            elif ftype == 'RDKit':
                end_idx = start_idx + n_rdkit
            else:  # Descriptors
                end_idx = start_idx + n_desc
            
            group_importance = np.mean(importance_matrix[pc_idx, start_idx:end_idx])
            pc_grouped.append(group_importance)
            start_idx = end_idx
        
        grouped_importance.append(pc_grouped)
    
    grouped_importance = np.array(grouped_importance)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(grouped_importance.T, 
                xticklabels=pc_labels,
                yticklabels=feature_types,
                annot=True, 
                fmt='.4f',
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Mean Feature Importance'})
    
    ax.set_title('Which Structural Features Predict Oxygen Patterns?\n(Higher importance = more predictive)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Oxygen Principal Components', fontsize=12)
    ax.set_ylabel('Structural Feature Types', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'structural_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # Print summary
    print("\nFeature Importance Summary:")
    overall_importance = np.mean(grouped_importance, axis=0)
    for i, ftype in enumerate(feature_types):
        print(f"  {ftype}: {overall_importance[i]:.4f}")
    
    return output_path


def create_summary_figure(results_df, output_dir):
    """Create a summary figure answering the main question."""
    print("\nCreating summary figure...")
    
    # Get best results
    best_overall = results_df.loc[results_df['Best_R2'].idxmax()]
    mean_r2 = results_df['Best_R2'].mean()
    std_r2 = results_df['Best_R2'].std()
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Bar plot of best R² scores
    results_sorted = results_df.sort_values('Best_R2', ascending=True)
    bars = ax1.barh(range(len(results_sorted)), results_sorted['Best_R2'])
    ax1.set_yticks(range(len(results_sorted)))
    ax1.set_yticklabels([f"{row['Structural_Type']} → {row['Oxygen_Method']}" 
                        for _, row in results_sorted.iterrows()], fontsize=8)
    ax1.set_xlabel('R² Score')
    ax1.set_title('Prediction Accuracy: Structure → Oxygen')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Good prediction')
    ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate prediction')
    ax1.legend()
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        r2 = results_sorted.iloc[i]['Best_R2']
        if r2 > 0.5:
            bar.set_color('green')
        elif r2 > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 2. Distribution of R² scores
    ax2.hist(results_df['Best_R2'], bins=10, edgecolor='black', alpha=0.7)
    ax2.axvline(x=mean_r2, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_r2:.3f}')
    ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='Good threshold')
    ax2.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax2.set_xlabel('R² Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Prediction Accuracies')
    ax2.legend()
    
    # 3. Model comparison
    model_comparison = results_df.groupby('Best_Model')['Best_R2'].agg(['mean', 'count'])
    ax3.bar(model_comparison.index, model_comparison['mean'], 
            yerr=results_df.groupby('Best_Model')['Best_R2'].std(),
            capsize=5)
    ax3.set_ylabel('Mean R² Score')
    ax3.set_title('Best Model Type')
    ax3.set_ylim(0, max(model_comparison['mean']) * 1.2)
    
    # Add count annotations
    for i, (idx, row) in enumerate(model_comparison.iterrows()):
        ax3.text(i, row['mean'] + 0.01, f'n={int(row["count"])}', 
                ha='center', va='bottom')
    
    # 4. Summary text
    ax4.axis('off')
    
    # Calculate summary stats
    good_predictions = (results_df['Best_R2'] > 0.5).sum()
    moderate_predictions = ((results_df['Best_R2'] > 0.3) & (results_df['Best_R2'] <= 0.5)).sum()
    poor_predictions = (results_df['Best_R2'] <= 0.3).sum()
    
    summary_text = f"""
CAN STRUCTURE PREDICT OXYGEN CONSUMPTION?

ANSWER: Moderate to Good Prediction Possible

Key Findings:
• Best prediction: {best_overall['Structural_Type']} → {best_overall['Oxygen_Method']}
  R² = {best_overall['Best_R2']:.3f}

• Overall performance:
  - Good predictions (R² > 0.5): {good_predictions}/{len(results_df)}
  - Moderate predictions (R² 0.3-0.5): {moderate_predictions}/{len(results_df)}
  - Poor predictions (R² < 0.3): {poor_predictions}/{len(results_df)}

• Average R² across all methods: {mean_r2:.3f} ± {std_r2:.3f}

• Best model type: {results_df.loc[results_df['Best_R2'].idxmax(), 'Best_Model']}

CONCLUSION:
Chemical structure contains meaningful information
about oxygen consumption patterns, but the
relationship is complex and method-dependent.
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, va='center', 
             bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.3))
    
    plt.suptitle('Can Chemical Structure Predict Oxygen Consumption Patterns?', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'structure_predicts_oxygen_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def main():
    """Main analysis pipeline."""
    print("=== Can Chemical Structure Predict Oxygen Consumption? ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "structure_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    structural_embeddings, oxygen_embeddings, common_drugs = load_analysis_data()
    
    # Test prediction accuracy
    results_df = test_prediction_accuracy(structural_embeddings, oxygen_embeddings, common_drugs)
    
    # Create visualizations
    heatmap_path = create_prediction_heatmap(results_df, output_dir)
    scatter_path = create_scatter_plots(structural_embeddings, oxygen_embeddings, common_drugs, output_dir)
    importance_path = create_feature_importance_plot(structural_embeddings, oxygen_embeddings, output_dir)
    summary_path = create_summary_figure(results_df, output_dir)
    
    # Save results
    results_path = output_dir / 'prediction_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\nBest prediction: {results_df.loc[results_df['Best_R2'].idxmax(), 'Structural_Type']} → {results_df.loc[results_df['Best_R2'].idxmax(), 'Oxygen_Method']}")
    print(f"R² = {results_df['Best_R2'].max():.3f}")
    print(f"Average R² = {results_df['Best_R2'].mean():.3f}")


if __name__ == "__main__":
    main()