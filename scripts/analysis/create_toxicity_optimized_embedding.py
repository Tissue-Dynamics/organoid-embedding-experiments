#!/usr/bin/env python3
"""
Create Toxicity-Optimized Hybrid Embedding

Combines the most predictive components from each embedding method to create
a compact, interpretable embedding specifically optimized for toxicity prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_results():
    """Load hierarchical results and correlation analysis."""
    print("Loading results...")
    
    # Load hierarchical embeddings
    results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    hierarchical_results = joblib.load(results_file)
    
    # Load correlation results
    corr_file = project_root / "results" / "figures" / "drug_correlations" / "significant_drug_correlations.csv"
    correlation_results = pd.read_csv(corr_file)
    
    print(f"  Loaded {len(correlation_results)} significant correlations")
    
    return hierarchical_results, correlation_results


def select_top_toxicity_features(correlation_results, n_features=15):
    """Select top features for toxicity prediction."""
    print(f"Selecting top {n_features} toxicity features...")
    
    # Focus on DILI and hepatotoxicity
    toxicity_corrs = correlation_results[
        correlation_results['property'].isin(['binary_dili', 'hepatotoxicity_boxed_warning'])
    ].copy()
    
    # Sort by absolute correlation/effect size
    toxicity_corrs = toxicity_corrs.sort_values('abs_correlation', ascending=False)
    
    # Select top features, ensuring diversity across methods
    selected_features = []
    method_counts = {'sax': 0, 'tsfresh': 0, 'custom': 0, 'catch22': 0, 'fourier': 0}
    max_per_method = n_features // 3  # Distribute across methods
    
    for _, row in toxicity_corrs.iterrows():
        method = row['method']
        if len(selected_features) < n_features:
            if method_counts[method] < max_per_method or len(selected_features) > n_features * 0.8:
                selected_features.append(row)
                method_counts[method] += 1
    
    selected_df = pd.DataFrame(selected_features)
    
    print("Selected features:")
    for _, row in selected_df.iterrows():
        print(f"  {row['method']} {row['component']} ↔ {row['property']}: {row['correlation']:.3f}")
    
    return selected_df


def create_toxicity_embedding(hierarchical_results, selected_features):
    """Create toxicity-optimized embedding from selected features."""
    print("Creating toxicity embedding...")
    
    # Get drug metadata and embeddings
    drug_metadata = hierarchical_results['drug_metadata']
    drug_embeddings = hierarchical_results['drug_embeddings']
    drug_names = drug_metadata['drug'].tolist()
    
    # Extract selected features
    toxicity_features = []
    feature_names = []
    
    for _, row in selected_features.iterrows():
        method = row['method']
        component = row['component']
        
        # Extract component index (e.g., 'PC30' -> 29)
        component_idx = int(component.replace('PC', '')) - 1
        
        # Get the feature vector
        method_embeddings = drug_embeddings[method]
        feature_vector = method_embeddings[:, component_idx]
        
        toxicity_features.append(feature_vector)
        feature_names.append(f"{method}_{component}")
    
    # Combine into matrix
    toxicity_matrix = np.column_stack(toxicity_features)
    
    # Create DataFrame
    toxicity_embedding = pd.DataFrame(
        toxicity_matrix,
        index=drug_names,
        columns=feature_names
    )
    
    print(f"  Created toxicity embedding: {toxicity_embedding.shape}")
    
    return toxicity_embedding


def validate_toxicity_prediction(toxicity_embedding, drug_metadata):
    """Validate toxicity prediction performance."""
    print("Validating toxicity prediction...")
    
    # Prepare targets
    targets = {}
    if 'binary_dili' in drug_metadata.columns:
        targets['DILI'] = drug_metadata['binary_dili'].fillna(0).astype(int)
    if 'hepatotoxicity_boxed_warning' in drug_metadata.columns:
        targets['Hepatotoxicity'] = drug_metadata['hepatotoxicity_boxed_warning'].fillna(0).astype(int)
    
    results = {}
    
    for target_name, y in targets.items():
        print(f"\n  {target_name} prediction:")
        
        # Remove missing values
        valid_mask = y.notna()
        X_valid = toxicity_embedding.loc[valid_mask]
        y_valid = y.loc[valid_mask]
        
        if len(y_valid.unique()) < 2:
            print(f"    Skipping {target_name} - not enough classes")
            continue
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # CV scores
        cv_scores = cross_val_score(model, X_scaled, y_valid, cv=cv, scoring='roc_auc')
        print(f"    CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Fit final model for feature importance
        model.fit(X_scaled, y_valid)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': toxicity_embedding.columns,
            'importance': np.abs(model.coef_[0]),
            'coefficient': model.coef_[0]
        }).sort_values('importance', ascending=False)
        
        print(f"    Top 3 features:")
        for _, row in importance.head(3).iterrows():
            print(f"      {row['feature']}: {row['coefficient']:.3f}")
        
        results[target_name] = {
            'cv_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': importance,
            'model': model,
            'scaler': scaler
        }
    
    return results


def create_toxicity_embedding_visualization(toxicity_embedding, drug_metadata, selected_features, output_dir):
    """Create comprehensive visualization of the toxicity embedding."""
    print("Creating toxicity embedding visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Feature correlation matrix
    ax1 = plt.subplot(2, 3, 1)
    corr_matrix = toxicity_embedding.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, ax=ax1)
    ax1.set_title('Toxicity Feature Correlations')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # 2. Feature distribution by DILI status
    ax2 = plt.subplot(2, 3, 2)
    if 'binary_dili' in drug_metadata.columns:
        dili_status = drug_metadata['binary_dili'].fillna(0)
        
        # Plot first few features
        for i, feature in enumerate(toxicity_embedding.columns[:5]):
            for dili in [0, 1]:
                mask = dili_status == dili
                values = toxicity_embedding.loc[mask, feature]
                ax2.hist(values, alpha=0.6, bins=20, 
                        label=f'{feature}_DILI{dili}' if i < 2 else '', density=True)
        
        ax2.set_title('Feature Distributions by DILI Status')
        ax2.legend()
    
    # 3. Method contribution
    ax3 = plt.subplot(2, 3, 3)
    method_counts = selected_features['method'].value_counts()
    colors = {'sax': 'blue', 'tsfresh': 'orange', 'custom': 'green', 
              'catch22': 'red', 'fourier': 'purple'}
    bars = ax3.bar(method_counts.index, method_counts.values, 
                   color=[colors.get(m, 'gray') for m in method_counts.index])
    ax3.set_title('Selected Features by Method')
    ax3.set_ylabel('Number of Features')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. 2D embedding visualization (PCA)
    ax4 = plt.subplot(2, 3, 4)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(toxicity_embedding))
    
    if 'binary_dili' in drug_metadata.columns:
        dili_status = drug_metadata['binary_dili'].fillna(-1)
        scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=dili_status, 
                             cmap='RdYlBu', alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='DILI Risk')
    else:
        ax4.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    
    ax4.set_title(f'2D Toxicity Embedding (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    
    # 5. Top correlations
    ax5 = plt.subplot(2, 3, 5)
    top_correlations = selected_features.head(10)
    y_pos = np.arange(len(top_correlations))
    
    colors_map = {'sax': 'blue', 'tsfresh': 'orange', 'custom': 'green', 
                  'catch22': 'red', 'fourier': 'purple'}
    bar_colors = [colors_map.get(method, 'gray') for method in top_correlations['method']]
    
    bars = ax5.barh(y_pos, top_correlations['abs_correlation'], color=bar_colors, alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f"{row['method']} {row['component']}" for _, row in top_correlations.iterrows()])
    ax5.set_xlabel('Absolute Correlation/Effect Size')
    ax5.set_title('Top Toxicity Correlations')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Summary text
    summary_text = f"""
Toxicity Embedding Summary

Dimensions: {toxicity_embedding.shape[1]}
Drugs: {toxicity_embedding.shape[0]}

Method Distribution:
"""
    for method, count in method_counts.items():
        summary_text += f"  {method}: {count}\n"
    
    summary_text += f"\nTop Correlations:\n"
    for _, row in selected_features.head(3).iterrows():
        summary_text += f"  {row['method']} {row['component']}: {row['correlation']:.3f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'toxicity_embedding_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def save_toxicity_embedding_system(toxicity_embedding, selected_features, validation_results, output_dir):
    """Save the complete toxicity embedding system."""
    print("Saving toxicity embedding system...")
    
    # Save embedding matrix
    embedding_path = output_dir / 'toxicity_embedding.csv'
    toxicity_embedding.to_csv(embedding_path)
    print(f"  Saved embedding: {embedding_path}")
    
    # Save feature metadata
    features_path = output_dir / 'toxicity_features_metadata.csv'
    selected_features.to_csv(features_path, index=False)
    print(f"  Saved features: {features_path}")
    
    # Save complete system
    system_data = {
        'embedding': toxicity_embedding,
        'features_metadata': selected_features,
        'validation_results': validation_results,
        'feature_names': toxicity_embedding.columns.tolist(),
        'drug_names': toxicity_embedding.index.tolist()
    }
    
    system_path = output_dir / 'toxicity_embedding_system.joblib'
    joblib.dump(system_data, system_path)
    print(f"  Saved system: {system_path}")
    
    # Create README
    readme_content = f"""# Toxicity-Optimized Drug Embedding System

## Overview
This is a compact, interpretable embedding specifically optimized for drug toxicity prediction.
It combines the most predictive components from multiple embedding methods.

## Files
- `toxicity_embedding.csv`: The embedding matrix ({toxicity_embedding.shape[0]} drugs × {toxicity_embedding.shape[1]} features)
- `toxicity_features_metadata.csv`: Metadata about selected features and their correlations
- `toxicity_embedding_system.joblib`: Complete system for loading and using the embedding
- `toxicity_embedding_analysis.png`: Comprehensive visualization

## Performance Summary
"""
    
    for target, results in validation_results.items():
        readme_content += f"- {target} prediction: CV AUC = {results['cv_auc']:.3f} ± {results['cv_std']:.3f}\n"
    
    readme_content += f"""
## Feature Composition
"""
    method_counts = selected_features['method'].value_counts()
    for method, count in method_counts.items():
        readme_content += f"- {method}: {count} features\n"
    
    readme_content += f"""
## Usage Example
```python
import joblib
import pandas as pd

# Load the system
system = joblib.load('toxicity_embedding_system.joblib')
embedding = system['embedding']
metadata = system['features_metadata']

# Use for prediction
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# ... train model with embedding features
```

## Citation
Generated from organoid oxygen consumption time series using hierarchical embedding analysis.
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  Saved README: {readme_path}")


def main():
    """Main pipeline for creating toxicity-optimized embedding."""
    print("Creating Toxicity-Optimized Embedding System...")
    
    # Create output directory
    output_dir = project_root / "results" / "embeddings" / "toxicity_optimized"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    hierarchical_results, correlation_results = load_results()
    
    # Select top toxicity features
    selected_features = select_top_toxicity_features(correlation_results, n_features=15)
    
    # Create toxicity embedding
    toxicity_embedding = create_toxicity_embedding(hierarchical_results, selected_features)
    
    # Validate prediction performance
    validation_results = validate_toxicity_prediction(
        toxicity_embedding, 
        hierarchical_results['drug_metadata']
    )
    
    # Create visualizations
    create_toxicity_embedding_visualization(
        toxicity_embedding, 
        hierarchical_results['drug_metadata'], 
        selected_features, 
        output_dir
    )
    
    # Save complete system
    save_toxicity_embedding_system(
        toxicity_embedding, 
        selected_features, 
        validation_results, 
        output_dir
    )
    
    print(f"\n✅ Toxicity embedding system created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Dimensions: {toxicity_embedding.shape[1]} features × {toxicity_embedding.shape[0]} drugs")
    
    if validation_results:
        print("\n🔬 Validation Results:")
        for target, results in validation_results.items():
            print(f"   {target}: AUC = {results['cv_auc']:.3f} ± {results['cv_std']:.3f}")


if __name__ == "__main__":
    main()