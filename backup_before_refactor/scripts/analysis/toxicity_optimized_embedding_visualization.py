#!/usr/bin/env python3
"""
Create embedding comparison visualization using toxicity-optimized features.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_toxicity_embedding():
    """Load the toxicity-optimized embedding system."""
    print("Loading toxicity-optimized embedding...")
    
    system_path = project_root / "results" / "embeddings" / "toxicity_optimized" / "toxicity_embedding_system.joblib"
    system_data = joblib.load(system_path)
    
    embedding = system_data['embedding']
    features_metadata = system_data['features_metadata']
    
    print(f"  Loaded embedding: {embedding.shape}")
    print(f"  Features: {list(embedding.columns)}")
    
    return embedding, features_metadata


def load_drug_metadata():
    """Load drug metadata for coloring."""
    print("Loading drug metadata...")
    
    hierarchical_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    results = joblib.load(hierarchical_file)
    drug_metadata = results['drug_metadata']
    
    print(f"  Loaded metadata for {len(drug_metadata)} drugs")
    
    return drug_metadata


def create_embedding_comparison_plot(embedding, drug_metadata, features_metadata, output_dir):
    """Create comprehensive embedding comparison plot using toxicity features."""
    print("Creating toxicity-optimized embedding comparison plot...")
    
    # Debug: Check data shapes and values
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding sample values: {embedding.iloc[:3, :3].values}")
    print(f"  Drug metadata shape: {drug_metadata.shape}")
    
    # Align drug metadata with embedding
    common_drugs = []
    embedding_aligned_indices = []
    metadata_aligned_indices = []
    
    for i, emb_drug in enumerate(embedding.index):
        # Try to find matching drug in metadata
        metadata_matches = drug_metadata[drug_metadata['drug'].str.lower() == emb_drug.lower()]
        if len(metadata_matches) > 0:
            common_drugs.append(emb_drug)
            embedding_aligned_indices.append(i)
            metadata_aligned_indices.append(metadata_matches.index[0])
    
    print(f"  Found {len(common_drugs)} common drugs for visualization")
    
    if len(common_drugs) < 10:
        print("  Warning: Very few common drugs found, using all embedding data")
        embedding_viz = embedding
        metadata_viz = None
    else:
        embedding_viz = embedding.iloc[embedding_aligned_indices]
        metadata_viz = drug_metadata.iloc[metadata_aligned_indices].reset_index(drop=True)
    
    # Standardize the embedding
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding_viz)
    
    # Create different 2D projections
    print("  Computing 2D projections...")
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    embedding_pca = pca.fit_transform(embedding_scaled)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embedding_viz)//4))
    embedding_tsne = tsne.fit_transform(embedding_scaled)
    
    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embedding_viz)//3))
    embedding_umap = umap_reducer.fit_transform(embedding_scaled)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Toxicity-Optimized Embedding Comparisons (15 Selected Features)', fontsize=16, fontweight='bold')
    
    # Prepare colors
    if metadata_viz is not None and 'binary_dili' in metadata_viz.columns:
        dili_colors = metadata_viz['binary_dili'].fillna(-1).astype(float)
        print(f"  DILI values: {dili_colors.value_counts()}")
    else:
        dili_colors = np.random.random(len(embedding_viz))  # Random colors as fallback
    
    if metadata_viz is not None and 'hepatotoxicity_boxed_warning' in metadata_viz.columns:
        hepato_colors = metadata_viz['hepatotoxicity_boxed_warning'].fillna(-1).astype(float)
        print(f"  Hepatotoxicity values: {hepato_colors.value_counts()}")
    else:
        hepato_colors = np.random.random(len(embedding_viz))  # Random colors as fallback
    
    # Top row: DILI risk
    # PCA - DILI
    scatter1 = axes[0, 0].scatter(embedding_pca[:, 0], embedding_pca[:, 1], 
                                 c=dili_colors, cmap='RdYlBu_r', alpha=0.7, s=50)
    axes[0, 0].set_title(f'PCA - DILI Risk\n(Explained Var: {pca.explained_variance_ratio_.sum():.2f})')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.colorbar(scatter1, ax=axes[0, 0], label='DILI Risk')
    
    # t-SNE - DILI
    scatter2 = axes[0, 1].scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], 
                                 c=dili_colors, cmap='RdYlBu_r', alpha=0.7, s=50)
    axes[0, 1].set_title('t-SNE - DILI Risk')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='DILI Risk')
    
    # UMAP - DILI
    scatter3 = axes[0, 2].scatter(embedding_umap[:, 0], embedding_umap[:, 1], 
                                 c=dili_colors, cmap='RdYlBu_r', alpha=0.7, s=50)
    axes[0, 2].set_title('UMAP - DILI Risk')
    axes[0, 2].set_xlabel('UMAP 1')
    axes[0, 2].set_ylabel('UMAP 2')
    plt.colorbar(scatter3, ax=axes[0, 2], label='DILI Risk')
    
    # Bottom row: Hepatotoxicity
    # PCA - Hepatotoxicity
    scatter4 = axes[1, 0].scatter(embedding_pca[:, 0], embedding_pca[:, 1], 
                                 c=hepato_colors, cmap='Reds', alpha=0.7, s=50)
    axes[1, 0].set_title('PCA - Hepatotoxicity Warning')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.colorbar(scatter4, ax=axes[1, 0], label='Hepatotoxicity')
    
    # t-SNE - Hepatotoxicity
    scatter5 = axes[1, 1].scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], 
                                 c=hepato_colors, cmap='Reds', alpha=0.7, s=50)
    axes[1, 1].set_title('t-SNE - Hepatotoxicity Warning')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter5, ax=axes[1, 1], label='Hepatotoxicity')
    
    # UMAP - Hepatotoxicity
    scatter6 = axes[1, 2].scatter(embedding_umap[:, 0], embedding_umap[:, 1], 
                                 c=hepato_colors, cmap='Reds', alpha=0.7, s=50)
    axes[1, 2].set_title('UMAP - Hepatotoxicity Warning')
    axes[1, 2].set_xlabel('UMAP 1')
    axes[1, 2].set_ylabel('UMAP 2')
    plt.colorbar(scatter6, ax=axes[1, 2], label='Hepatotoxicity')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'toxicity_optimized_embedding_comparisons.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return output_path


def create_feature_importance_plot(embedding, features_metadata, output_dir):
    """Create feature importance visualization."""
    print("Creating feature importance plot...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Feature importance by correlation strength
    features_sorted = features_metadata.sort_values('abs_correlation', ascending=True)
    
    # Color by method
    method_colors = {'sax': 'blue', 'tsfresh': 'orange', 'custom': 'green', 
                     'catch22': 'red', 'fourier': 'purple'}
    colors = [method_colors.get(method, 'gray') for method in features_sorted['method']]
    
    bars = ax1.barh(range(len(features_sorted)), features_sorted['abs_correlation'], 
                    color=colors, alpha=0.7)
    
    # Customize
    ax1.set_yticks(range(len(features_sorted)))
    ax1.set_yticklabels([f"{row['method']} {row['component']}" for _, row in features_sorted.iterrows()])
    ax1.set_xlabel('Absolute Correlation/Effect Size')
    ax1.set_title('Selected Features by Toxicity Correlation Strength')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, features_sorted['abs_correlation'])):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontsize=9)
    
    # Plot 2: Method contribution
    method_counts = features_metadata['method'].value_counts()
    pie_colors = [method_colors.get(method, 'gray') for method in method_counts.index]
    
    wedges, texts, autotexts = ax2.pie(method_counts.values, labels=method_counts.index, 
                                      colors=pie_colors, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Feature Distribution by Method')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'toxicity_features_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return output_path


def create_feature_correlation_heatmap(embedding, features_metadata, output_dir):
    """Create correlation heatmap of selected features."""
    print("Creating feature correlation heatmap...")
    
    # Compute correlation matrix
    corr_matrix = embedding.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                fmt='.2f', ax=ax)
    
    ax.set_title('Toxicity Features Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'toxicity_features_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return output_path


def main():
    """Main function."""
    print("Creating toxicity-optimized embedding visualizations...")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "toxicity_optimized_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    embedding, features_metadata = load_toxicity_embedding()
    drug_metadata = load_drug_metadata()
    
    # Create visualizations
    comparison_plot = create_embedding_comparison_plot(embedding, drug_metadata, features_metadata, output_dir)
    importance_plot = create_feature_importance_plot(embedding, features_metadata, output_dir)
    correlation_plot = create_feature_correlation_heatmap(embedding, features_metadata, output_dir)
    
    print(f"\n✅ Toxicity embedding visualizations complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Generated plots:")
    print(f"   - Embedding comparisons: {comparison_plot.name}")
    print(f"   - Feature importance: {importance_plot.name}")
    print(f"   - Feature correlations: {correlation_plot.name}")


if __name__ == "__main__":
    main()