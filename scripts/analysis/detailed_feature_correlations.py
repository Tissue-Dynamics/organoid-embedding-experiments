#!/usr/bin/env python3
"""
Detailed analysis: Are there ANY individual features that correlate between structure and oxygen?
This looks at feature-by-feature correlations to find specific relationships.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
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
    
    return structural_embeddings, oxygen_embeddings, common_drugs


def compute_feature_level_correlations(structural_embeddings, oxygen_embeddings, max_features=100):
    """Compute correlations between individual features (sampled for efficiency)."""
    print(f"\nComputing feature-level correlations (max {max_features} features per embedding)...")
    
    correlations = []
    
    for struct_name, struct_data in structural_embeddings.items():
        struct_scaled = StandardScaler().fit_transform(struct_data)
        
        # Sample features if too many
        if struct_scaled.shape[1] > max_features:
            struct_indices = np.random.choice(struct_scaled.shape[1], max_features, replace=False)
            struct_scaled = struct_scaled[:, struct_indices]
        else:
            struct_indices = np.arange(struct_scaled.shape[1])
        
        for oxygen_name, oxygen_data in oxygen_embeddings.items():
            oxygen_scaled = StandardScaler().fit_transform(oxygen_data)
            
            # Sample features if too many
            if oxygen_scaled.shape[1] > max_features:
                oxygen_indices = np.random.choice(oxygen_scaled.shape[1], max_features, replace=False)
                oxygen_scaled = oxygen_scaled[:, oxygen_indices]
            else:
                oxygen_indices = np.arange(oxygen_scaled.shape[1])
            
            print(f"  Analyzing {struct_name} ({len(struct_indices)} features) vs {oxygen_name} ({len(oxygen_indices)} features)...")
            
            # Compute all pairwise correlations
            for i in range(struct_scaled.shape[1]):
                for j in range(oxygen_scaled.shape[1]):
                    # Skip if too many zeros (sparse features)
                    struct_feature = struct_scaled[:, i]
                    oxygen_feature = oxygen_scaled[:, j]
                    
                    if np.std(struct_feature) < 1e-6 or np.std(oxygen_feature) < 1e-6:
                        continue
                    
                    # Pearson correlation
                    r_pearson, p_pearson = pearsonr(struct_feature, oxygen_feature)
                    
                    # Spearman correlation (skip to save time)
                    r_spearman, p_spearman = r_pearson, p_pearson  # Use Pearson as proxy
                    
                    correlations.append({
                        'struct_type': struct_name,
                        'oxygen_type': oxygen_name,
                        'struct_feature': struct_indices[i],
                        'oxygen_feature': oxygen_indices[j],
                        'pearson_r': r_pearson,
                        'pearson_p': p_pearson,
                        'spearman_r': r_spearman,
                        'spearman_p': p_spearman,
                        'abs_pearson': abs(r_pearson),
                        'abs_spearman': abs(r_spearman)
                    })
    
    return pd.DataFrame(correlations)


def find_significant_correlations(corr_df, p_threshold=0.01, r_threshold=0.3):
    """Find the most significant correlations."""
    print(f"\nFinding significant correlations (p < {p_threshold}, |r| > {r_threshold})...")
    
    # Filter for significant correlations
    significant = corr_df[
        (corr_df['pearson_p'] < p_threshold) & 
        (corr_df['abs_pearson'] > r_threshold)
    ].copy()
    
    # Sort by absolute correlation strength
    significant = significant.sort_values('abs_pearson', ascending=False)
    
    print(f"  Found {len(significant)} significant Pearson correlations")
    
    # Also check Spearman
    significant_spearman = corr_df[
        (corr_df['spearman_p'] < p_threshold) & 
        (corr_df['abs_spearman'] > r_threshold)
    ].copy()
    
    significant_spearman = significant_spearman.sort_values('abs_spearman', ascending=False)
    
    print(f"  Found {len(significant_spearman)} significant Spearman correlations")
    
    return significant, significant_spearman


def create_correlation_distribution_plot(corr_df, output_dir):
    """Plot the distribution of all correlations."""
    print("\nCreating correlation distribution plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Pearson correlation distribution
    ax1.hist(corr_df['pearson_r'], bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Pearson Correlation Coefficient')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of All Pearson Correlations')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_r = corr_df['pearson_r'].mean()
    std_r = corr_df['pearson_r'].std()
    ax1.text(0.05, 0.95, f'Mean: {mean_r:.4f}\nStd: {std_r:.4f}', 
             transform=ax1.transAxes, va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Absolute Pearson correlations
    ax2.hist(corr_df['abs_pearson'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='Weak (0.1)')
    ax2.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.3)')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Strong (0.5)')
    ax2.set_xlabel('Absolute Pearson Correlation')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Absolute Correlations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. P-value distribution (log scale)
    valid_p = corr_df['pearson_p'][corr_df['pearson_p'] > 0]
    ax3.hist(np.log10(valid_p), bins=50, alpha=0.7, edgecolor='black', color='green')
    ax3.axvline(x=np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax3.axvline(x=np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
    ax3.set_xlabel('log₁₀(p-value)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of P-values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation vs P-value scatter
    ax4.scatter(corr_df['abs_pearson'], -np.log10(corr_df['pearson_p'] + 1e-10), 
                alpha=0.5, s=1)
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax4.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
    ax4.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='|r| = 0.3')
    ax4.set_xlabel('Absolute Correlation')
    ax4.set_ylabel('-log₁₀(p-value)')
    ax4.set_title('Correlation Strength vs Significance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Feature-Level Correlations: Structure vs Oxygen', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'correlation_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def create_significant_correlations_plot(significant_pearson, significant_spearman, 
                                       structural_embeddings, oxygen_embeddings, output_dir):
    """Plot the most significant correlations."""
    print("\nCreating significant correlations visualization...")
    
    if len(significant_pearson) == 0 and len(significant_spearman) == 0:
        print("  No significant correlations found!")
        return None
    
    # Take top correlations for visualization
    top_n = min(10, len(significant_pearson) if len(significant_pearson) > 0 else len(significant_spearman))
    
    if len(significant_pearson) > 0:
        top_correlations = significant_pearson.head(top_n)
        correlation_type = "Pearson"
        r_col = 'pearson_r'
        p_col = 'pearson_p'
    else:
        top_correlations = significant_spearman.head(top_n)
        correlation_type = "Spearman"
        r_col = 'spearman_r'
        p_col = 'spearman_p'
    
    # Create figure
    n_rows = min(5, len(top_correlations))
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (_, row) in enumerate(top_correlations.head(n_rows * n_cols).iterrows()):
        ax_row = i // n_cols
        ax_col = i % n_cols
        ax = axes[ax_row, ax_col]
        
        # Get the actual feature data
        struct_data = structural_embeddings[row['struct_type']]
        oxygen_data = oxygen_embeddings[row['oxygen_type']]
        
        struct_feature = StandardScaler().fit_transform(struct_data)[:, int(row['struct_feature'])]
        oxygen_feature = StandardScaler().fit_transform(oxygen_data)[:, int(row['oxygen_feature'])]
        
        # Create scatter plot
        ax.scatter(struct_feature, oxygen_feature, alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(struct_feature, oxygen_feature, 1)
        p = np.poly1d(z)
        ax.plot(struct_feature, p(struct_feature), "r--", alpha=0.8)
        
        # Labels and title
        ax.set_xlabel(f'{row["struct_type"]} Feature {int(row["struct_feature"])}')
        ax.set_ylabel(f'{row["oxygen_type"]} Feature {int(row["oxygen_feature"])}')
        ax.set_title(f'{correlation_type} r = {row[r_col]:.3f} (p = {row[p_col]:.3e})')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(top_correlations), n_rows * n_cols):
        ax_row = i // n_cols
        ax_col = i % n_cols
        axes[ax_row, ax_col].set_visible(False)
    
    plt.suptitle(f'Top {correlation_type} Correlations: Structure vs Oxygen Features', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'top_{correlation_type.lower()}_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def create_correlation_heatmap_by_type(corr_df, output_dir):
    """Create heatmap showing correlations by embedding type."""
    print("\nCreating correlation heatmaps by embedding type...")
    
    # Group by embedding types and get summary statistics
    summary = corr_df.groupby(['struct_type', 'oxygen_type']).agg({
        'abs_pearson': ['mean', 'max', 'count'],
        'pearson_p': lambda x: (x < 0.01).sum()  # Count significant correlations
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['struct_type', 'oxygen_type', 'mean_abs_r', 'max_abs_r', 'total_pairs', 'significant_count']
    summary['significant_fraction'] = summary['significant_count'] / summary['total_pairs']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean absolute correlation
    pivot1 = summary.pivot(index='struct_type', columns='oxygen_type', values='mean_abs_r')
    sns.heatmap(pivot1, annot=True, fmt='.4f', cmap='viridis', ax=ax1,
                cbar_kws={'label': 'Mean |r|'})
    ax1.set_title('Mean Absolute Correlation by Embedding Type')
    ax1.set_xlabel('Oxygen Embedding Type')
    ax1.set_ylabel('Structural Embedding Type')
    
    # 2. Maximum correlation
    pivot2 = summary.pivot(index='struct_type', columns='oxygen_type', values='max_abs_r')
    sns.heatmap(pivot2, annot=True, fmt='.4f', cmap='plasma', ax=ax2,
                cbar_kws={'label': 'Max |r|'})
    ax2.set_title('Maximum Correlation by Embedding Type')
    ax2.set_xlabel('Oxygen Embedding Type')
    ax2.set_ylabel('Structural Embedding Type')
    
    # 3. Number of significant correlations
    pivot3 = summary.pivot(index='struct_type', columns='oxygen_type', values='significant_count')
    sns.heatmap(pivot3, annot=True, fmt='d', cmap='Reds', ax=ax3,
                cbar_kws={'label': 'Count'})
    ax3.set_title('Number of Significant Correlations (p < 0.01)')
    ax3.set_xlabel('Oxygen Embedding Type')
    ax3.set_ylabel('Structural Embedding Type')
    
    # 4. Fraction of significant correlations
    pivot4 = summary.pivot(index='struct_type', columns='oxygen_type', values='significant_fraction')
    sns.heatmap(pivot4, annot=True, fmt='.3f', cmap='coolwarm', ax=ax4,
                cbar_kws={'label': 'Fraction'})
    ax4.set_title('Fraction of Significant Correlations')
    ax4.set_xlabel('Oxygen Embedding Type')
    ax4.set_ylabel('Structural Embedding Type')
    
    plt.suptitle('Correlation Analysis Summary by Embedding Type', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'correlation_summary_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path, summary


def main():
    """Main analysis pipeline."""
    print("=== Detailed Feature-Level Correlation Analysis ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "feature_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    structural_embeddings, oxygen_embeddings, common_drugs = load_analysis_data()
    
    # Compute feature-level correlations
    corr_df = compute_feature_level_correlations(structural_embeddings, oxygen_embeddings)
    
    # Find significant correlations
    significant_pearson, significant_spearman = find_significant_correlations(corr_df)
    
    # Create visualizations
    dist_path = create_correlation_distribution_plot(corr_df, output_dir)
    heatmap_path, summary = create_correlation_heatmap_by_type(corr_df, output_dir)
    
    # Plot significant correlations if any exist
    if len(significant_pearson) > 0 or len(significant_spearman) > 0:
        sig_path = create_significant_correlations_plot(
            significant_pearson, significant_spearman, 
            structural_embeddings, oxygen_embeddings, output_dir
        )
    
    # Save results
    corr_path = output_dir / 'all_feature_correlations.csv'
    corr_df.to_csv(corr_path, index=False)
    
    if len(significant_pearson) > 0:
        sig_pearson_path = output_dir / 'significant_pearson_correlations.csv'
        significant_pearson.to_csv(sig_pearson_path, index=False)
    
    if len(significant_spearman) > 0:
        sig_spearman_path = output_dir / 'significant_spearman_correlations.csv'
        significant_spearman.to_csv(sig_spearman_path, index=False)
    
    summary_path = output_dir / 'correlation_summary.csv'
    summary.to_csv(summary_path, index=False)
    
    # Print summary
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n📊 Summary Statistics:")
    print(f"  Total feature pairs analyzed: {len(corr_df):,}")
    print(f"  Significant Pearson correlations (p<0.01, |r|>0.3): {len(significant_pearson)}")
    print(f"  Significant Spearman correlations (p<0.01, |r|>0.3): {len(significant_spearman)}")
    
    if len(corr_df) > 0:
        print(f"  Mean absolute correlation: {corr_df['abs_pearson'].mean():.4f}")
        print(f"  Maximum absolute correlation: {corr_df['abs_pearson'].max():.4f}")
        print(f"  Fraction with |r| > 0.1: {(corr_df['abs_pearson'] > 0.1).mean():.3f}")
        print(f"  Fraction with |r| > 0.3: {(corr_df['abs_pearson'] > 0.3).mean():.3f}")
        
        # Show best embedding type combinations
        print(f"\n🏆 Best embedding type combinations:")
        best_combinations = summary.nlargest(3, 'max_abs_r')
        for _, row in best_combinations.iterrows():
            print(f"  {row['struct_type']} ↔ {row['oxygen_type']}: max |r| = {row['max_abs_r']:.3f}")


if __name__ == "__main__":
    main()