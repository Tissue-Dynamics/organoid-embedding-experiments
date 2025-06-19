#!/usr/bin/env python3
"""
Drug Embedding Correlation Analysis

Correlates drug embeddings with drug properties including DILI risk, hepatotoxicity, 
pharmacokinetics, and chemical descriptors.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import duckdb
from dotenv import load_dotenv
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_drug_embeddings():
    """Load drug embeddings from hierarchical results file."""
    print("Loading drug embeddings...")
    
    results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    
    if not results_file.exists():
        print(f"  Error: Hierarchical results file not found at {results_file}")
        return {}
    
    try:
        results = joblib.load(results_file)
        print(f"  Loaded hierarchical results with keys: {list(results.keys())}")
        
        # Extract drug-level embeddings and metadata from the hierarchical structure
        drug_embeddings = {}
        drug_names = None
        
        # Get drug metadata to extract drug names
        if 'drug_metadata' in results:
            drug_meta = results['drug_metadata']
            if isinstance(drug_meta, pd.DataFrame) and 'drug' in drug_meta.columns:
                drug_names = drug_meta['drug'].tolist()
                print(f"  Found {len(drug_names)} drug names from metadata")
            else:
                print(f"  drug_metadata format unexpected: {type(drug_meta)}")
        
        # Check if there's a direct drug_embeddings key
        if 'drug_embeddings' in results:
            drug_data = results['drug_embeddings']
            print(f"  Found drug_embeddings with keys: {list(drug_data.keys()) if hasattr(drug_data, 'keys') else 'Not a dict'}")
            
            # If it's a dict with method keys
            if isinstance(drug_data, dict) and drug_names is not None:
                for method in ['fourier', 'sax', 'catch22', 'tsfresh', 'custom']:
                    if method in drug_data:
                        # Convert numpy array to DataFrame with drug names as index
                        embeddings_array = drug_data[method]
                        embeddings_df = pd.DataFrame(
                            embeddings_array, 
                            index=drug_names,
                            columns=[f'{method}_PC{i+1}' for i in range(embeddings_array.shape[1])]
                        )
                        drug_embeddings[method] = embeddings_df
                        print(f"  {method}: {embeddings_df.shape}")
            else:
                print(f"  drug_embeddings is not a dict or no drug names: {type(drug_data)}")
        else:
            print("  No drug_embeddings key found in results")
            print(f"  Available keys: {list(results.keys())}")
        
        return drug_embeddings
        
    except Exception as e:
        print(f"  Error loading hierarchical results: {e}")
        return {}


def load_drug_metadata():
    """Load drug metadata from database using DuckDB connection."""
    print("Loading drug metadata...")
    
    # Connect to database
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    attach_query = f"""
    ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
    user={parsed.username} password={parsed.password}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    conn.execute(attach_query)
    
    # Get drug metadata
    query = """
    SELECT 
        drug,
        dili,
        dili_risk_category,
        binary_dili,
        hepatotoxicity_boxed_warning,
        specific_toxicity_flags,
        c_max,
        half_life_hours,
        bioavailability_percent,
        logp,
        molecular_weight,
        smiles,
        atc,
        experimental_names
    FROM supabase.public.drugs
    WHERE drug IS NOT NULL
    """
    
    try:
        drugs_df = conn.execute(query).df()
        print(f"  Loaded metadata for {len(drugs_df)} drugs")
        print(f"  Columns: {list(drugs_df.columns)}")
        conn.close()
        return drugs_df
    except Exception as e:
        print(f"  Error loading drug metadata: {e}")
        conn.close()
        return pd.DataFrame()


def align_embeddings_with_metadata(drug_embeddings, drugs_df):
    """Align drug embeddings with metadata by drug names."""
    print("Aligning embeddings with metadata...")
    
    aligned_data = {}
    
    for method, embeddings_df in drug_embeddings.items():
        # Get common drugs between embeddings and metadata
        embedding_drugs = set(embeddings_df.index.str.lower())
        metadata_drugs = set(drugs_df['drug'].str.lower())
        
        print(f"  {method} debug:")
        print(f"    Sample embedding drugs: {list(embeddings_df.index[:5])}")
        print(f"    Sample metadata drugs: {list(drugs_df['drug'][:5])}")
        
        # Convert embedding index to lowercase for matching
        embeddings_df_lower = embeddings_df.copy()
        embeddings_df_lower.index = embeddings_df_lower.index.str.lower()
        
        # Find intersection
        common_drugs = embedding_drugs.intersection(metadata_drugs)
        print(f"    {len(common_drugs)} drugs with both embeddings and metadata")
        
        if len(common_drugs) > 0:
            # Filter embeddings and metadata to common drugs
            aligned_embeddings = embeddings_df_lower.loc[list(common_drugs)]
            aligned_metadata = drugs_df[drugs_df['drug'].str.lower().isin(common_drugs)].copy()
            aligned_metadata = aligned_metadata.set_index(aligned_metadata['drug'].str.lower())
            
            # Reorder to match
            aligned_metadata = aligned_metadata.loc[aligned_embeddings.index]
            
            aligned_data[method] = {
                'embeddings': aligned_embeddings,
                'metadata': aligned_metadata
            }
    
    return aligned_data


def calculate_correlations(embeddings, metadata, method_name):
    """Calculate correlations between embedding components and drug properties."""
    print(f"Calculating correlations for {method_name}...")
    
    # Prepare numeric metadata columns
    numeric_cols = []
    categorical_cols = []
    
    for col in metadata.columns:
        if col == 'drug_name':
            continue
            
        # Try to convert to numeric
        try:
            numeric_data = pd.to_numeric(metadata[col], errors='coerce')
            if numeric_data.notna().sum() > 10:  # At least 10 non-null values
                numeric_cols.append(col)
        except:
            pass
        
        # Check if categorical with reasonable number of categories
        if metadata[col].dtype == 'object' or metadata[col].dtype == 'bool':
            # Skip columns that contain arrays or other unhashable types
            try:
                n_categories = metadata[col].nunique()
                if 2 <= n_categories <= 10:  # Between 2-10 categories
                    categorical_cols.append(col)
            except (TypeError, ValueError):
                # Skip columns with unhashable types (arrays, etc.)
                continue
    
    print(f"  Numeric columns: {numeric_cols}")
    print(f"  Categorical columns: {categorical_cols}")
    
    # Calculate correlations for numeric columns
    numeric_correlations = {}
    for col in numeric_cols:
        y = pd.to_numeric(metadata[col], errors='coerce')
        valid_mask = y.notna()
        
        if valid_mask.sum() < 10:  # Skip if too few valid values
            continue
            
        correlations = []
        p_values = []
        
        for i in range(embeddings.shape[1]):
            x = embeddings.iloc[:, i][valid_mask]
            y_valid = y[valid_mask]
            
            if len(x) > 5:  # Need at least 5 points for correlation
                corr, p_val = stats.pearsonr(x, y_valid)
                correlations.append(corr)
                p_values.append(p_val)
            else:
                correlations.append(np.nan)
                p_values.append(np.nan)
        
        numeric_correlations[col] = {
            'correlations': np.array(correlations),
            'p_values': np.array(p_values),
            'n_samples': valid_mask.sum()
        }
    
    # Calculate effect sizes for categorical columns (using Cohen's d)
    categorical_effects = {}
    for col in categorical_cols:
        try:
            categories = metadata[col].dropna().unique()
            if len(categories) == 2:  # Binary categorical
                cat1_mask = metadata[col] == categories[0]
                cat2_mask = metadata[col] == categories[1]
                
                if cat1_mask.sum() >= 5 and cat2_mask.sum() >= 5:  # At least 5 in each group
                    effect_sizes = []
                    p_values = []
                    
                    for i in range(embeddings.shape[1]):
                        # Use boolean indexing properly
                        x1 = embeddings.loc[cat1_mask, embeddings.columns[i]]
                        x2 = embeddings.loc[cat2_mask, embeddings.columns[i]]
                        
                        # Cohen's d
                        pooled_std = np.sqrt(((len(x1) - 1) * x1.var() + (len(x2) - 1) * x2.var()) / (len(x1) + len(x2) - 2))
                        cohens_d = (x1.mean() - x2.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        # t-test p-value
                        _, p_val = stats.ttest_ind(x1, x2)
                        
                        effect_sizes.append(cohens_d)
                        p_values.append(p_val)
                    
                    categorical_effects[col] = {
                        'effect_sizes': np.array(effect_sizes),
                        'p_values': np.array(p_values),
                        'categories': categories,
                        'n_samples': [cat1_mask.sum(), cat2_mask.sum()]
                    }
        except Exception as e:
            print(f"    Warning: Error processing categorical column {col}: {e}")
            continue
    
    return {
        'numeric_correlations': numeric_correlations,
        'categorical_effects': categorical_effects,
        'embedding_shape': embeddings.shape,
        'metadata_shape': metadata.shape
    }


def create_correlation_heatmap(correlation_results, method_name, output_dir):
    """Create correlation heatmap for a specific embedding method."""
    print(f"Creating correlation heatmap for {method_name}...")
    
    # Prepare data for heatmap
    all_correlations = []
    all_labels = []
    all_p_values = []
    
    # Add numeric correlations
    for col, data in correlation_results['numeric_correlations'].items():
        all_correlations.append(data['correlations'])
        all_labels.append(f"{col} (r)")
        all_p_values.append(data['p_values'])
    
    # Add categorical effect sizes
    for col, data in correlation_results['categorical_effects'].items():
        all_correlations.append(data['effect_sizes'])
        all_labels.append(f"{col} (d)")
        all_p_values.append(data['p_values'])
    
    if len(all_correlations) == 0:
        print(f"  No correlations to plot for {method_name}")
        return
    
    # Create correlation matrix
    corr_matrix = np.array(all_correlations)
    p_matrix = np.array(all_p_values)
    
    # Create significance mask
    sig_mask = p_matrix < 0.05
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, corr_matrix.shape[1] * 0.3), max(6, len(all_labels) * 0.4)))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add significance markers
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if sig_mask[i, j] and not np.isnan(corr_matrix[i, j]):
                ax.text(j, i, '*', ha='center', va='center', color='black', fontsize=8, fontweight='bold')
    
    # Customize
    ax.set_xticks(range(corr_matrix.shape[1]))
    ax.set_xticklabels([f'PC{i+1}' for i in range(corr_matrix.shape[1])], rotation=45)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels)
    ax.set_title(f'{method_name.title()} Embedding Correlations with Drug Properties')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation / Effect Size')
    
    # Add legend for significance
    ax.text(0.02, 0.98, '* p < 0.05', transform=ax.transAxes, va='top', ha='left', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{method_name}_drug_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_summary_analysis(all_correlation_results, aligned_data, output_dir):
    """Create summary analysis across all embedding methods."""
    print("Creating summary analysis...")
    
    # Find most significant correlations across all methods
    significant_results = []
    
    for method, results in all_correlation_results.items():
        # Numeric correlations
        for col, data in results['numeric_correlations'].items():
            for i, (corr, p_val) in enumerate(zip(data['correlations'], data['p_values'])):
                if not np.isnan(corr) and p_val < 0.05:
                    significant_results.append({
                        'method': method,
                        'property': col,
                        'component': f'PC{i+1}',
                        'correlation': corr,
                        'p_value': p_val,
                        'type': 'correlation',
                        'n_samples': data['n_samples']
                    })
        
        # Categorical effects
        for col, data in results['categorical_effects'].items():
            for i, (effect, p_val) in enumerate(zip(data['effect_sizes'], data['p_values'])):
                if not np.isnan(effect) and p_val < 0.05:
                    significant_results.append({
                        'method': method,
                        'property': col,
                        'component': f'PC{i+1}',
                        'correlation': effect,
                        'p_value': p_val,
                        'type': 'effect_size',
                        'n_samples': sum(data['n_samples'])
                    })
    
    if len(significant_results) == 0:
        print("  No significant correlations found")
        return
    
    # Convert to DataFrame and sort by absolute correlation/effect size
    results_df = pd.DataFrame(significant_results)
    results_df['abs_correlation'] = results_df['correlation'].abs()
    results_df = results_df.sort_values('abs_correlation', ascending=False)
    
    # Save detailed results
    results_path = output_dir / 'significant_drug_correlations.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Saved detailed results: {results_path}")
    
    # Create summary visualization
    top_results = results_df.head(20)  # Top 20 most significant
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top correlations by method
    method_colors = {'fourier': 'blue', 'sax': 'green', 'catch22': 'red', 
                     'tsfresh': 'orange', 'custom': 'purple'}
    
    for i, (_, row) in enumerate(top_results.iterrows()):
        color = method_colors.get(row['method'], 'gray')
        ax1.barh(i, row['correlation'], color=color, alpha=0.7)
        ax1.text(row['correlation'] + 0.01 * np.sign(row['correlation']), i, 
                f"{row['property']} ({row['method']})", va='center', fontsize=8)
    
    ax1.set_yticks(range(len(top_results)))
    ax1.set_yticklabels([])
    ax1.set_xlabel('Correlation / Effect Size')
    ax1.set_title('Top Drug Property Correlations Across All Embedding Methods')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # Method comparison
    method_summary = results_df.groupby('method').agg({
        'abs_correlation': ['count', 'mean'],
        'p_value': 'mean'
    }).round(3)
    
    methods = method_summary.index
    counts = method_summary[('abs_correlation', 'count')]
    means = method_summary[('abs_correlation', 'mean')]
    
    x_pos = np.arange(len(methods))
    bars = ax2.bar(x_pos, counts, color=[method_colors.get(m, 'gray') for m in methods], alpha=0.7)
    
    # Add mean effect size as text on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'μ={mean_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Number of Significant Correlations')
    ax2.set_title('Embedding Method Comparison (p < 0.05)')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = output_dir / 'drug_correlation_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary plot: {summary_path}")
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"Total significant correlations found: {len(results_df)}")
    print(f"Methods ranked by number of significant correlations:")
    for method, count in results_df['method'].value_counts().items():
        mean_effect = results_df[results_df['method'] == method]['abs_correlation'].mean()
        print(f"  {method}: {count} correlations (mean |effect| = {mean_effect:.3f})")
    
    print(f"\nTop 5 strongest correlations:")
    for _, row in results_df.head().iterrows():
        print(f"  {row['method']} {row['component']} ↔ {row['property']}: "
              f"{row['correlation']:.3f} (p={row['p_value']:.3e})")


def main():
    """Main analysis pipeline."""
    print("Starting drug embedding correlation analysis...")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "drug_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    drug_embeddings = load_drug_embeddings()
    drugs_df = load_drug_metadata()
    
    if not drug_embeddings:
        print("No drug embeddings found. Please run hierarchical clustering analysis first.")
        return
    
    # Align embeddings with metadata
    aligned_data = align_embeddings_with_metadata(drug_embeddings, drugs_df)
    
    if not aligned_data:
        print("No common drugs found between embeddings and metadata.")
        return
    
    # Calculate correlations for each method
    all_correlation_results = {}
    for method, data in aligned_data.items():
        correlation_results = calculate_correlations(
            data['embeddings'], data['metadata'], method
        )
        all_correlation_results[method] = correlation_results
        
        # Create individual heatmap
        create_correlation_heatmap(correlation_results, method, output_dir)
    
    # Create summary analysis
    create_summary_analysis(all_correlation_results, aligned_data, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()