#!/usr/bin/env python3
"""
Apply Louvain clustering to all embedding methods (Fourier, SAX, catch22, TSFresh, Custom).
Uses the same network-based approach for consistent high-resolution clustering.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from dotenv import load_dotenv
from urllib.parse import urlparse
import networkx as nx
import community as community_louvain
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_embeddings():
    """Load all embedding methods from hierarchical results."""
    print("Loading embeddings from hierarchical results...")
    
    results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    results = joblib.load(results_file)
    
    # Extract drug-level embeddings
    drug_embeddings = results['drug_embeddings']  # Dict of method -> embedding matrix
    drug_metadata = results['drug_metadata']  # DataFrame with drug info
    
    print(f"  Loaded embeddings for {len(drug_metadata)} drugs")
    print(f"  Available methods: {list(drug_embeddings.keys())}")
    
    return drug_embeddings, drug_metadata


def load_oxygen_time_series():
    """Load original oxygen time series data with proper exclusion filtering."""
    print("Loading oxygen time series data (excluding flagged data)...")
    
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
    
    # Get time series data with PROPER exclusion filtering
    query = """
    WITH drug_structure AS (
        SELECT 
            w.drug,
            w.concentration,
            w.well_number,
            w.plate_id,
            p.timestamp,
            p.median_o2,
            w.is_excluded as well_excluded,
            p.is_excluded as measurement_excluded,
            EXTRACT(EPOCH FROM (p.timestamp - MIN(p.timestamp) OVER (PARTITION BY w.drug))) / 3600.0 as hours_from_start
        FROM supabase.public.well_map_data w
        JOIN supabase.public.processed_data p
            ON w.plate_id = p.plate_id AND w.well_number = p.well_number
        WHERE w.drug != '' 
        AND w.drug IS NOT NULL
        AND w.is_excluded = false  -- Filter out excluded wells
        AND p.is_excluded = false  -- Filter out excluded measurements
        AND w.concentration >= 0
    )
    SELECT 
        drug,
        concentration,
        well_number,
        plate_id,
        hours_from_start,
        median_o2
    FROM drug_structure
    WHERE hours_from_start <= 350  -- ~2 weeks
    ORDER BY drug, concentration, well_number, hours_from_start
    """
    
    try:
        time_series_df = conn.execute(query).df()
        print(f"  Loaded {len(time_series_df):,} time series measurements (exclusions filtered)")
        
        # Check how many unique drugs we have
        n_drugs = time_series_df['drug'].nunique()
        print(f"  Unique drugs: {n_drugs}")
        
        conn.close()
        return time_series_df
    except Exception as e:
        print(f"  Error loading time series: {e}")
        conn.close()
        return pd.DataFrame()


def create_drug_network(embedding, n_neighbors=10, metric='euclidean'):
    """Create a drug similarity network from embeddings."""
    print(f"  Creating drug similarity network (k={n_neighbors})...")
    
    # Standardize embedding
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)
    
    # Create k-nearest neighbors graph
    A = kneighbors_graph(embedding_scaled, n_neighbors=n_neighbors, 
                         mode='distance', metric=metric, include_self=False)
    
    # Convert distances to similarities
    distances = A.data
    sigma = np.median(distances)
    similarities = np.exp(-distances**2 / (2 * sigma**2))
    A.data = similarities
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_array(A)
    
    # Add drug names as node attributes
    drug_mapping = {i: drug for i, drug in enumerate(embedding.index)}
    nx.relabel_nodes(G, drug_mapping, copy=False)
    
    print(f"    Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, scaler


def apply_louvain_clustering(G, resolution=1.0):
    """Apply Louvain community detection algorithm."""
    # Detect communities
    partition = community_louvain.best_partition(G, resolution=resolution)
    
    # Convert to DataFrame
    cluster_df = pd.DataFrame(list(partition.items()), columns=['drug', 'cluster'])
    cluster_df = cluster_df.sort_values('drug')
    
    # Get cluster statistics
    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    n_clusters = len(cluster_counts)
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G)
    
    return cluster_df, n_clusters, modularity


def aggregate_oxygen_curves(time_series_df, cluster_df):
    """Aggregate oxygen curves by cluster and concentration."""
    # Merge with cluster information
    time_series_with_clusters = time_series_df.merge(
        cluster_df[['drug', 'cluster']], 
        on='drug', 
        how='inner'
    )
    
    # Aggregate by cluster, concentration, and time
    aggregated = time_series_with_clusters.groupby([
        'cluster', 'concentration', 'hours_from_start'
    ]).agg({
        'median_o2': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['cluster', 'concentration', 'hours_from_start', 'o2_mean', 'o2_std', 'n_wells']
    
    # Only keep time points with sufficient data
    aggregated = aggregated[aggregated['n_wells'] >= 3]
    
    return aggregated


def create_method_visualization(method_name, embedding, cluster_df, aggregated_curves, 
                              n_clusters, modularity, output_dir):
    """Create visualization for a single embedding method."""
    print(f"  Creating visualization for {method_name}...")
    
    # Compute t-SNE
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)
    
    perplexity = min(30, len(embedding) // 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedding_tsne = tsne.fit_transform(embedding_scaled)
    
    # Add t-SNE coordinates to cluster_df
    cluster_df = cluster_df.copy()
    cluster_df['tsne_1'] = embedding_tsne[:, 0]
    cluster_df['tsne_2'] = embedding_tsne[:, 1]
    
    # Get clusters sorted by size
    cluster_sizes = cluster_df['cluster'].value_counts().sort_values(ascending=False)
    clusters_sorted = cluster_sizes.index.tolist()
    
    # Create figure (similar layout to original)
    fig = plt.figure(figsize=(20, 12))
    
    # Grid layout: 1 row for t-SNE, multiple rows for oxygen curves
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[1.2, 1, 1],
                         hspace=0.3, wspace=0.25)
    
    # 1. t-SNE plot (spans 2 columns)
    ax_tsne = fig.add_subplot(gs[0, :2])
    
    # Color palette
    n_colors = min(n_clusters, 20)
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_clusters))
    
    # Plot each community
    for i, cluster_id in enumerate(clusters_sorted[:20]):
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        color = colors[i % n_colors]
        
        ax_tsne.scatter(
            cluster_data['tsne_1'], 
            cluster_data['tsne_2'],
            c=[color], 
            label=f'C{cluster_id} (n={len(cluster_data)})',
            alpha=0.7, 
            s=50
        )
    
    ax_tsne.set_xlabel('t-SNE 1')
    ax_tsne.set_ylabel('t-SNE 2')
    ax_tsne.set_title(f'{method_name} Hierarchical Embeddings Analysis\n'
                     f'Louvain: {n_clusters} communities, Modularity: {modularity:.3f}')
    ax_tsne.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax_tsne.grid(True, alpha=0.3)
    
    # 2. Summary text (top right)
    ax_summary = fig.add_subplot(gs[0, 2:])
    ax_summary.axis('off')
    
    summary_text = f"{method_name} Louvain Clustering\n" + "="*30 + "\n"
    summary_text += f"Total drugs: {len(cluster_df)}\n"
    summary_text += f"Communities: {n_clusters}\n"
    summary_text += f"Modularity: {modularity:.3f}\n\n"
    summary_text += "Community sizes:\n"
    
    for i, (cluster_id, size) in enumerate(cluster_sizes.head(10).items()):
        summary_text += f"  C{cluster_id}: {size} drugs\n"
    if len(cluster_sizes) > 10:
        summary_text += f"  ... and {len(cluster_sizes)-10} more\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 3. Oxygen curves for top communities
    concentrations = sorted(aggregated_curves['concentration'].unique())
    conc_colors = plt.cm.viridis(np.linspace(0, 1, len(concentrations)))
    
    # Plot up to 8 communities
    max_communities = min(8, n_clusters)
    for plot_idx in range(max_communities):
        if plot_idx >= len(clusters_sorted):
            break
            
        cluster_id = clusters_sorted[plot_idx]
        
        # Position in grid (rows 1-2, 4 columns)
        row = (plot_idx // 4) + 1
        col = plot_idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        cluster_data = aggregated_curves[aggregated_curves['cluster'] == cluster_id]
        
        if len(cluster_data) > 0:
            # Plot curves for each concentration
            for i, conc in enumerate(concentrations):
                conc_data = cluster_data[cluster_data['concentration'] == conc]
                
                if len(conc_data) > 0:
                    conc_data = conc_data.sort_values('hours_from_start')
                    
                    color = conc_colors[i]
                    label = 'Control' if conc == 0 else f'{conc:.1e}'
                    
                    # Plot mean curve
                    ax.plot(conc_data['hours_from_start'], conc_data['o2_mean'], 
                           color=color, linewidth=1.5, label=label, alpha=0.8)
                    
                    # Add error bands
                    if len(conc_data) > 1:
                        std_err = conc_data['o2_std'] / np.sqrt(conc_data['n_wells'])
                        ax.fill_between(conc_data['hours_from_start'], 
                                       conc_data['o2_mean'] - std_err,
                                       conc_data['o2_mean'] + std_err,
                                       color=color, alpha=0.15)
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('O2 (%)')
        n_drugs_in_cluster = cluster_sizes[cluster_id]
        ax.set_title(f'Community {cluster_id} ({n_drugs_in_cluster} drugs)')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for first plot
        if plot_idx == 0:
            ax.legend(fontsize=7, ncol=2, loc='best')
        
        # Set appropriate axis limits
        ax.set_xlim(-10, 350)
        ax.set_ylim(-10, 100)
    
    # Save plot
    output_path = output_dir / f'{method_name.lower()}_louvain_clusters.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {output_path}")
    
    return output_path


def process_all_methods(drug_embeddings, drug_metadata, time_series_df, output_dir):
    """Process Louvain clustering for all embedding methods."""
    results = {}
    
    # Methods to process (skip tsfresh if not available)
    available_methods = list(drug_embeddings.keys())
    
    for method in available_methods:
        print(f"\nProcessing {method} embeddings...")
        
        try:
            # Get embedding matrix for this method
            embedding_matrix = drug_embeddings[method]
            
            if embedding_matrix is None:
                print(f"  No embeddings found for {method}")
                continue
            
            # Convert to DataFrame with drug names as index
            embedding_df = pd.DataFrame(
                embedding_matrix,
                index=drug_metadata['drug'].values
            )
            print(f"  Embedding shape: {embedding_df.shape}")
            
            # Create network and apply Louvain
            G, scaler = create_drug_network(embedding_df, n_neighbors=10)
            
            # Try different resolutions to find optimal
            best_resolution = 1.0
            best_modularity = -1
            best_n_clusters = 0
            
            for resolution in [0.8, 1.0, 1.2, 1.4, 1.6]:
                cluster_df, n_clusters, modularity = apply_louvain_clustering(G, resolution)
                
                # Prefer 8-15 clusters with good modularity
                if 8 <= n_clusters <= 15 and modularity > best_modularity:
                    best_resolution = resolution
                    best_modularity = modularity
                    best_n_clusters = n_clusters
            
            # Apply with best resolution
            cluster_df, n_clusters, modularity = apply_louvain_clustering(G, best_resolution)
            print(f"  Optimal: {n_clusters} communities (resolution={best_resolution:.1f}, modularity={modularity:.3f})")
            
            # Aggregate oxygen curves
            aggregated_curves = aggregate_oxygen_curves(time_series_df, cluster_df)
            
            # Create visualization
            viz_path = create_method_visualization(
                method.capitalize(), embedding_df, cluster_df, aggregated_curves,
                n_clusters, modularity, output_dir
            )
            
            # Save cluster assignments
            assignments_path = output_dir / f'{method}_louvain_assignments.csv'
            cluster_df.to_csv(assignments_path, index=False)
            
            results[method] = {
                'n_clusters': n_clusters,
                'modularity': modularity,
                'resolution': best_resolution,
                'viz_path': viz_path
            }
            
        except Exception as e:
            print(f"  Error processing {method}: {e}")
            continue
    
    return results


def main():
    """Main pipeline for Louvain clustering on all embeddings."""
    print("Applying Louvain Clustering to All Embedding Methods")
    print("=" * 60)
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "embedding_comparisons_louvain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    drug_embeddings, drug_metadata = load_embeddings()
    
    # Load time series data (with exclusions filtered)
    time_series_df = load_oxygen_time_series()
    
    if time_series_df.empty:
        print("Error: Could not load time series data")
        return
    
    # Process all methods
    results = process_all_methods(drug_embeddings, drug_metadata, time_series_df, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Louvain Clustering Results:")
    print("="*60)
    
    for method, info in results.items():
        print(f"\n{method.capitalize()}:")
        print(f"  Communities: {info['n_clusters']}")
        print(f"  Modularity: {info['modularity']:.3f}")
        print(f"  Resolution: {info['resolution']:.1f}")
    
    print(f"\n✅ Complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()