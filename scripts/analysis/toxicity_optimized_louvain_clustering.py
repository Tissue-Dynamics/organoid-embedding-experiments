#!/usr/bin/env python3
"""
Create cluster visualization using toxicity-optimized embeddings with Louvain community detection.

Uses Louvain algorithm for higher resolution clustering based on network structure.
Shows t-SNE of drugs and oxygen curves colored by concentration for each community.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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


def load_toxicity_embedding():
    """Load the toxicity-optimized embedding system."""
    print("Loading toxicity-optimized embedding...")
    
    system_path = project_root / "results" / "embeddings" / "toxicity_optimized" / "toxicity_embedding_system.joblib"
    system_data = joblib.load(system_path)
    
    embedding = system_data['embedding']
    features_metadata = system_data['features_metadata']
    drug_names = system_data['drug_names']
    
    print(f"  Loaded embedding: {embedding.shape}")
    print(f"  Drug names: {len(drug_names)}")
    
    return embedding, features_metadata, drug_names


def load_concentration_data():
    """Load concentration-level data for oxygen curves."""
    print("Loading concentration-level data...")
    
    # Load hierarchical results to get concentration data
    results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    results = joblib.load(results_file)
    
    concentration_metadata = results['concentration_metadata']
    print(f"  Loaded {len(concentration_metadata)} concentration records")
    
    return concentration_metadata


def load_oxygen_time_series():
    """Load original oxygen time series data."""
    print("Loading oxygen time series data...")
    
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
    
    # Get time series data with metadata
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
        AND w.is_excluded = false
        AND p.is_excluded = false
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
    WHERE hours_from_start <= 300
    ORDER BY drug, concentration, well_number, hours_from_start
    """
    
    try:
        time_series_df = conn.execute(query).df()
        print(f"  Loaded {len(time_series_df)} time series measurements")
        conn.close()
        return time_series_df
    except Exception as e:
        print(f"  Error loading time series: {e}")
        conn.close()
        return pd.DataFrame()


def create_drug_network(embedding, n_neighbors=10, metric='euclidean'):
    """Create a drug similarity network from embeddings."""
    print(f"Creating drug similarity network (k={n_neighbors})...")
    
    # Standardize embedding
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)
    
    # Create k-nearest neighbors graph
    A = kneighbors_graph(embedding_scaled, n_neighbors=n_neighbors, 
                         mode='distance', metric=metric, include_self=False)
    
    # Convert distances to similarities (using Gaussian kernel)
    # sigma is set to the median distance
    distances = A.data
    sigma = np.median(distances)
    similarities = np.exp(-distances**2 / (2 * sigma**2))
    A.data = similarities
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_array(A)
    
    # Add drug names as node attributes
    drug_mapping = {i: drug for i, drug in enumerate(embedding.index)}
    nx.relabel_nodes(G, drug_mapping, copy=False)
    
    print(f"  Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, scaler


def apply_louvain_clustering(G, resolution=1.0):
    """Apply Louvain community detection algorithm."""
    print(f"Applying Louvain clustering (resolution={resolution})...")
    
    # Detect communities
    partition = community_louvain.best_partition(G, resolution=resolution)
    
    # Convert to DataFrame
    cluster_df = pd.DataFrame(list(partition.items()), columns=['drug', 'cluster'])
    cluster_df = cluster_df.sort_values('drug')
    
    # Get cluster statistics
    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    n_clusters = len(cluster_counts)
    
    print(f"  Found {n_clusters} communities")
    print(f"  Community sizes: {cluster_counts.to_dict()}")
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G)
    print(f"  Modularity: {modularity:.3f}")
    
    return cluster_df, partition, modularity


def compute_tsne_projection(embedding, scaler):
    """Compute t-SNE projection of embeddings."""
    print("Computing t-SNE projection...")
    
    embedding_scaled = scaler.transform(embedding)
    
    # t-SNE with appropriate perplexity
    perplexity = min(30, len(embedding) // 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedding_tsne = tsne.fit_transform(embedding_scaled)
    
    return embedding_tsne


def find_optimal_resolution(G, embedding, scaler, resolution_range=np.arange(0.5, 3.0, 0.1)):
    """Find optimal resolution parameter for Louvain clustering."""
    print("Finding optimal resolution parameter...")
    
    embedding_scaled = scaler.transform(embedding)
    
    modularities = []
    silhouettes = []
    n_clusters_list = []
    
    for resolution in resolution_range:
        partition = community_louvain.best_partition(G, resolution=resolution)
        
        # Calculate metrics
        modularity = community_louvain.modularity(partition, G)
        modularities.append(modularity)
        
        # Convert partition to cluster labels
        cluster_labels = np.array([partition[drug] for drug in embedding.index])
        n_clusters = len(set(cluster_labels))
        n_clusters_list.append(n_clusters)
        
        # Calculate silhouette score if more than 1 cluster
        if n_clusters > 1 and n_clusters < len(embedding) - 1:
            sil_score = silhouette_score(embedding_scaled, cluster_labels)
            silhouettes.append(sil_score)
        else:
            silhouettes.append(0)
    
    # Find optimal based on combined criteria
    # Normalize scores
    mod_norm = (modularities - np.min(modularities)) / (np.max(modularities) - np.min(modularities))
    sil_norm = (silhouettes - np.min(silhouettes)) / (np.max(silhouettes) - np.min(silhouettes))
    
    # Combined score (weighted average)
    combined_scores = 0.7 * mod_norm + 0.3 * sil_norm
    optimal_idx = np.argmax(combined_scores)
    optimal_resolution = resolution_range[optimal_idx]
    
    print(f"  Optimal resolution: {optimal_resolution:.2f}")
    print(f"  Modularity: {modularities[optimal_idx]:.3f}")
    print(f"  Silhouette: {silhouettes[optimal_idx]:.3f}")
    print(f"  Number of clusters: {n_clusters_list[optimal_idx]}")
    
    return optimal_resolution, {
        'resolutions': resolution_range.tolist(),
        'modularities': modularities,
        'silhouettes': silhouettes,
        'n_clusters': n_clusters_list,
        'combined_scores': combined_scores.tolist()
    }


def aggregate_oxygen_curves_by_concentration(time_series_df, cluster_df):
    """Aggregate oxygen curves by drug cluster and concentration."""
    print("Aggregating oxygen curves by concentration...")
    
    # Merge with cluster information
    time_series_with_clusters = time_series_df.merge(
        cluster_df[['drug', 'cluster']], 
        on='drug', 
        how='inner'
    )
    
    print(f"  Matched {len(time_series_with_clusters)} measurements with clusters")
    
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
    
    print(f"  Generated {len(aggregated)} aggregated time points")
    
    return aggregated


def create_louvain_cluster_visualization(cluster_df, aggregated_curves, embedding_tsne, 
                                       modularity, output_dir, max_plots_per_row=4):
    """Create comprehensive Louvain cluster visualization."""
    print("Creating Louvain cluster visualization...")
    
    # Add t-SNE coordinates to cluster_df
    cluster_df['tsne_1'] = embedding_tsne[:, 0]
    cluster_df['tsne_2'] = embedding_tsne[:, 1]
    
    # Get clusters sorted by size
    cluster_sizes = cluster_df['cluster'].value_counts().sort_values(ascending=False)
    clusters_sorted = cluster_sizes.index.tolist()
    n_clusters = len(clusters_sorted)
    
    # Create figure layout
    n_rows = ((n_clusters - 1) // max_plots_per_row) + 2  # +1 for t-SNE, +1 for curves
    fig = plt.figure(figsize=(5 * max_plots_per_row, 4 * n_rows))
    
    # Create grid
    gs = fig.add_gridspec(nrows=n_rows, ncols=max_plots_per_row, 
                         height_ratios=[1.5] + [1] * (n_rows - 1),
                         hspace=0.3, wspace=0.3)
    
    # 1. t-SNE plot with communities (spans 2 columns)
    ax_tsne = fig.add_subplot(gs[0, :2])
    
    # Color palette for clusters
    n_colors = min(n_clusters, 20)  # Use up to 20 distinct colors
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_clusters))
    
    # Plot each community
    for i, cluster_id in enumerate(clusters_sorted[:20]):  # Show max 20 in legend
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        color = colors[i % n_colors]
        
        ax_tsne.scatter(
            cluster_data['tsne_1'], 
            cluster_data['tsne_2'],
            c=[color], 
            label=f'C{cluster_id} (n={len(cluster_data)})',
            alpha=0.7, 
            s=60
        )
    
    ax_tsne.set_xlabel('t-SNE 1')
    ax_tsne.set_ylabel('t-SNE 2')
    ax_tsne.set_title(f'Louvain Communities ({n_clusters} clusters)\nModularity: {modularity:.3f}')
    ax_tsne.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax_tsne.grid(True, alpha=0.3)
    
    # 2. Resolution analysis plot
    ax_resolution = fig.add_subplot(gs[0, 2:])
    ax_resolution.text(0.1, 0.9, f"Louvain Clustering Summary\n" + 
                      f"{'='*30}\n" +
                      f"Total communities: {n_clusters}\n" +
                      f"Modularity: {modularity:.3f}\n" +
                      f"Largest community: {cluster_sizes.iloc[0]} drugs\n" +
                      f"Smallest community: {cluster_sizes.iloc[-1]} drugs\n" +
                      f"Median community size: {cluster_sizes.median():.0f} drugs\n\n" +
                      f"Community size distribution:\n" +
                      f"  1-5 drugs: {(cluster_sizes <= 5).sum()} communities\n" +
                      f"  6-10 drugs: {((cluster_sizes > 5) & (cluster_sizes <= 10)).sum()} communities\n" +
                      f"  11-20 drugs: {((cluster_sizes > 10) & (cluster_sizes <= 20)).sum()} communities\n" +
                      f"  >20 drugs: {(cluster_sizes > 20).sum()} communities",
                      transform=ax_resolution.transAxes, fontsize=10, 
                      verticalalignment='top', fontfamily='monospace')
    ax_resolution.axis('off')
    
    # 3. Oxygen curves for largest communities
    # Get concentration range for colormap
    concentrations = sorted(aggregated_curves['concentration'].unique())
    conc_colors = plt.cm.viridis(np.linspace(0, 1, len(concentrations)))
    
    # Plot oxygen curves for top communities (up to 12)
    max_communities_to_plot = min(12, n_clusters)
    for plot_idx in range(max_communities_to_plot):
        cluster_id = clusters_sorted[plot_idx]
        
        row = (plot_idx // max_plots_per_row) + 1
        col = plot_idx % max_plots_per_row
        ax = fig.add_subplot(gs[row, col])
        
        cluster_data = aggregated_curves[aggregated_curves['cluster'] == cluster_id]
        
        if len(cluster_data) > 0:
            # Plot curves for each concentration
            for i, conc in enumerate(concentrations):
                conc_data = cluster_data[cluster_data['concentration'] == conc]
                
                if len(conc_data) > 0:
                    # Sort by time
                    conc_data = conc_data.sort_values('hours_from_start')
                    
                    color = conc_colors[i]
                    label = 'Control' if conc == 0 else f'{conc:.1e}'
                    
                    # Plot mean curve
                    ax.plot(conc_data['hours_from_start'], conc_data['o2_mean'], 
                           color=color, linewidth=2, label=label, alpha=0.8)
                    
                    # Add error bands
                    if len(conc_data) > 1:
                        std_err = conc_data['o2_std'] / np.sqrt(conc_data['n_wells'])
                        ax.fill_between(conc_data['hours_from_start'], 
                                       conc_data['o2_mean'] - std_err,
                                       conc_data['o2_mean'] + std_err,
                                       color=color, alpha=0.2)
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('O2 (%)')
        n_drugs_in_cluster = cluster_sizes[cluster_id]
        ax.set_title(f'Community {cluster_id} (n={n_drugs_in_cluster})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='best')
        
        # Set appropriate axis limits
        ax.set_xlim(-10, 350)
        ax.set_ylim(-10, 100)
    
    plt.suptitle('Louvain Community Detection on Toxicity-Optimized Embeddings', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    output_path = output_dir / 'louvain_cluster_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return output_path


def create_cluster_summary_table(cluster_df, aggregated_curves, output_dir):
    """Create summary table of cluster characteristics."""
    print("Creating cluster summary table...")
    
    summary_data = []
    
    for cluster_id in sorted(cluster_df['cluster'].unique()):
        cluster_drugs = cluster_df[cluster_df['cluster'] == cluster_id]['drug'].tolist()
        cluster_curves = aggregated_curves[aggregated_curves['cluster'] == cluster_id]
        
        # Basic stats
        n_drugs = len(cluster_drugs)
        
        # Oxygen pattern characteristics
        if len(cluster_curves) > 0:
            # Average final oxygen level
            final_o2_data = cluster_curves.groupby('concentration')['o2_mean'].apply(
                lambda x: x.iloc[-1] if len(x) > 0 else np.nan
            )
            avg_final_o2 = final_o2_data.mean()
            
            # Oxygen decline rate
            decline_rates = []
            for conc in cluster_curves['concentration'].unique():
                conc_data = cluster_curves[cluster_curves['concentration'] == conc].sort_values('hours_from_start')
                if len(conc_data) >= 2:
                    start_o2 = conc_data['o2_mean'].iloc[0]
                    end_o2 = conc_data['o2_mean'].iloc[-1]
                    time_span = conc_data['hours_from_start'].iloc[-1] - conc_data['hours_from_start'].iloc[0]
                    if time_span > 0:
                        decline_rates.append((end_o2 - start_o2) / time_span)
            
            avg_decline_rate = np.mean(decline_rates) if decline_rates else np.nan
        else:
            avg_final_o2 = np.nan
            avg_decline_rate = np.nan
        
        summary_data.append({
            'community': cluster_id,
            'n_drugs': n_drugs,
            'avg_final_o2': avg_final_o2,
            'avg_decline_rate': avg_decline_rate,
            'sample_drugs': ', '.join(cluster_drugs[:5]) + ('...' if len(cluster_drugs) > 5 else '')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('n_drugs', ascending=False)
    
    # Save summary
    summary_path = output_dir / 'louvain_cluster_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed drug assignments
    drug_assignments_path = output_dir / 'louvain_drug_assignments.csv'
    cluster_df[['drug', 'cluster', 'tsne_1', 'tsne_2']].to_csv(drug_assignments_path, index=False)
    
    print(f"  Saved summary: {summary_path}")
    print(f"  Saved assignments: {drug_assignments_path}")
    
    # Print top communities
    print("\nTop 10 Communities by Size:")
    for _, row in summary_df.head(10).iterrows():
        print(f"  Community {row['community']}: {row['n_drugs']} drugs, "
              f"final O2: {row['avg_final_o2']:.1f}%, "
              f"decline rate: {row['avg_decline_rate']:.3f}%/hr")
    
    return summary_df


def main():
    """Main pipeline for Louvain clustering visualization."""
    print("Creating Louvain Clustering Visualization...")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "louvain_clusters"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    embedding, features_metadata, drug_names = load_toxicity_embedding()
    concentration_metadata = load_concentration_data()
    time_series_df = load_oxygen_time_series()
    
    if time_series_df.empty:
        print("Error: Could not load time series data")
        return
    
    # Create drug similarity network
    G, scaler = create_drug_network(embedding, n_neighbors=10)
    
    # Find optimal resolution
    optimal_resolution, resolution_metrics = find_optimal_resolution(G, embedding, scaler)
    
    # Apply Louvain clustering with optimal resolution
    cluster_df, partition, modularity = apply_louvain_clustering(G, resolution=optimal_resolution)
    
    # Compute t-SNE projection
    embedding_tsne = compute_tsne_projection(embedding, scaler)
    
    # Aggregate oxygen curves by concentration
    aggregated_curves = aggregate_oxygen_curves_by_concentration(time_series_df, cluster_df)
    
    # Create visualization
    viz_path = create_louvain_cluster_visualization(
        cluster_df, aggregated_curves, embedding_tsne, modularity, output_dir
    )
    
    # Create summary
    summary_df = create_cluster_summary_table(cluster_df, aggregated_curves, output_dir)
    
    # Save resolution analysis
    resolution_analysis_path = output_dir / 'resolution_analysis.joblib'
    joblib.dump({
        'optimal_resolution': optimal_resolution,
        'metrics': resolution_metrics,
        'modularity': modularity,
        'network': G
    }, resolution_analysis_path)
    
    print(f"\n✅ Louvain clustering complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Main visualization: {viz_path.name}")
    print(f"📊 Number of communities: {cluster_df['cluster'].nunique()}")
    print(f"💊 Total drugs clustered: {len(cluster_df)}")
    print(f"🔍 Optimal resolution: {optimal_resolution:.2f}")
    print(f"📈 Modularity score: {modularity:.3f}")


if __name__ == "__main__":
    main()