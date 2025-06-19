#!/usr/bin/env python3
"""Complete embedding method comparison for organoid time series."""

import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import duckdb
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from embeddings import (
    DTWEmbedder, FourierEmbedder, SAXEmbedder,
    TSFreshEmbedder, Catch22Embedder, CustomFeaturesEmbedder
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingComparison:
    """Compare all embedding methods on organoid data."""
    
    def __init__(self):
        self.conn = self._connect_db()
        self.embedders = self._initialize_embedders()
        self.results = {}
        
    def _connect_db(self):
        """Connect to database."""
        import os
        from dotenv import load_dotenv
        from urllib.parse import urlparse
        
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
        return conn
    
    def _initialize_embedders(self):
        """Initialize all embedding methods."""
        return {
            'dtw': DTWEmbedder(n_components=10),
            'fourier': FourierEmbedder(n_components=30),
            'sax': SAXEmbedder(n_segments=20, n_symbols=5),
            'tsfresh': TSFreshEmbedder(n_jobs=4),
            'catch22': Catch22Embedder(),
            'custom': CustomFeaturesEmbedder()
        }
    
    def load_test_data(self, drug_name: str = 'Sanofi-1', max_series: int = 50):
        """Load test dataset with control normalization option."""
        logger.info(f"Loading test data for {drug_name}...")
        
        # Get time series for drug
        query = f"""
        WITH drug_series AS (
            SELECT 
                w.plate_id,
                w.well_number,
                w.concentration,
                w.drug,
                p.timestamp,
                p.median_o2
            FROM supabase.public.well_map_data w
            JOIN supabase.public.processed_data p
                ON w.plate_id = p.plate_id AND w.well_number = p.well_number
            WHERE w.drug = '{drug_name}'
            AND w.is_excluded = false
            AND p.is_excluded = false
            AND w.concentration > 0
            ORDER BY w.concentration, w.well_number, p.timestamp
        ),
        control_series AS (
            SELECT 
                w.plate_id,
                p.timestamp,
                AVG(p.median_o2) as control_o2
            FROM supabase.public.well_map_data w
            JOIN supabase.public.processed_data p
                ON w.plate_id = p.plate_id AND w.well_number = p.well_number
            WHERE w.plate_id IN (SELECT DISTINCT plate_id FROM drug_series)
            AND (w.drug = '' OR w.drug IS NULL)
            AND p.is_excluded = false
            GROUP BY w.plate_id, p.timestamp
        )
        SELECT 
            d.*,
            c.control_o2,
            d.median_o2 - COALESCE(c.control_o2, d.median_o2) as normalized_o2
        FROM drug_series d
        LEFT JOIN control_series c
            ON d.plate_id = c.plate_id AND d.timestamp = c.timestamp
        """
        
        df = self.conn.execute(query).df()
        
        if len(df) == 0:
            logger.warning(f"No data found for {drug_name}")
            return None, None
            
        # Prepare time series matrix
        series_dict = {}
        metadata = []
        
        for (well, conc), group in df.groupby(['well_number', 'concentration']):
            if len(series_dict) >= max_series:
                break
                
            # Sort by time and get values
            group = group.sort_values('timestamp')
            
            # Use normalized values if available, else raw
            if 'control_o2' in group.columns and group['control_o2'].notna().any():
                values = group['normalized_o2'].values
                logger.info(f"Using control-normalized data for well {well}")
            else:
                values = group['median_o2'].values
                
            if len(values) > 50:  # Minimum length
                series_dict[f"{well}_{conc}"] = values
                metadata.append({
                    'well': well,
                    'concentration': conc,
                    'plate_id': group['plate_id'].iloc[0]
                })
        
        # Convert to matrix (pad to same length)
        max_len = max(len(ts) for ts in series_dict.values())
        X = np.zeros((len(series_dict), max_len))
        
        for i, (key, ts) in enumerate(series_dict.items()):
            X[i, :len(ts)] = ts
            
        logger.info(f"Loaded {X.shape[0]} time series of length {X.shape[1]}")
        
        return X, pd.DataFrame(metadata)
    
    def compare_embeddings(self, X: np.ndarray, metadata: pd.DataFrame):
        """Run all embedding methods and compare."""
        
        results = {}
        
        for name, embedder in self.embedders.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {name.upper()} embedder...")
            
            try:
                # Time the embedding
                start_time = time.time()
                
                if name == 'dtw':
                    # DTW needs special handling - compute distance matrix first
                    from tslearn.metrics import cdist_dtw
                    logger.info("Computing DTW distance matrix...")
                    dist_matrix = cdist_dtw(X)
                    
                    # Use MDS for embedding
                    from sklearn.manifold import MDS
                    mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42)
                    embeddings = mds.fit_transform(dist_matrix)
                    
                elif name == 'tsfresh':
                    # TSFresh needs DataFrame format
                    from tsfresh import extract_features
                    from tsfresh.utilities.dataframe_functions import impute
                    
                    # Prepare data
                    dfs = []
                    for i in range(X.shape[0]):
                        df = pd.DataFrame({
                            'id': i,
                            'time': range(X.shape[1]),
                            'value': X[i, :]
                        })
                        dfs.append(df)
                    
                    ts_df = pd.concat(dfs)
                    
                    # Extract features
                    features = extract_features(ts_df, column_id='id', column_sort='time')
                    features = impute(features)
                    
                    # Reduce dimensionality
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    pca = PCA(n_components=min(50, features.shape[1]))
                    embeddings = pca.fit_transform(features_scaled)
                    
                else:
                    # Standard embedders
                    embeddings = embedder.fit_transform(X)
                
                embed_time = time.time() - start_time
                
                # Evaluate quality
                logger.info(f"Embedding shape: {embeddings.shape}")
                logger.info(f"Time taken: {embed_time:.2f} seconds")
                
                # Clustering metrics
                if embeddings.shape[0] > 10:  # Need enough samples
                    # Try different numbers of clusters
                    best_silhouette = -1
                    best_k = 2
                    
                    for k in range(2, min(8, embeddings.shape[0] // 2)):
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        labels = kmeans.fit_predict(embeddings)
                        
                        sil_score = silhouette_score(embeddings, labels)
                        if sil_score > best_silhouette:
                            best_silhouette = sil_score
                            best_k = k
                    
                    # Final clustering with best k
                    kmeans = KMeans(n_clusters=best_k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    
                    # Metrics
                    silhouette = silhouette_score(embeddings, labels)
                    davies_bouldin = davies_bouldin_score(embeddings, labels)
                    calinski = calinski_harabasz_score(embeddings, labels)
                    
                    # Check replicate consistency
                    # Group by concentration and see if replicates cluster together
                    replicate_scores = []
                    for conc in metadata['concentration'].unique():
                        conc_mask = metadata['concentration'] == conc
                        if conc_mask.sum() > 1:
                            conc_labels = labels[conc_mask]
                            # Proportion of replicates in same cluster
                            most_common = pd.Series(conc_labels).mode()[0]
                            consistency = (conc_labels == most_common).mean()
                            replicate_scores.append(consistency)
                    
                    replicate_consistency = np.mean(replicate_scores) if replicate_scores else 0
                    
                else:
                    silhouette = davies_bouldin = calinski = replicate_consistency = np.nan
                
                results[name] = {
                    'embeddings': embeddings,
                    'time': embed_time,
                    'dimensions': embeddings.shape[1],
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'calinski_harabasz_score': calinski,
                    'replicate_consistency': replicate_consistency,
                    'best_k': best_k if 'best_k' in locals() else None
                }
                
                logger.info(f"Silhouette Score: {silhouette:.3f}")
                logger.info(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
                logger.info(f"Calinski-Harabasz Score: {calinski:.1f}")
                logger.info(f"Replicate Consistency: {replicate_consistency:.2%}")
                
            except Exception as e:
                logger.error(f"Failed to run {name}: {e}")
                import traceback
                traceback.print_exc()
                
        return results
    
    def visualize_comparison(self, results: Dict, metadata: pd.DataFrame):
        """Create comprehensive visualization of results."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Embedding visualizations (2D projections)
        n_methods = len(results)
        for i, (name, res) in enumerate(results.items()):
            ax = plt.subplot(3, n_methods, i + 1)
            
            embeddings = res['embeddings']
            
            # Project to 2D if needed
            if embeddings.shape[1] > 2:
                pca = PCA(n_components=2)
                embed_2d = pca.fit_transform(embeddings)
            else:
                embed_2d = embeddings
            
            # Color by concentration
            concentrations = metadata['concentration'].values
            scatter = ax.scatter(embed_2d[:, 0], embed_2d[:, 1], 
                               c=concentrations, cmap='viridis', 
                               alpha=0.7, s=100)
            
            ax.set_title(f'{name.upper()}\nTime: {res["time"]:.1f}s')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            
            if i == n_methods - 1:
                plt.colorbar(scatter, ax=ax, label='Concentration')
        
        # 2. Metrics comparison
        ax = plt.subplot(3, 1, 2)
        
        # Prepare metrics data
        metrics_data = []
        for name, res in results.items():
            metrics_data.append({
                'Method': name,
                'Silhouette': res['silhouette_score'],
                'Davies-Bouldin': -res['davies_bouldin_score'],  # Negative for better
                'Calinski-Harabasz': res['calinski_harabasz_score'] / 100,  # Scale down
                'Replicate Consistency': res['replicate_consistency']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index('Method').plot(kind='bar', ax=ax)
        ax.set_title('Embedding Quality Metrics Comparison')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Computational efficiency
        ax = plt.subplot(3, 2, 5)
        
        times = [res['time'] for res in results.values()]
        dims = [res['dimensions'] for res in results.values()]
        methods = list(results.keys())
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, times, width, label='Time (s)', alpha=0.7)
        ax.set_xlabel('Method')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_title('Computational Time')
        
        # Dimensions on secondary axis
        ax2 = ax.twinx()
        ax2.bar(x + width/2, dims, width, label='Dimensions', alpha=0.7, color='orange')
        ax2.set_ylabel('Embedding Dimensions')
        ax2.legend(loc='upper right')
        
        # 4. Summary table
        ax = plt.subplot(3, 2, 6)
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for name, res in results.items():
            summary_data.append([
                name.upper(),
                f"{res['dimensions']}D",
                f"{res['time']:.1f}s",
                f"{res['silhouette_score']:.3f}" if not np.isnan(res['silhouette_score']) else 'N/A',
                f"{res['replicate_consistency']:.1%}" if not np.isnan(res['replicate_consistency']) else 'N/A'
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Method', 'Dims', 'Time', 'Silhouette', 'Rep. Consistency'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Summary Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig('embedding_method_comparison.png', dpi=150, bbox_inches='tight')
        logger.info("\n💾 Saved comparison to: embedding_method_comparison.png")
        
    def run_full_comparison(self):
        """Run complete comparison pipeline."""
        
        # Load test data
        X, metadata = self.load_test_data(drug_name='Sanofi-1', max_series=40)
        
        if X is None:
            logger.error("Failed to load data")
            return
        
        # Run comparisons
        results = self.compare_embeddings(X, metadata)
        
        # Visualize
        self.visualize_comparison(results, metadata)
        
        # Save results
        joblib.dump({
            'results': results,
            'metadata': metadata,
            'data_shape': X.shape
        }, 'embedding_comparison_results.joblib')
        
        logger.info("\n💾 Saved results to: embedding_comparison_results.joblib")
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        
        # Rank methods by different criteria
        methods = list(results.keys())
        
        # By silhouette score
        sil_scores = [(m, results[m]['silhouette_score']) for m in methods 
                     if not np.isnan(results[m]['silhouette_score'])]
        sil_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\n🏆 Best clustering quality (Silhouette Score):")
        for i, (method, score) in enumerate(sil_scores[:3]):
            logger.info(f"  {i+1}. {method.upper()}: {score:.3f}")
        
        # By speed
        time_scores = [(m, results[m]['time']) for m in methods]
        time_scores.sort(key=lambda x: x[1])
        
        logger.info("\n⚡ Fastest methods:")
        for i, (method, time_val) in enumerate(time_scores[:3]):
            logger.info(f"  {i+1}. {method.upper()}: {time_val:.2f}s")
        
        # By replicate consistency
        rep_scores = [(m, results[m]['replicate_consistency']) for m in methods 
                     if not np.isnan(results[m]['replicate_consistency'])]
        rep_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\n🎯 Best replicate consistency:")
        for i, (method, score) in enumerate(rep_scores[:3]):
            logger.info(f"  {i+1}. {method.upper()}: {score:.1%}")
        
        return results


if __name__ == "__main__":
    comparison = EmbeddingComparison()
    comparison.run_full_comparison()