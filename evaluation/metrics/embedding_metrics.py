"""Comprehensive embedding quality evaluation metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


class EmbeddingEvaluator:
    """
    Comprehensive evaluation of embedding quality.
    
    Provides multiple metrics for assessing how well embeddings
    capture the structure and relationships in the original data.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize embedding evaluator."""
        self.random_state = random_state
        self.results_cache = {}
        
    def evaluate_clustering_quality(self, embeddings: np.ndarray,
                                  true_labels: Optional[np.ndarray] = None,
                                  n_clusters_range: Tuple[int, int] = (2, 10)) -> Dict[str, Any]:
        """
        Evaluate clustering quality of embeddings.
        
        Args:
            embeddings: Embedding vectors
            true_labels: True cluster labels (if available)
            n_clusters_range: Range of cluster numbers to test
            
        Returns:
            Dictionary with clustering metrics
        """
        results = {}
        
        # Internal clustering metrics (no true labels needed)
        internal_metrics = self._compute_internal_clustering_metrics(
            embeddings, n_clusters_range
        )
        results.update(internal_metrics)
        
        # External clustering metrics (require true labels)
        if true_labels is not None:
            external_metrics = self._compute_external_clustering_metrics(
                embeddings, true_labels
            )
            results.update(external_metrics)
        
        return results
    
    def _compute_internal_clustering_metrics(self, embeddings: np.ndarray,
                                           n_clusters_range: Tuple[int, int]) -> Dict[str, Any]:
        """Compute internal clustering quality metrics."""
        min_clusters, max_clusters = n_clusters_range
        max_clusters = min(max_clusters, len(embeddings) - 1)
        
        if min_clusters >= max_clusters:
            logger.warning(f"Invalid cluster range: {n_clusters_range}")
            return {}
        
        # Test different numbers of clusters
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        inertias = []
        
        cluster_range = range(min_clusters, max_clusters + 1)
        
        for n_clusters in cluster_range:
            try:
                # K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state,
                               n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Compute metrics
                sil_score = silhouette_score(embeddings, cluster_labels)
                cal_score = calinski_harabasz_score(embeddings, cluster_labels)
                db_score = davies_bouldin_score(embeddings, cluster_labels)
                
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
                davies_bouldin_scores.append(db_score)
                inertias.append(kmeans.inertia_)
                
            except Exception as e:
                logger.warning(f"Failed clustering with {n_clusters} clusters: {e}")
                silhouette_scores.append(np.nan)
                calinski_scores.append(np.nan)
                davies_bouldin_scores.append(np.nan)
                inertias.append(np.nan)
        
        # Find optimal number of clusters
        valid_sil = [s for s in silhouette_scores if not np.isnan(s)]
        valid_cal = [c for c in calinski_scores if not np.isnan(c)]
        valid_db = [d for d in davies_bouldin_scores if not np.isnan(d)]
        
        results = {
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'inertias': inertias,
            'cluster_range': list(cluster_range)
        }
        
        if valid_sil:
            best_sil_idx = np.argmax(valid_sil)
            results['best_silhouette_score'] = valid_sil[best_sil_idx]
            results['best_silhouette_k'] = cluster_range[best_sil_idx]
        
        if valid_cal:
            best_cal_idx = np.argmax(valid_cal)
            results['best_calinski_score'] = valid_cal[best_cal_idx]
            results['best_calinski_k'] = cluster_range[best_cal_idx]
        
        if valid_db:
            best_db_idx = np.argmin(valid_db)  # Lower is better for Davies-Bouldin
            results['best_davies_bouldin_score'] = valid_db[best_db_idx]
            results['best_davies_bouldin_k'] = cluster_range[best_db_idx]
        
        return results
    
    def _compute_external_clustering_metrics(self, embeddings: np.ndarray,
                                           true_labels: np.ndarray) -> Dict[str, Any]:
        """Compute external clustering quality metrics using true labels."""
        results = {}
        
        # Use true number of clusters
        n_true_clusters = len(np.unique(true_labels))
        
        try:
            # K-means with true number of clusters
            kmeans = KMeans(n_clusters=n_true_clusters, random_state=self.random_state)
            predicted_labels = kmeans.fit_predict(embeddings)
            
            # Compute external metrics
            ari = adjusted_rand_score(true_labels, predicted_labels)
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            
            results.update({
                'adjusted_rand_score': ari,
                'normalized_mutual_info': nmi,
                'adjusted_mutual_info': ami,
                'n_true_clusters': n_true_clusters
            })
            
            # Silhouette score with true labels
            if len(np.unique(true_labels)) > 1:
                true_silhouette = silhouette_score(embeddings, true_labels)
                results['true_labels_silhouette'] = true_silhouette
            
        except Exception as e:
            logger.warning(f"Failed external clustering evaluation: {e}")
        
        return results
    
    def evaluate_neighborhood_preservation(self, original_data: np.ndarray,
                                         embeddings: np.ndarray,
                                         k: int = 10) -> Dict[str, float]:
        """
        Evaluate how well embeddings preserve local neighborhoods.
        
        Args:
            original_data: Original high-dimensional data
            embeddings: Low-dimensional embeddings
            k: Number of neighbors to consider
            
        Returns:
            Dictionary with neighborhood preservation metrics
        """
        if len(original_data) != len(embeddings):
            raise ValueError("Original data and embeddings must have same number of samples")
        
        n_samples = len(original_data)
        k = min(k, n_samples - 1)
        
        # Find k-nearest neighbors in original space
        original_nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        original_nn.fit(original_data)
        original_neighbors = original_nn.kneighbors(original_data, return_distance=False)
        
        # Find k-nearest neighbors in embedding space
        embedding_nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        embedding_nn.fit(embeddings)
        embedding_neighbors = embedding_nn.kneighbors(embeddings, return_distance=False)
        
        # Compute neighborhood preservation metrics
        preservation_scores = []
        
        for i in range(n_samples):
            # Exclude self (first neighbor)
            orig_neighs = set(original_neighbors[i][1:])
            emb_neighs = set(embedding_neighbors[i][1:])
            
            # Jaccard similarity
            intersection = len(orig_neighs & emb_neighs)
            union = len(orig_neighs | emb_neighs)
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 1.0
            
            preservation_scores.append(jaccard)
        
        # Trustworthiness and continuity
        trustworthiness = self._compute_trustworthiness(
            original_data, embeddings, k
        )
        continuity = self._compute_continuity(
            original_data, embeddings, k
        )
        
        return {
            'mean_neighborhood_preservation': np.mean(preservation_scores),
            'std_neighborhood_preservation': np.std(preservation_scores),
            'trustworthiness': trustworthiness,
            'continuity': continuity,
            'k_neighbors': k
        }
    
    def _compute_trustworthiness(self, original_data: np.ndarray,
                               embeddings: np.ndarray, k: int) -> float:
        """Compute trustworthiness metric."""
        n = len(original_data)
        
        # Find neighbors
        orig_nn = NearestNeighbors(n_neighbors=k + 1)
        orig_nn.fit(original_data)
        
        emb_nn = NearestNeighbors(n_neighbors=k + 1)
        emb_nn.fit(embeddings)
        emb_neighbors = emb_nn.kneighbors(embeddings, return_distance=False)
        
        # Compute trustworthiness
        trustworthiness_sum = 0
        
        for i in range(n):
            # Neighbors in embedding space (excluding self)
            emb_neighs = emb_neighbors[i][1:]
            
            for j in emb_neighs:
                # Rank of j in original space relative to i
                distances = np.linalg.norm(original_data - original_data[i], axis=1)
                rank = np.sum(distances < distances[j])
                
                if rank > k:
                    trustworthiness_sum += rank - k
        
        normalizer = n * k * (2 * n - 3 * k - 1) / 2
        trustworthiness = 1 - 2 * trustworthiness_sum / normalizer
        
        return trustworthiness
    
    def _compute_continuity(self, original_data: np.ndarray,
                          embeddings: np.ndarray, k: int) -> float:
        """Compute continuity metric."""
        n = len(original_data)
        
        # Find neighbors
        orig_nn = NearestNeighbors(n_neighbors=k + 1)
        orig_nn.fit(original_data)
        orig_neighbors = orig_nn.kneighbors(original_data, return_distance=False)
        
        emb_nn = NearestNeighbors(n_neighbors=k + 1)
        emb_nn.fit(embeddings)
        
        # Compute continuity
        continuity_sum = 0
        
        for i in range(n):
            # Neighbors in original space (excluding self)
            orig_neighs = orig_neighbors[i][1:]
            
            for j in orig_neighs:
                # Rank of j in embedding space relative to i
                distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
                rank = np.sum(distances < distances[j])
                
                if rank > k:
                    continuity_sum += rank - k
        
        normalizer = n * k * (2 * n - 3 * k - 1) / 2
        continuity = 1 - 2 * continuity_sum / normalizer
        
        return continuity
    
    def evaluate_intrinsic_dimensionality(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the intrinsic dimensionality of embeddings.
        
        Args:
            embeddings: Embedding vectors
            
        Returns:
            Dictionary with dimensionality metrics
        """
        # PCA-based effective dimensionality
        pca = PCA()
        pca.fit(embeddings)
        
        # Explained variance ratio
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        # Find number of components for different variance thresholds
        var_90 = np.argmax(cumulative_var >= 0.90) + 1
        var_95 = np.argmax(cumulative_var >= 0.95) + 1
        var_99 = np.argmax(cumulative_var >= 0.99) + 1
        
        # Participation ratio
        participation_ratio = np.sum(explained_var_ratio) ** 2 / np.sum(explained_var_ratio ** 2)
        
        return {
            'effective_dimensionality_90': var_90,
            'effective_dimensionality_95': var_95,
            'effective_dimensionality_99': var_99,
            'participation_ratio': participation_ratio,
            'total_explained_variance': np.sum(explained_var_ratio),
            'explained_variance_ratio': explained_var_ratio.tolist()
        }
    
    def evaluate_stability(self, embeddings_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate stability of embeddings across multiple runs.
        
        Args:
            embeddings_list: List of embedding arrays from different runs
            
        Returns:
            Dictionary with stability metrics
        """
        if len(embeddings_list) < 2:
            logger.warning("Need at least 2 embedding runs for stability evaluation")
            return {}
        
        # Pairwise correlations between runs
        correlations = []
        
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                emb1 = embeddings_list[i].flatten()
                emb2 = embeddings_list[j].flatten()
                
                if len(emb1) == len(emb2):
                    corr = np.corrcoef(emb1, emb2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if not correlations:
            return {'stability_correlation': 0.0}
        
        return {
            'stability_correlation': np.mean(correlations),
            'stability_std': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations)
        }
    
    def evaluate_embedding_quality(self, embeddings: np.ndarray,
                                 original_data: Optional[np.ndarray] = None,
                                 true_labels: Optional[np.ndarray] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Comprehensive embedding quality evaluation.
        
        Args:
            embeddings: Embedding vectors
            original_data: Original high-dimensional data (optional)
            true_labels: True cluster labels (optional)
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1] if embeddings.ndim > 1 else 1
        }
        
        # Clustering quality
        clustering_results = self.evaluate_clustering_quality(
            embeddings, true_labels, **kwargs.get('clustering', {})
        )
        results['clustering'] = clustering_results
        
        # Neighborhood preservation (if original data available)
        if original_data is not None:
            neighborhood_results = self.evaluate_neighborhood_preservation(
                original_data, embeddings, **kwargs.get('neighborhood', {})
            )
            results['neighborhood_preservation'] = neighborhood_results
        
        # Intrinsic dimensionality
        dimensionality_results = self.evaluate_intrinsic_dimensionality(embeddings)
        results['dimensionality'] = dimensionality_results
        
        # Basic statistics
        results['statistics'] = {
            'mean': np.mean(embeddings, axis=0).tolist(),
            'std': np.std(embeddings, axis=0).tolist(),
            'min': np.min(embeddings, axis=0).tolist(),
            'max': np.max(embeddings, axis=0).tolist()
        }
        
        return results