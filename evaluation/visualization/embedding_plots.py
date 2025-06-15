"""Embedding visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Any, Union
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """
    Visualization utilities for embedding analysis.
    
    Provides various plotting methods for visualizing embeddings,
    their quality, and relationships to metadata.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8),
                 style: str = 'whitegrid',
                 palette: str = 'husl'):
        """
        Initialize embedding visualizer.
        
        Args:
            figsize: Default figure size
            style: Seaborn style
            palette: Color palette
        """
        self.figsize = figsize
        self.style = style
        self.palette = palette
        
        # Set style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_embedding_2d(self, embeddings: np.ndarray,
                         labels: Optional[np.ndarray] = None,
                         method: str = 'tsne',
                         title: Optional[str] = None,
                         save_path: Optional[str] = None,
                         **kwargs) -> plt.Figure:
        """
        Create 2D visualization of embeddings.
        
        Args:
            embeddings: Embedding vectors
            labels: Optional labels for coloring
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            title: Plot title
            save_path: Path to save plot
            **kwargs: Additional parameters for reduction methods
            
        Returns:
            Matplotlib figure
        """
        # Reduce to 2D if needed
        if embeddings.shape[1] > 2:
            embeddings_2d = self._reduce_to_2d(embeddings, method, **kwargs)
        else:
            embeddings_2d = embeddings
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            # Color by labels
            unique_labels = np.unique(labels)
            colors = sns.color_palette(self.palette, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[colors[i]], label=str(label), alpha=0.7, s=50)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        else:
            # Single color
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.7, s=50)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        
        if title is None:
            title = f'Embedding Visualization ({method.upper()})'
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _reduce_to_2d(self, embeddings: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """Reduce embeddings to 2D using specified method."""
        if method == 'tsne':
            perplexity = kwargs.get('perplexity', min(30, len(embeddings) - 1))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        
        elif method == 'umap':
            n_neighbors = kwargs.get('n_neighbors', min(15, len(embeddings) - 1))
            min_dist = kwargs.get('min_dist', 0.1)
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                               min_dist=min_dist, random_state=42)
        
        elif method == 'pca':
            reducer = PCA(n_components=2)
        
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reducer.fit_transform(embeddings)
    
    def plot_embedding_comparison(self, embeddings_dict: Dict[str, np.ndarray],
                                labels: Optional[np.ndarray] = None,
                                method: str = 'tsne',
                                ncols: int = 3,
                                save_path: Optional[str] = None,
                                **kwargs) -> plt.Figure:
        """
        Compare multiple embeddings side by side.
        
        Args:
            embeddings_dict: Dictionary of {name: embeddings}
            labels: Optional labels for coloring
            method: Reduction method for visualization
            ncols: Number of columns in subplot grid
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        n_embeddings = len(embeddings_dict)
        nrows = (n_embeddings + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        if n_embeddings == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, (name, embeddings) in enumerate(embeddings_dict.items()):
            ax = axes_flat[i]
            
            # Reduce to 2D
            if embeddings.shape[1] > 2:
                embeddings_2d = self._reduce_to_2d(embeddings, method, **kwargs)
            else:
                embeddings_2d = embeddings
            
            # Plot
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = sns.color_palette(self.palette, len(unique_labels))
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=[colors[j]], label=str(label), alpha=0.7, s=30)
            else:
                ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          alpha=0.7, s=30)
            
            ax.set_title(name)
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
        
        # Hide unused subplots
        for i in range(n_embeddings, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Add legend
        if labels is not None:
            handles, labels_legend = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels_legend, loc='center right',
                      bbox_to_anchor=(1.1, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_embedding_metrics(self, metrics_dict: Dict[str, Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize embedding quality metrics.
        
        Args:
            metrics_dict: Dictionary of {method_name: metrics}
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Extract key metrics for comparison
        methods = list(metrics_dict.keys())
        
        # Define metrics to plot
        metric_keys = [
            ('clustering', 'best_silhouette_score'),
            ('clustering', 'best_calinski_score'),
            ('clustering', 'best_davies_bouldin_score'),
            ('neighborhood_preservation', 'mean_neighborhood_preservation'),
            ('neighborhood_preservation', 'trustworthiness'),
            ('neighborhood_preservation', 'continuity'),
            ('dimensionality', 'participation_ratio')
        ]
        
        # Collect available metrics
        available_metrics = []
        metric_names = []
        
        for category, metric_name in metric_keys:
            values = []
            for method in methods:
                if category in metrics_dict[method] and metric_name in metrics_dict[method][category]:
                    values.append(metrics_dict[method][category][metric_name])
                else:
                    values.append(np.nan)
            
            if not all(np.isnan(values)):
                available_metrics.append(values)
                metric_names.append(f"{category}.{metric_name}")
        
        if not available_metrics:
            logger.warning("No comparable metrics found")
            return plt.figure()
        
        # Create plots
        n_metrics = len(available_metrics)
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        if n_metrics == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        
        for i, (values, metric_name) in enumerate(zip(available_metrics, metric_names)):
            ax = axes_flat[i]
            
            # Remove NaN values
            valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
            valid_methods = [methods[j] for j in valid_indices]
            valid_values = [values[j] for j in valid_indices]
            
            if valid_values:
                bars = ax.bar(valid_methods, valid_values)
                ax.set_title(metric_name, fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                
                # Color bars
                colors = sns.color_palette(self.palette, len(valid_values))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dimension_analysis(self, embeddings: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and visualize dimensional properties of embeddings.
        
        Args:
            embeddings: Embedding vectors
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # PCA analysis
        pca = PCA()
        pca.fit(embeddings)
        
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Explained variance ratio
        axes[0, 0].bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
        axes[0, 0].set_title('Explained Variance Ratio by Component')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        
        # 2. Cumulative explained variance
        axes[0, 1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95%')
        axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='90%')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Eigenvalue spectrum
        eigenvalues = pca.explained_variance_
        axes[1, 0].semilogy(range(1, len(eigenvalues) + 1), eigenvalues, 'go-')
        axes[1, 0].set_title('Eigenvalue Spectrum')
        axes[1, 0].set_xlabel('Component')
        axes[1, 0].set_ylabel('Eigenvalue (log scale)')
        axes[1, 0].grid(True)
        
        # 4. Dimension distribution
        axes[1, 1].hist(embeddings.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 1].set_title('Distribution of Embedding Values')
        axes[1, 1].set_xlabel('Embedding Value')
        axes[1, 1].set_ylabel('Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_clustering_analysis(self, embeddings: np.ndarray,
                               n_clusters_range: Tuple[int, int] = (2, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze clustering properties of embeddings.
        
        Args:
            embeddings: Embedding vectors
            n_clusters_range: Range of cluster numbers to test
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        min_k, max_k = n_clusters_range
        k_range = range(min_k, min(max_k + 1, len(embeddings)))
        
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            sil_score = silhouette_score(embeddings, labels)
            cal_score = calinski_harabasz_score(embeddings, labels)
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            inertias.append(kmeans.inertia_)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Elbow curve
        axes[0, 0].plot(k_range, inertias, 'bo-')
        axes[0, 0].set_title('Elbow Curve')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].grid(True)
        
        # 2. Silhouette score
        axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
        axes[0, 1].set_title('Silhouette Score')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].grid(True)
        
        # 3. Calinski-Harabasz score
        axes[1, 0].plot(k_range, calinski_scores, 'go-')
        axes[1, 0].set_title('Calinski-Harabasz Score')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].grid(True)
        
        # 4. Best clustering visualization
        best_k = k_range[np.argmax(silhouette_scores)]
        kmeans_best = KMeans(n_clusters=best_k, random_state=42)
        best_labels = kmeans_best.fit_predict(embeddings)
        
        # Reduce to 2D for visualization
        if embeddings.shape[1] > 2:
            embeddings_2d = self._reduce_to_2d(embeddings, 'pca')
        else:
            embeddings_2d = embeddings
        
        colors = sns.color_palette(self.palette, best_k)
        for i in range(best_k):
            mask = best_labels == i
            axes[1, 1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=[colors[i]], label=f'Cluster {i}', alpha=0.7)
        
        axes[1, 1].set_title(f'Best Clustering (k={best_k})')
        axes[1, 1].set_xlabel('Dimension 1')
        axes[1, 1].set_ylabel('Dimension 2')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig