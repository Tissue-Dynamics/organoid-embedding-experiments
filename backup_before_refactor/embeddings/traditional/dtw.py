"""Dynamic Time Warping (DTW) embedding implementation."""

import numpy as np
from typing import Optional
import logging
from tslearn.metrics import dtw
from joblib import Parallel, delayed
from tqdm import tqdm

from ..base import TraditionalEmbedder

logger = logging.getLogger(__name__)


class DTWEmbedder(TraditionalEmbedder):
    """
    Dynamic Time Warping embedding for time series data.
    
    Uses DTW distance to compute pairwise similarities and then applies
    dimensionality reduction to create embeddings.
    """
    
    def __init__(self, n_components: int = 50, sakoe_chiba_radius: Optional[int] = None,
                 itakura_max_slope: Optional[float] = None, n_jobs: int = -1, **kwargs):
        """
        Initialize DTW embedder.
        
        Args:
            n_components: Target embedding dimension
            sakoe_chiba_radius: Constraint window for DTW alignment
            itakura_max_slope: Itakura parallelogram constraint
            n_jobs: Number of parallel jobs for distance computation
        """
        super().__init__(name="DTW", **kwargs)
        self.n_components = n_components
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.itakura_max_slope = itakura_max_slope
        self.n_jobs = n_jobs
        self.training_data = None
        
    def _compute_dtw_distance(self, ts1: np.ndarray, ts2: np.ndarray) -> float:
        """Compute DTW distance between two time series."""
        kwargs = {}
        if self.sakoe_chiba_radius is not None:
            kwargs['sakoe_chiba_radius'] = self.sakoe_chiba_radius
        if self.itakura_max_slope is not None:
            kwargs['itakura_max_slope'] = self.itakura_max_slope
            
        return dtw(ts1, ts2, **kwargs)
    
    def _compute_distance_row(self, i: int, X: np.ndarray, reference_data: np.ndarray) -> np.ndarray:
        """Compute DTW distances for one row of the distance matrix."""
        distances = np.zeros(len(reference_data))
        for j, ref_ts in enumerate(reference_data):
            distances[j] = self._compute_dtw_distance(X[i], ref_ts)
        return distances
    
    def compute_distance_matrix(self, X: np.ndarray, reference_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute DTW distance matrix.
        
        Args:
            X: Time series data
            reference_data: Reference time series (if None, uses X)
            
        Returns:
            Distance matrix
        """
        if reference_data is None:
            reference_data = X
            
        n_samples = len(X)
        n_reference = len(reference_data)
        
        logger.info(f"Computing DTW distance matrix ({n_samples} x {n_reference})")
        
        # Parallel computation of distance matrix
        distances = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_distance_row)(i, X, reference_data)
            for i in tqdm(range(n_samples), desc="Computing DTW distances")
        )
        
        return np.array(distances)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DTWEmbedder':
        """
        Fit DTW embedder by storing training data.
        
        Args:
            X: Training time series data of shape (n_samples, n_timepoints)
            y: Ignored for unsupervised method
            
        Returns:
            Self for method chaining
        """
        self.training_data = X.copy()
        self.embedding_dim = self.n_components
        self.is_fitted = True
        
        logger.info(f"DTW embedder fitted with {len(X)} training samples")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to DTW embedding space.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            DTW embeddings of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("DTW embedder must be fitted before transform")
        
        # Compute distances to training data
        distance_matrix = self.compute_distance_matrix(X, self.training_data)
        
        # Apply MDS to get embeddings
        embeddings = self.distance_to_embedding(
            distance_matrix, 
            method='mds', 
            n_components=self.n_components
        )
        
        return embeddings
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step (optimized for symmetric case).
        
        Args:
            X: Time series data
            y: Ignored
            
        Returns:
            DTW embeddings
        """
        self.training_data = X.copy()
        self.embedding_dim = self.n_components
        self.is_fitted = True
        
        # Compute symmetric distance matrix
        n_samples = len(X)
        distance_matrix = np.zeros((n_samples, n_samples))
        
        logger.info(f"Computing symmetric DTW distance matrix ({n_samples} x {n_samples})")
        
        # Compute upper triangle (DTW is symmetric)
        for i in tqdm(range(n_samples), desc="Computing DTW distances"):
            for j in range(i + 1, n_samples):
                dist = self._compute_dtw_distance(X[i], X[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Apply MDS
        embeddings = self.distance_to_embedding(
            distance_matrix,
            method='mds',
            n_components=self.n_components
        )
        
        return embeddings