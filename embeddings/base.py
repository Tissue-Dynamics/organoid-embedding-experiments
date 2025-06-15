"""Base classes for embedding methods."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all embedding methods."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.is_fitted = False
        self.embedding_dim = None
        self.config = kwargs
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseEmbedder':
        """
        Fit the embedding method to training data.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            y: Optional labels for supervised methods
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series data to embedding space.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            Embeddings of shape (n_samples, embedding_dim)
        """
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration parameters."""
        return {
            'name': self.name,
            'embedding_dim': self.embedding_dim,
            **self.config
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class TraditionalEmbedder(BaseEmbedder):
    """Base class for traditional time series embedding methods."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.distance_matrix = None
        
    def compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix for embedding.
        Should be implemented by subclasses.
        """
        raise NotImplementedError
        
    def distance_to_embedding(self, distance_matrix: np.ndarray, 
                            method: str = 'mds', n_components: int = 50) -> np.ndarray:
        """
        Convert distance matrix to embedding using dimensionality reduction.
        
        Args:
            distance_matrix: Pairwise distances
            method: Dimensionality reduction method ('mds', 'tsne', 'umap')
            n_components: Target embedding dimension
            
        Returns:
            Low-dimensional embedding
        """
        from sklearn.manifold import MDS
        
        if method == 'mds':
            mds = MDS(n_components=n_components, dissimilarity='precomputed', 
                     random_state=42, n_jobs=-1)
            return mds.fit_transform(distance_matrix)
        else:
            raise NotImplementedError(f"Method {method} not implemented")


class FeatureEmbedder(BaseEmbedder):
    """Base class for feature-based embedding methods."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.feature_extractor = None
        self.scaler = None
        
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from time series data."""
        raise NotImplementedError
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureEmbedder':
        """Fit feature extractor and scaler."""
        features = self.extract_features(X)
        
        # Fit scaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        self.embedding_dim = features.shape[1]
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract and scale features."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before transform")
            
        features = self.extract_features(X)
        return self.scaler.transform(features)


class DeepLearningEmbedder(BaseEmbedder):
    """Base class for deep learning embedding methods."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.model = None
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Build the neural network model."""
        raise NotImplementedError
        
    def train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                   epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the model and return training history."""
        raise NotImplementedError
        
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from trained model."""
        raise NotImplementedError
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DeepLearningEmbedder':
        """Fit the deep learning model."""
        input_shape = X.shape[1:]
        self.model = self.build_model(input_shape)
        
        training_config = {
            'epochs': self.config.get('epochs', 100),
            'batch_size': self.config.get('batch_size', 32)
        }
        
        history = self.train_model(X, y, **training_config)
        self.training_history = history
        
        # Determine embedding dimension
        sample_embedding = self.get_embeddings(X[:1])
        self.embedding_dim = sample_embedding.shape[1]
        
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using trained model."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before transform")
            
        return self.get_embeddings(X)