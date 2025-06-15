"""
Embedding methods for organoid time series data.

This package provides various embedding strategies for time series analysis
including traditional methods, feature-based approaches, and deep learning techniques.
"""

import logging

# Import base classes
from .base import BaseEmbedder, TraditionalEmbedder, FeatureEmbedder, DeepLearningEmbedder

# Import all embedding classes
from .traditional.dtw import DTWEmbedder
from .traditional.fourier import FourierEmbedder
from .traditional.sax import SAXEmbedder
from .features.tsfresh_embedder import TSFreshEmbedder
from .features.catch22_embedder import Catch22Embedder
from .features.custom_features import CustomFeaturesEmbedder
from .deep_learning.autoencoder import AutoencoderEmbedder
from .deep_learning.transformer import TransformerEmbedder
from .deep_learning.triplet_network import TripletNetworkEmbedder

logger = logging.getLogger(__name__)

# Export all classes
__all__ = [
    'BaseEmbedder', 'TraditionalEmbedder', 'FeatureEmbedder', 'DeepLearningEmbedder',
    'DTWEmbedder', 'FourierEmbedder', 'SAXEmbedder',
    'TSFreshEmbedder', 'Catch22Embedder', 'CustomFeaturesEmbedder',
    'AutoencoderEmbedder', 'TransformerEmbedder', 'TripletNetworkEmbedder'
]