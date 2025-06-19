"""Deep learning embedding methods for time series."""

from .autoencoder import AutoencoderEmbedder
from .transformer import TransformerEmbedder
from .triplet_network import TripletNetworkEmbedder

__all__ = ['AutoencoderEmbedder', 'TransformerEmbedder', 'TripletNetworkEmbedder']