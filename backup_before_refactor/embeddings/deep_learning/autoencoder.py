"""Autoencoder embedding implementation for time series."""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..base import DeepLearningEmbedder

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for time series."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=1,  # Single feature (time series value)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Linear(hidden_size, embedding_dim)
        
        # Decoder
        self.decoder_embedding = nn.Linear(embedding_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def encode(self, x):
        """Encode input to embedding space."""
        # x shape: (batch_size, seq_length, 1)
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        
        # Use the last hidden state for encoding
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        embedding = self.bottleneck(last_hidden)
        
        return embedding
    
    def decode(self, embedding, seq_length):
        """Decode embedding back to time series."""
        batch_size = embedding.size(0)
        
        # Transform embedding to decoder input
        decoder_input = self.decoder_embedding(embedding)  # (batch_size, hidden_size)
        
        # Repeat for sequence length
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_length, 1)
        
        # Decode
        lstm_out, _ = self.decoder_lstm(decoder_input)
        output = self.output_layer(lstm_out)
        
        return output
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        seq_length = x.size(1)
        embedding = self.encode(x)
        reconstruction = self.decode(embedding, seq_length)
        return reconstruction, embedding


class CNNAutoencoder(nn.Module):
    """CNN-based autoencoder for time series."""
    
    def __init__(self, input_length: int, embedding_dim: int, 
                 num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [7, 5, 3],
                 dropout: float = 0.1):
        super().__init__()
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        
        # Encoder
        encoder_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the size after convolutions
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            encoded_size = self.encoder(dummy_input).view(1, -1).size(1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(encoded_size, embedding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder_fc = nn.Linear(embedding_dim, encoded_size)
        
        decoder_layers = []
        in_channels = num_filters[-1]
        
        for i, (out_channels, kernel_size) in enumerate(zip(reversed(num_filters[:-1]), reversed(kernel_sizes[:-1]))):
            decoder_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size//2, output_padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        # Final layer
        decoder_layers.append(nn.ConvTranspose1d(in_channels, 1, kernel_sizes[0], stride=2, padding=kernel_sizes[0]//2, output_padding=1))
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.encoded_shape = (-1, num_filters[-1], input_length // (2 ** len(num_filters)))
        
    def encode(self, x):
        """Encode input to embedding space."""
        # x shape: (batch_size, 1, seq_length)
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        embedding = self.bottleneck(flattened)
        return embedding
    
    def decode(self, embedding):
        """Decode embedding back to time series."""
        decoded_flat = self.decoder_fc(embedding)
        decoded_shaped = decoded_flat.view(self.encoded_shape)
        reconstruction = self.decoder(decoded_shaped)
        
        # Adjust output length if necessary
        if reconstruction.size(2) != self.input_length:
            reconstruction = torch.nn.functional.interpolate(
                reconstruction, size=self.input_length, mode='linear', align_corners=False
            )
        
        return reconstruction
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding


class AutoencoderEmbedder(DeepLearningEmbedder):
    """
    Autoencoder embedding for time series data.
    
    Supports both LSTM and CNN architectures for learning compressed
    representations of time series data.
    """
    
    def __init__(self, embedding_dim: int = 64, architecture: str = 'lstm',
                 hidden_size: int = 128, num_layers: int = 2,
                 num_filters: Optional[List[int]] = None,
                 kernel_sizes: Optional[List[int]] = None,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 dropout: float = 0.1, **kwargs):
        """
        Initialize autoencoder embedder.
        
        Args:
            embedding_dim: Size of the embedding space
            architecture: 'lstm' or 'cnn'
            hidden_size: Hidden size for LSTM
            num_layers: Number of LSTM layers
            num_filters: Filter sizes for CNN layers
            kernel_sizes: Kernel sizes for CNN layers
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            dropout: Dropout rate
        """
        super().__init__(name=f"Autoencoder-{architecture.upper()}", **kwargs)
        self.embedding_dim = embedding_dim
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_filters = num_filters or [32, 64, 128]
        self.kernel_sizes = kernel_sizes or [7, 5, 3]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build the autoencoder model."""
        seq_length = input_shape[0]
        
        if self.architecture == 'lstm':
            model = LSTMAutoencoder(
                input_size=seq_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                embedding_dim=self.embedding_dim,
                dropout=self.dropout
            )
        elif self.architecture == 'cnn':
            model = CNNAutoencoder(
                input_length=seq_length,
                embedding_dim=self.embedding_dim,
                num_filters=self.num_filters,
                kernel_sizes=self.kernel_sizes,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        return model.to(self.device)
    
    def _prepare_data(self, X: np.ndarray) -> torch.Tensor:
        """Prepare data for training."""
        # Handle NaN values
        X_clean = X.copy()
        for i in range(len(X_clean)):
            ts = X_clean[i]
            if np.any(np.isnan(ts)):
                # Simple interpolation
                valid_mask = ~np.isnan(ts)
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(~valid_mask)[0]
                    X_clean[i, invalid_indices] = np.interp(
                        invalid_indices, valid_indices, ts[valid_indices]
                    )
                else:
                    X_clean[i] = 0.0  # All NaN case
        
        # Normalize
        X_normalized = []
        for ts in X_clean:
            ts_norm = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
            X_normalized.append(ts_norm)
        
        X_tensor = torch.FloatTensor(np.array(X_normalized)).to(self.device)
        
        if self.architecture == 'lstm':
            # Add feature dimension: (batch_size, seq_length, 1)
            X_tensor = X_tensor.unsqueeze(-1)
        elif self.architecture == 'cnn':
            # Add channel dimension: (batch_size, 1, seq_length)
            X_tensor = X_tensor.unsqueeze(1)
        
        return X_tensor
    
    def train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                   epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the autoencoder model."""
        logger.info(f"Training {self.architecture} autoencoder for {epochs} epochs")
        
        # Prepare data
        X_tensor = self._prepare_data(X)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # Input = target for autoencoder
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        history = {'loss': [], 'lr': []}
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstruction, embedding = self.model(batch_x)
                loss = criterion(reconstruction, batch_target)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            
            history['loss'].append(avg_loss)
            history['lr'].append(current_lr)
            
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        logger.info(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
        return history
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from trained model."""
        self.model.eval()
        
        X_tensor = self._prepare_data(X)
        
        embeddings = []
        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 64
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data from learned representations."""
        self.model.eval()
        
        X_tensor = self._prepare_data(X)
        
        reconstructions = []
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                batch_reconstruction, _ = self.model(batch)
                
                # Convert back to original format
                if self.architecture == 'lstm':
                    batch_reconstruction = batch_reconstruction.squeeze(-1)
                elif self.architecture == 'cnn':
                    batch_reconstruction = batch_reconstruction.squeeze(1)
                
                reconstructions.append(batch_reconstruction.cpu().numpy())
        
        return np.vstack(reconstructions)
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error for each sample."""
        reconstructions = self.reconstruct(X)
        
        # Normalize X the same way as in training
        X_normalized = []
        for ts in X:
            if np.any(np.isnan(ts)):
                valid_mask = ~np.isnan(ts)
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(~valid_mask)[0]
                    ts_clean = ts.copy()
                    ts_clean[invalid_indices] = np.interp(
                        invalid_indices, valid_indices, ts[valid_indices]
                    )
                else:
                    ts_clean = np.zeros_like(ts)
            else:
                ts_clean = ts
            
            ts_norm = (ts_clean - np.mean(ts_clean)) / (np.std(ts_clean) + 1e-8)
            X_normalized.append(ts_norm)
        
        X_normalized = np.array(X_normalized)
        
        # Compute MSE for each sample
        mse_errors = np.mean((X_normalized - reconstructions) ** 2, axis=1)
        return mse_errors