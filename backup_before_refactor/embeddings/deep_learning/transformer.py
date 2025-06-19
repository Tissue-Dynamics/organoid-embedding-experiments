"""Transformer embedding implementation for time series."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import Optional, Dict, List, Tuple
import logging

from ..base import DeepLearningEmbedder

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series embedding."""
    
    def __init__(self, input_dim: int = 1, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 512,
                 embedding_dim: int = 64, dropout: float = 0.1,
                 max_length: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embedding_dim)
        )
        
        # Classification head (for supervised training)
        self.classifier = nn.Linear(embedding_dim, 1)  # Can be modified for multi-class
        
    def create_padding_mask(self, x, padding_value=0):
        """Create padding mask for variable length sequences."""
        return (x == padding_value).all(dim=-1)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            return_attention: Whether to return attention weights
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Create attention mask for padding
        src_key_padding_mask = self.create_padding_mask(x)
        
        # Transformer encoding
        if return_attention:
            encoded, attention_weights = self.transformer_encoder(
                x, src_key_padding_mask=src_key_padding_mask,
                return_attention_weights=True
            )
        else:
            encoded = self.transformer_encoder(
                x, src_key_padding_mask=src_key_padding_mask
            )
        
        # Global pooling
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        
        # Generate embedding
        embedding = self.embedding_layer(pooled)
        
        # Classification output
        logits = self.classifier(embedding)
        
        if return_attention:
            return embedding, logits, attention_weights
        else:
            return embedding, logits


class TransformerEmbedder(DeepLearningEmbedder):
    """
    Transformer embedding for time series data.
    
    Uses transformer architecture with self-attention to learn
    representations that capture long-range dependencies in time series.
    """
    
    def __init__(self, embedding_dim: int = 64, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, dropout: float = 0.1,
                 warmup_epochs: int = 10, use_scheduler: bool = True,
                 supervised: bool = False, **kwargs):
        """
        Initialize transformer embedder.
        
        Args:
            embedding_dim: Size of the embedding space
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            dropout: Dropout rate
            warmup_epochs: Number of warmup epochs for learning rate
            use_scheduler: Whether to use learning rate scheduler
            supervised: Whether to use supervised training (requires labels)
        """
        super().__init__(name="Transformer", **kwargs)
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.warmup_epochs = warmup_epochs
        self.use_scheduler = use_scheduler
        self.supervised = supervised
        
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build the transformer model."""
        seq_length = input_shape[0]
        
        model = TimeSeriesTransformer(
            input_dim=1,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
            max_length=seq_length
        )
        
        return model.to(self.device)
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        
        X_tensor = torch.FloatTensor(np.array(X_normalized)).unsqueeze(-1).to(self.device)
        
        y_tensor = None
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def _get_scheduler(self, optimizer, total_steps):
        """Get learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            if current_step < self.warmup_epochs:
                return float(current_step) / float(max(1, self.warmup_epochs))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - self.warmup_epochs)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                   epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the transformer model."""
        logger.info(f"Training transformer for {epochs} epochs")
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Create data loader
        if self.supervised and y_tensor is not None:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            # Unsupervised: use autoencoding objective
            dataset = TensorDataset(X_tensor, X_tensor.squeeze(-1))
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.supervised and y_tensor is not None:
            criterion = nn.MSELoss()  # Regression
        else:
            criterion = nn.MSELoss()  # Reconstruction loss
        
        # Setup scheduler
        total_steps = epochs
        if self.use_scheduler:
            scheduler = self._get_scheduler(optimizer, total_steps)
        
        # Training loop
        history = {'loss': [], 'lr': []}
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                if self.supervised and y_tensor is not None:
                    batch_x, batch_y = batch
                    optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings, logits = self.model(batch_x)
                    loss = criterion(logits.squeeze(), batch_y)
                else:
                    batch_x, batch_target = batch
                    optimizer.zero_grad()
                    
                    # Forward pass - use embedding to reconstruct
                    embeddings, _ = self.model(batch_x)
                    
                    # Simple reconstruction loss (could be improved)
                    # This is a simplified approach - you might want to add a decoder
                    reconstructed = torch.mean(embeddings, dim=1, keepdim=True).repeat(1, batch_target.size(1))
                    loss = criterion(reconstructed, batch_target)
                
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
            
            if self.use_scheduler:
                scheduler.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        logger.info(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
        return history
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from trained model."""
        self.model.eval()
        
        X_tensor, _ = self._prepare_data(X)
        
        embeddings = []
        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 64
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                batch_embeddings, _ = self.model(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_attention_weights(self, X: np.ndarray, layer_idx: int = -1) -> np.ndarray:
        """
        Extract attention weights from the model.
        
        Args:
            X: Input time series data
            layer_idx: Which layer's attention to extract (-1 for last layer)
            
        Returns:
            Attention weights array
        """
        self.model.eval()
        
        X_tensor, _ = self._prepare_data(X)
        
        attention_weights = []
        with torch.no_grad():
            batch_size = 32  # Smaller batch size for attention extraction
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                _, _, attention = self.model(batch, return_attention=True)
                
                # Extract specific layer attention
                if layer_idx == -1:
                    layer_attention = attention[-1]  # Last layer
                else:
                    layer_attention = attention[layer_idx]
                
                attention_weights.append(layer_attention.cpu().numpy())
        
        return np.vstack(attention_weights)
    
    def visualize_attention(self, X: np.ndarray, sample_idx: int = 0, 
                          layer_idx: int = -1, head_idx: int = 0):
        """
        Visualize attention patterns for a specific sample.
        
        Args:
            X: Input time series data
            sample_idx: Which sample to visualize
            layer_idx: Which layer's attention to visualize
            head_idx: Which attention head to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Matplotlib and seaborn required for attention visualization")
            return
        
        # Get attention weights
        attention_weights = self.get_attention_weights(X[sample_idx:sample_idx+1], layer_idx)
        
        # Extract specific head
        attention_matrix = attention_weights[0, head_idx]  # (seq_len, seq_len)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, cmap='Blues', cbar=True)
        plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.show()