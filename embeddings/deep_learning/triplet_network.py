"""Triplet Network embedding implementation for time series."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
import logging
import random
from itertools import combinations

from ..base import DeepLearningEmbedder

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset for triplet training."""
    
    def __init__(self, X: np.ndarray, labels: Optional[np.ndarray] = None,
                 replicate_info: Optional[np.ndarray] = None,
                 triplets_per_sample: int = 5):
        """
        Initialize triplet dataset.
        
        Args:
            X: Time series data
            labels: Class labels (optional)
            replicate_info: Replicate information for organoid data
            triplets_per_sample: Number of triplets to generate per sample
        """
        self.X = torch.FloatTensor(X)
        self.labels = labels
        self.replicate_info = replicate_info
        self.triplets_per_sample = triplets_per_sample
        
        # Generate triplets
        self.triplets = self._generate_triplets()
        
    def _generate_triplets(self) -> List[Tuple[int, int, int]]:
        """Generate triplets (anchor, positive, negative)."""
        triplets = []
        n_samples = len(self.X)
        
        if self.replicate_info is not None:
            # Use replicate information for organoid data
            triplets.extend(self._generate_replicate_triplets())
        elif self.labels is not None:
            # Use class labels
            triplets.extend(self._generate_class_triplets())
        else:
            # Random triplets based on similarity
            triplets.extend(self._generate_similarity_triplets())
        
        # Ensure we have enough triplets
        min_triplets = min(len(triplets), n_samples * self.triplets_per_sample)
        return triplets[:min_triplets]
    
    def _generate_replicate_triplets(self) -> List[Tuple[int, int, int]]:
        """Generate triplets based on replicate information."""
        triplets = []
        
        # Group samples by experimental condition (excluding replicate ID)
        condition_groups = {}
        for i, rep_info in enumerate(self.replicate_info):
            # Assume replicate_info contains [drug, concentration, replicate_id]
            condition_key = tuple(rep_info[:2])  # Drug and concentration
            if condition_key not in condition_groups:
                condition_groups[condition_key] = []
            condition_groups[condition_key].append(i)
        
        # Generate triplets within and across conditions
        for condition, sample_indices in condition_groups.items():
            if len(sample_indices) < 2:
                continue
                
            # Create positive pairs within the same condition
            for anchor_idx in sample_indices:
                positive_candidates = [idx for idx in sample_indices if idx != anchor_idx]
                
                for _ in range(self.triplets_per_sample):
                    if not positive_candidates:
                        break
                        
                    positive_idx = random.choice(positive_candidates)
                    
                    # Find negative from different condition
                    negative_candidates = []
                    for other_condition, other_indices in condition_groups.items():
                        if other_condition != condition:
                            negative_candidates.extend(other_indices)
                    
                    if negative_candidates:
                        negative_idx = random.choice(negative_candidates)
                        triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets
    
    def _generate_class_triplets(self) -> List[Tuple[int, int, int]]:
        """Generate triplets based on class labels."""
        triplets = []
        unique_labels = np.unique(self.labels)
        
        # Group samples by label
        label_groups = {}
        for label in unique_labels:
            label_groups[label] = np.where(self.labels == label)[0].tolist()
        
        # Generate triplets
        for label, sample_indices in label_groups.items():
            if len(sample_indices) < 2:
                continue
                
            for anchor_idx in sample_indices:
                positive_candidates = [idx for idx in sample_indices if idx != anchor_idx]
                
                for _ in range(self.triplets_per_sample):
                    if not positive_candidates:
                        break
                        
                    positive_idx = random.choice(positive_candidates)
                    
                    # Find negative from different class
                    negative_candidates = []
                    for other_label, other_indices in label_groups.items():
                        if other_label != label:
                            negative_candidates.extend(other_indices)
                    
                    if negative_candidates:
                        negative_idx = random.choice(negative_candidates)
                        triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets
    
    def _generate_similarity_triplets(self) -> List[Tuple[int, int, int]]:
        """Generate triplets based on time series similarity."""
        triplets = []
        n_samples = len(self.X)
        
        # Compute pairwise similarities (simplified - could use DTW)
        similarities = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Simple correlation-based similarity
                ts1 = self.X[i].numpy()
                ts2 = self.X[j].numpy()
                
                # Handle NaN values
                valid_mask = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid_mask) > 1:
                    corr = np.corrcoef(ts1[valid_mask], ts2[valid_mask])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                
                similarities[i, j] = corr
                similarities[j, i] = corr
        
        # Generate triplets
        for anchor_idx in range(n_samples):
            anchor_similarities = similarities[anchor_idx]
            
            # Sort by similarity
            sorted_indices = np.argsort(anchor_similarities)[::-1]  # Descending order
            
            for _ in range(self.triplets_per_sample):
                # Pick positive from most similar
                positive_candidates = sorted_indices[:len(sorted_indices)//3]  # Top third
                positive_candidates = [idx for idx in positive_candidates if idx != anchor_idx]
                
                # Pick negative from least similar
                negative_candidates = sorted_indices[-len(sorted_indices)//3:]  # Bottom third
                negative_candidates = [idx for idx in negative_candidates if idx != anchor_idx]
                
                if positive_candidates and negative_candidates:
                    positive_idx = random.choice(positive_candidates)
                    negative_idx = random.choice(negative_candidates)
                    triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        
        anchor = self.X[anchor_idx]
        positive = self.X[positive_idx]
        negative = self.X[negative_idx]
        
        return anchor, positive, negative


class TimeSeriesEncoder(nn.Module):
    """Time series encoder for triplet network."""
    
    def __init__(self, input_length: int, embedding_dim: int = 128,
                 hidden_sizes: List[int] = [256, 512, 256],
                 dropout: float = 0.1, architecture: str = 'cnn'):
        super().__init__()
        
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.architecture = architecture
        
        if architecture == 'cnn':
            self.encoder = self._build_cnn_encoder(hidden_sizes, dropout)
        elif architecture == 'lstm':
            self.encoder = self._build_lstm_encoder(hidden_sizes, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Final embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def _build_cnn_encoder(self, hidden_sizes: List[int], dropout: float):
        """Build CNN-based encoder."""
        layers = []
        in_channels = 1
        
        kernel_sizes = [7, 5, 3]
        
        for i, out_channels in enumerate(hidden_sizes):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_lstm_encoder(self, hidden_sizes: List[int], dropout: float):
        """Build LSTM-based encoder."""
        lstm_layers = []
        input_size = 1
        
        for hidden_size in hidden_sizes[:-1]:
            lstm_layers.append(nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout
            ))
            input_size = hidden_size
        
        # Final LSTM layer
        final_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[-1],
            batch_first=True
        )
        
        self.lstm_layers = nn.ModuleList(lstm_layers + [final_lstm])
        
        def lstm_forward(x):
            # x shape: (batch_size, seq_length, 1)
            for lstm in self.lstm_layers:
                x, _ = lstm(x)
            
            # Take the last output
            return x[:, -1, :]  # (batch_size, hidden_size)
        
        return lstm_forward
    
    def forward(self, x):
        """Forward pass through encoder."""
        if self.architecture == 'cnn':
            # Add channel dimension for CNN
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
            features = self.encoder(x)
        else:  # LSTM
            # Add feature dimension for LSTM
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # (batch_size, seq_length, 1)
            features = self.encoder(x)
        
        embedding = self.embedding_layer(features)
        
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class TripletLoss(nn.Module):
    """Triplet loss with margin."""
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, anchor, positive, negative):
        """Compute triplet loss."""
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class TripletNetworkEmbedder(DeepLearningEmbedder):
    """
    Triplet Network embedding for time series data.
    
    Uses triplet loss to learn embeddings where similar time series
    (e.g., replicates) are close and dissimilar ones are far apart.
    """
    
    def __init__(self, embedding_dim: int = 128, 
                 hidden_sizes: List[int] = [256, 512, 256],
                 architecture: str = 'cnn', margin: float = 1.0,
                 distance_metric: str = 'euclidean',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 dropout: float = 0.1, triplets_per_sample: int = 5,
                 **kwargs):
        """
        Initialize triplet network embedder.
        
        Args:
            embedding_dim: Size of the embedding space
            hidden_sizes: Hidden layer sizes for encoder
            architecture: 'cnn' or 'lstm'
            margin: Triplet loss margin
            distance_metric: 'euclidean' or 'cosine'
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            dropout: Dropout rate
            triplets_per_sample: Number of triplets per sample
        """
        super().__init__(name=f"TripletNetwork-{architecture.upper()}", **kwargs)
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.architecture = architecture
        self.margin = margin
        self.distance_metric = distance_metric
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.triplets_per_sample = triplets_per_sample
        
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Build the triplet network model."""
        seq_length = input_shape[0]
        
        model = TimeSeriesEncoder(
            input_length=seq_length,
            embedding_dim=self.embedding_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            architecture=self.architecture
        )
        
        return model.to(self.device)
    
    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
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
        
        return np.array(X_normalized)
    
    def train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                   epochs: int = 100, batch_size: int = 32,
                   replicate_info: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the triplet network model.
        
        Args:
            X: Time series data
            y: Labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            replicate_info: Replicate information for organoid data
        """
        logger.info(f"Training triplet network for {epochs} epochs")
        
        # Prepare data
        X_clean = self._prepare_data(X)
        
        # Create triplet dataset
        dataset = TripletDataset(
            X_clean, labels=y, replicate_info=replicate_info,
            triplets_per_sample=self.triplets_per_sample
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        criterion = TripletLoss(margin=self.margin, distance_metric=self.distance_metric)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        history = {'loss': [], 'lr': []}
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for anchor, positive, negative in dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                
                # Compute triplet loss
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
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
        
        X_clean = self._prepare_data(X)
        X_tensor = torch.FloatTensor(X_clean).to(self.device)
        
        embeddings = []
        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 64
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def compute_distances(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise distances in embedding space.
        
        Args:
            X1: First set of time series
            X2: Second set of time series (if None, computes X1 vs X1)
            
        Returns:
            Distance matrix
        """
        emb1 = self.get_embeddings(X1)
        
        if X2 is None:
            emb2 = emb1
        else:
            emb2 = self.get_embeddings(X2)
        
        if self.distance_metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(emb1, emb2)
        elif self.distance_metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(emb1, emb2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def find_nearest_neighbors(self, X: np.ndarray, query_idx: int, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a query sample.
        
        Args:
            X: Time series data
            query_idx: Index of query sample
            k: Number of neighbors to find
            
        Returns:
            Indices and distances of nearest neighbors
        """
        distances = self.compute_distances(X[query_idx:query_idx+1], X)[0]
        
        # Exclude the query itself
        distances[query_idx] = np.inf
        
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        
        return nearest_indices, nearest_distances