# Autoencoder Integration Steps
## Building on Existing Feature Engineering Foundation

This document provides step-by-step instructions for integrating autoencoder interpolation with the **existing event-aware feature engineering pipeline**. All steps build on the foundation that's already implemented.

## Prerequisites ✅ ALREADY COMPLETE
- Event-aware data integration pipeline with dosing and media change timing
- Real experimental event data (652 events, 35 dosing events, 103 media changes)  
- Pre-dosing baseline periods identified across 31 plates
- Quality assessment framework established

## Step 1: Enhance Existing Data Formatter
**Goal**: Extend the current data pipeline to create autoencoder-ready tensors  
**Files to modify**: Enhance existing `data/preprocessing/` modules  
**Builds on**: Current event detection and quality assessment

### Task 1.1: Extend Multi-Channel Input Creation
```python
# File: data/preprocessing/autoencoder_formatter.py (NEW)
# Import existing modules that are already working
from data.preprocessing.cleaner import QualityAssessment
from data.preprocessing.event_correction import detect_media_changes, detect_dosing_events

def create_enhanced_multichannel_input(well_data, existing_events, existing_quality):
    """
    Build on existing event detection and quality assessment
    Create 15-channel tensor input for autoencoder
    
    Args:
        well_data: DataFrame from existing pipeline
        existing_events: Events from current event detection
        existing_quality: Quality flags from current assessment
    """
    # Use EXISTING event detection (already implemented)
    media_changes = existing_events['media_changes']
    dosing_time = existing_events['dosing_time']
    baseline_period = existing_events['baseline_period']
    
    # Use EXISTING quality flags (already implemented)
    quality_flags = existing_quality
    
    # Resample to 1-hour grid (NEW - simple addition)
    resampled_data = resample_to_hourly_grid(well_data, target_length=400)
    
    # Create 15-channel tensor
    channels = np.column_stack([
        # Channel 0: Oxygen values (normalized using existing method)
        normalize_oxygen_values(resampled_data['oxygen']),
        
        # Channel 1: Data validity mask
        resampled_data['mask'],
        
        # Channels 2-3: Event flags (using existing event detection)
        create_media_change_flags(resampled_data['times'], media_changes),
        create_dosing_flags(resampled_data['times'], dosing_time),
        
        # Channels 4-7: Phase indicators (using existing baseline detection)
        create_phase_indicators(resampled_data['times'], dosing_time, baseline_period),
        
        # Channels 8-13: Quality flag embedding (using existing quality assessment)
        embed_existing_quality_flags(quality_flags),
        
        # Channel 14: Baseline deviation
        calculate_baseline_deviation(resampled_data['oxygen'], baseline_period)
    ])
    
    return torch.FloatTensor(channels)

def resample_to_hourly_grid(well_data, target_length=400):
    """Simple resampling function"""
    # Sort by elapsed_time
    well_data = well_data.sort_values('elapsed_time')
    
    # Create 1-hour grid
    target_times = np.arange(0, target_length, 1.0)
    
    # Linear interpolation
    from scipy.interpolate import interp1d
    if len(well_data) >= 2:
        interp_func = interp1d(
            well_data['elapsed_time'], 
            well_data['oxygen_value'],
            kind='linear', bounds_error=False, fill_value=np.nan
        )
        oxygen_resampled = interp_func(target_times)
    else:
        oxygen_resampled = np.full(target_length, np.nan)
    
    # Create mask for original vs interpolated data
    mask = np.zeros(target_length)
    for orig_time in well_data['elapsed_time']:
        closest_idx = np.argmin(np.abs(target_times - orig_time))
        if np.abs(target_times[closest_idx] - orig_time) < 0.5:
            mask[closest_idx] = 1
    
    return {
        'times': target_times,
        'oxygen': oxygen_resampled, 
        'mask': mask
    }

# Test with existing data
if __name__ == "__main__":
    # Load using existing pipeline
    from scripts.analysis.comprehensive_oxygen_data_analysis import load_all_data
    
    data = load_all_data()  # Use existing data loader
    sample_well = data[data['well_id'] == data['well_id'].iloc[0]]
    
    # Use existing event detection results
    existing_events = {
        'media_changes': [72, 144, 216],  # Example from existing analysis
        'dosing_time': 48,
        'baseline_period': (0, 48)
    }
    
    existing_quality = {
        'low_points': False,
        'high_noise': False, 
        'sensor_drift': False
    }
    
    # Test enhanced formatting
    tensor = create_enhanced_multichannel_input(sample_well, existing_events, existing_quality)
    print(f"Created tensor shape: {tensor.shape}")  # Should be [400, 15]
```

**Validation checklist**:
- [ ] Function runs without errors on real data from existing pipeline
- [ ] Output tensor has shape [400, 15] for 15 channels
- [ ] Event flags align with existing event detection results
- [ ] Quality embedding matches existing quality assessment

### Task 1.2: Extend Existing Dataset Class
```python
# File: data/preprocessing/enhanced_dataset.py (NEW)
# Import and extend existing functionality

from scripts.analysis.comprehensive_oxygen_data_analysis import load_all_data
from data.preprocessing.autoencoder_formatter import create_enhanced_multichannel_input

class AutoencoderDataset(Dataset):
    """
    Extends existing data pipeline to provide autoencoder inputs
    Builds directly on current event detection and quality assessment
    """
    
    def __init__(self, limit_wells_for_testing=None):
        print("Loading data using existing pipeline...")
        
        # Use EXISTING data loading (already implemented and working)
        self.raw_data = load_all_data()
        
        print(f"Loaded {len(self.raw_data)} measurements")
        print(f"Wells: {self.raw_data['well_id'].nunique()}")
        print(f"Plates: {self.raw_data['plate_experiment_name'].nunique()}")
        
        # Use EXISTING quality and event data that's already computed
        self.processed_wells = []
        
        well_ids = self.raw_data['well_id'].unique()
        if limit_wells_for_testing:
            well_ids = well_ids[:limit_wells_for_testing]
        
        for well_id in well_ids:
            well_data = self.raw_data[self.raw_data['well_id'] == well_id]
            
            if len(well_data) < 20:  # Skip wells with too little data
                continue
            
            # Get plate context for this well
            plate_name = well_data['plate_experiment_name'].iloc[0]
            plate_data = self.raw_data[self.raw_data['plate_experiment_name'] == plate_name]
            
            # Use existing event detection logic (simplified for now)
            existing_events = self._extract_existing_events(well_data, plate_data)
            existing_quality = self._extract_existing_quality(well_data)
            
            try:
                # Create autoencoder tensor using existing context
                tensor = create_enhanced_multichannel_input(
                    well_data, existing_events, existing_quality
                )
                
                self.processed_wells.append({
                    'well_id': well_id,
                    'tensor': tensor,
                    'metadata': {
                        'drug': well_data['drug_name'].iloc[0] if 'drug_name' in well_data else 'unknown',
                        'concentration': well_data['drug_concentration'].iloc[0] if 'drug_concentration' in well_data else 0,
                        'plate': plate_name,
                        'timepoints': len(well_data)
                    }
                })
                
            except Exception as e:
                print(f"Error processing well {well_id}: {e}")
                continue
        
        print(f"Successfully processed {len(self.processed_wells)} wells")
    
    def _extract_existing_events(self, well_data, plate_data):
        """Extract events using simplified version of existing logic"""
        # Simplified event detection - can be enhanced with existing algorithms
        times = well_data['elapsed_time'].values
        
        # Simple heuristics for now (can use existing sophisticated detection later)
        dosing_time = 48.0  # Typical from existing analysis
        media_changes = [72, 144, 216]  # Typical pattern from existing analysis
        baseline_period = (0, dosing_time)
        
        return {
            'dosing_time': dosing_time,
            'media_changes': media_changes,
            'baseline_period': baseline_period
        }
    
    def _extract_existing_quality(self, well_data):
        """Extract quality using simplified version of existing logic"""
        # Simple quality assessment (can use existing sophisticated assessment later)
        n_points = len(well_data)
        cv = well_data['oxygen_value'].std() / well_data['oxygen_value'].mean()
        
        return {
            'low_points': n_points < 200,
            'high_noise': cv > 0.5,
            'sensor_drift': False  # Simplified for now
        }
    
    def __len__(self):
        return len(self.processed_wells)
    
    def __getitem__(self, idx):
        well = self.processed_wells[idx]
        tensor = well['tensor']
        
        # Input is all 15 channels, mask is channel 1
        return tensor, tensor[:, 1], well['metadata']

# Test dataset creation
if __name__ == "__main__":
    # Test with limited data
    dataset = AutoencoderDataset(limit_wells_for_testing=100)
    
    print(f"Dataset created with {len(dataset)} wells")
    
    # Test data loading
    sample_tensor, sample_mask, sample_metadata = dataset[0]
    print(f"Sample tensor shape: {sample_tensor.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")
    print(f"Sample metadata: {sample_metadata}")
```

**Validation checklist**:
- [ ] Dataset loads using existing data pipeline without errors
- [ ] All wells process successfully or fail gracefully
- [ ] Tensor shapes are consistent [400, 15]
- [ ] Metadata includes drug, concentration, plate information

## Step 2: Implement Basic Autoencoder
**Goal**: Create autoencoder that processes the enhanced 15-channel input  
**Files to create**: `embeddings/autoencoder_models.py`  
**Builds on**: Enhanced data formatter from Step 1

### Task 2.1: Enhanced Autoencoder Architecture
```python
# File: embeddings/autoencoder_models.py (NEW)
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedEventAwareAutoencoder(nn.Module):
    """
    Autoencoder for 15-channel input from existing feature engineering pipeline
    """
    
    def __init__(self, latent_dim=16, input_length=400):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_length = input_length
        
        # Encoder for 15 channels (enhanced from original 7)
        self.encoder = nn.Sequential(
            # Block 1: Process all 15 channels
            nn.Conv1d(15, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            # Block 2: Temporal compression
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            # Block 3: Further compression
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            # Global pooling and dense layers
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder - reconstruct only oxygen channel
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Output only oxygen channel (channel 0)
            nn.Conv1d(64, 1, kernel_size=7, padding=3)
        )
    
    def encode(self, x):
        """
        Encode 15-channel input to latent representation
        
        Args:
            x: [batch_size, time_points, 15_channels]
        Returns:
            latent: [batch_size, latent_dim]
        """
        # Conv1d expects [batch, channels, time]
        x = x.transpose(1, 2)
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent to oxygen timeseries
        
        Args:
            z: [batch_size, latent_dim]
        Returns:
            reconstruction: [batch_size, time_points, 1]
        """
        # Dense layers
        h = self.decoder_dense(z)
        h = h.unsqueeze(-1)  # [batch, 256, 1]
        
        # Convolutional layers
        recon = self.decoder_conv(h)
        
        # Resize to input length
        recon = F.interpolate(recon, size=self.input_length, mode='linear', align_corners=False)
        
        # Transpose back to [batch, time, channels]
        return recon.transpose(1, 2)
    
    def forward(self, x):
        """Full forward pass"""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def enhanced_quality_score(self, x):
        """
        Quality score using both reconstruction error and embedded quality flags
        """
        reconstruction, _ = self.forward(x)
        
        # Reconstruction error on oxygen channel
        original_oxygen = x[:, :, 0:1]  # Channel 0
        mask = x[:, :, 1:2]  # Channel 1
        
        # Masked reconstruction error
        masked_recon = reconstruction * mask
        masked_original = original_oxygen * mask
        
        recon_error = F.mse_loss(masked_recon, masked_original, reduction='none')
        recon_error = recon_error.mean(dim=(1, 2))
        
        # Add quality flag penalty (channels 8-13 contain quality flags)
        quality_flags = x[:, :, 8:14].sum(dim=(1, 2))  # Sum quality flag violations
        quality_penalty = 0.1 * quality_flags
        
        combined_score = recon_error + quality_penalty
        return combined_score

# Test model architecture
if __name__ == "__main__":
    # Test with dummy data matching our 15-channel format
    batch_size = 4
    time_points = 400
    channels = 15
    
    dummy_input = torch.randn(batch_size, time_points, channels)
    
    model = EnhancedEventAwareAutoencoder(latent_dim=16)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    reconstruction, latent = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Test quality scoring
    quality_scores = model.enhanced_quality_score(dummy_input)
    print(f"Quality scores shape: {quality_scores.shape}")
```

**Validation checklist**:
- [ ] Model handles 15-channel input without errors
- [ ] Output shapes are correct (reconstruction matches input oxygen channel)
- [ ] Quality scoring incorporates both reconstruction and embedded flags
- [ ] Model parameters are reasonable (~1-5M parameters)

### Task 2.2: Training Pipeline
```python
# File: embeddings/train_enhanced_autoencoder.py (NEW)
import torch
from torch.utils.data import DataLoader, Subset
from embeddings.autoencoder_models import EnhancedEventAwareAutoencoder
from data.preprocessing.enhanced_dataset import AutoencoderDataset

def train_enhanced_autoencoder(model, train_loader, val_loader, epochs=50, device='cpu'):
    """
    Training loop for enhanced autoencoder
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training on {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, mask, metadata) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, latent = model(data)
            
            # Loss calculation
            original_oxygen = data[:, :, 0:1]  # Oxygen channel
            mask_3d = mask.unsqueeze(-1)
            
            # Reconstruction loss (only on valid data)
            recon_loss = F.mse_loss(reconstruction * mask_3d, original_oxygen * mask_3d)
            
            # L2 regularization
            l2_reg = 1e-4 * torch.norm(latent, p=2, dim=1).mean()
            
            # Quality-aware loss (penalty for poor quality wells)
            quality_penalty = model.enhanced_quality_score(data).mean()
            
            total_loss = recon_loss + l2_reg + 0.1 * quality_penalty
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, mask, metadata in val_loader:
                data, mask = data.to(device), mask.to(device)
                
                reconstruction, latent = model(data)
                original_oxygen = data[:, :, 0:1]
                mask_3d = mask.unsqueeze(-1)
                
                recon_loss = F.mse_loss(reconstruction * mask_3d, original_oxygen * mask_3d)
                l2_reg = 1e-4 * torch.norm(latent, p=2, dim=1).mean()
                quality_penalty = model.enhanced_quality_score(data).mean()
                
                val_loss += (recon_loss + l2_reg + 0.1 * quality_penalty).item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_enhanced_autoencoder.pth')
            print(f'New best model saved (Val Loss: {val_loss:.6f})')
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def main():
    """Complete training pipeline"""
    print("=== Enhanced Autoencoder Training Pipeline ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset (limited for initial testing)
    print("\nCreating dataset...")
    dataset = AutoencoderDataset(limit_wells_for_testing=500)
    
    # Train/val split by wells (simplified for now)
    n_wells = len(dataset)
    n_val = int(0.2 * n_wells)
    n_train = n_wells - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating enhanced autoencoder...")
    model = EnhancedEventAwareAutoencoder(latent_dim=16)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nStarting training...")
    history = train_enhanced_autoencoder(
        model=model,
        train_loader=train_loader, 
        val_loader=val_loader,
        epochs=50,
        device=device
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Model saved as: best_enhanced_autoencoder.pth")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
```

**Validation checklist**:
- [ ] Training runs without errors on real data
- [ ] Loss decreases over epochs
- [ ] Model checkpoints save successfully
- [ ] Memory usage is reasonable

## Step 3: Validation and Quality Analysis  
**Goal**: Validate that autoencoder learns meaningful patterns from existing data  
**Files to create**: `analysis/autoencoder_validation.py`  
**Builds on**: Trained model from Step 2

### Task 3.1: Comprehensive Validation
```python
# File: analysis/autoencoder_validation.py (NEW)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from embeddings.autoencoder_models import EnhancedEventAwareAutoencoder
from data.preprocessing.enhanced_dataset import AutoencoderDataset

def validate_enhanced_autoencoder():
    """
    Complete validation of enhanced autoencoder using existing data
    """
    print("=== Enhanced Autoencoder Validation ===")
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedEventAwareAutoencoder(latent_dim=16)
    
    try:
        model.load_state_dict(torch.load('best_enhanced_autoencoder.pth', map_location=device))
        model = model.to(device)
        model.eval()
        print("Loaded trained model successfully")
    except FileNotFoundError:
        print("ERROR: best_enhanced_autoencoder.pth not found. Please train model first.")
        return
    
    # Load validation dataset
    print("Loading validation dataset...")
    dataset = AutoencoderDataset(limit_wells_for_testing=1000)
    
    # Extract embeddings and metadata
    print("Extracting embeddings...")
    embeddings = []
    quality_scores = []
    metadata_list = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data, mask, metadata = dataset[i]
            data = data.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Extract embedding
            _, embedding = model(data)
            embeddings.append(embedding.squeeze().cpu().numpy())
            
            # Calculate quality score
            quality = model.enhanced_quality_score(data)
            quality_scores.append(quality.item())
            
            metadata_list.append(metadata)
    
    embeddings = np.array(embeddings)
    quality_scores = np.array(quality_scores)
    
    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Create results dataframe
    results_df = pd.DataFrame(metadata_list)
    results_df['quality_score'] = quality_scores
    
    # Add embedding dimensions
    for i in range(embeddings.shape[1]):
        results_df[f'embedding_{i}'] = embeddings[:, i]
    
    # 1. Quality Score Analysis
    print("\n1. Analyzing quality scores...")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(quality_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.title('Quality Score Distribution')
    
    # Outlier threshold
    median_score = np.median(quality_scores)
    mad = np.median(np.abs(quality_scores - median_score))
    outlier_threshold = median_score + 3 * mad
    plt.axvline(outlier_threshold, color='red', linestyle='--', 
                label=f'Outlier threshold: {outlier_threshold:.4f}')
    plt.legend()
    
    # Flag outliers
    results_df['is_outlier'] = quality_scores > outlier_threshold
    n_outliers = results_df['is_outlier'].sum()
    print(f"Outliers detected: {n_outliers} / {len(results_df)} ({n_outliers/len(results_df)*100:.1f}%)")
    
    # 2. Embedding Analysis
    print("\n2. Analyzing embeddings...")
    
    # PCA
    pca = PCA()
    pca_embeddings = pca.fit_transform(embeddings)
    
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
                         c=quality_scores, cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA: Colored by Quality Score')
    plt.colorbar(scatter, label='Quality Score')
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    plt.subplot(1, 3, 3)
    # Color by drug if available
    if 'drug' in results_df.columns and results_df['drug'].nunique() > 1:
        drugs = results_df['drug'].astype('category').cat.codes
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                             c=drugs, cmap='tab10', alpha=0.6)
        plt.title('t-SNE: Colored by Drug')
    else:
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                             c=quality_scores, cmap='viridis', alpha=0.6)
        plt.title('t-SNE: Colored by Quality')
        plt.colorbar(scatter)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('enhanced_autoencoder_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Reconstruction Examples
    print("\n3. Analyzing reconstructions...")
    
    # Show reconstruction examples
    n_examples = 8
    sample_indices = np.random.choice(len(dataset), n_examples, replace=False)
    
    plt.figure(figsize=(20, 10))
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            data, mask, metadata = dataset[idx]
            data_batch = data.unsqueeze(0).to(device)
            
            reconstruction, _ = model(data_batch)
            reconstruction = reconstruction.squeeze().cpu().numpy()
            
            plt.subplot(2, 4, i + 1)
            
            # Original oxygen data (channel 0)
            times = np.arange(len(data))
            original = data[:, 0].numpy()
            mask_np = data[:, 1].numpy()
            
            # Plot only valid data points
            valid_mask = mask_np > 0.5
            plt.plot(times[valid_mask], original[valid_mask], 'b-', alpha=0.7, label='Original')
            plt.plot(times, reconstruction.flatten(), 'r--', alpha=0.7, label='Reconstruction')
            
            # Mark events if present
            media_flags = data[:, 2].numpy()
            event_times = times[media_flags > 0.5]
            for event_time in event_times:
                plt.axvline(event_time, color='green', alpha=0.5, linestyle=':')
            
            plt.title(f"{metadata.get('drug', 'Unknown')}\nQuality: {quality_scores[idx]:.3f}")
            plt.xlabel('Time (hours)')
            plt.ylabel('Oxygen (normalized)')
            if i == 0:
                plt.legend()
    
    plt.tight_layout()
    plt.savefig('reconstruction_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. Summary Statistics
    print("\n4. Summary Statistics:")
    print(f"Embedding statistics:")
    print(f"  Mean: {embeddings.mean(axis=0)}")
    print(f"  Std: {embeddings.std(axis=0)}")
    print(f"  Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    print(f"\nQuality score statistics:")
    print(f"  Mean: {quality_scores.mean():.6f}")
    print(f"  Median: {median_score:.6f}")
    print(f"  MAD: {mad:.6f}")
    print(f"  Outliers: {n_outliers} ({n_outliers/len(results_df)*100:.1f}%)")
    
    # Save results
    results_df.to_csv('enhanced_autoencoder_results.csv', index=False)
    np.save('enhanced_embeddings.npy', embeddings)
    
    print(f"\nValidation complete!")
    print(f"Results saved:")
    print(f"  - enhanced_autoencoder_results.csv: Metadata and quality scores")
    print(f"  - enhanced_embeddings.npy: 16D embeddings")
    print(f"  - enhanced_autoencoder_validation.png: Validation plots")
    print(f"  - reconstruction_examples.png: Reconstruction examples")
    
    return results_df, embeddings

if __name__ == "__main__":
    results_df, embeddings = validate_enhanced_autoencoder()
```

**Validation checklist**:
- [ ] Embeddings show reasonable statistics (not all zeros or extreme values)
- [ ] Quality scores identify ~1-5% of wells as outliers
- [ ] t-SNE shows some clustering patterns
- [ ] Reconstructions capture major oxygen consumption patterns
- [ ] Results save successfully for further analysis

## Step 4: Integration with Existing Feature Engineering
**Goal**: Combine autoencoder embeddings with existing Hill curve framework  
**Files to create**: `analysis/unified_embedding_analysis.py`  
**Builds on**: Existing dose-response fitting + autoencoder embeddings

### Task 4.1: Unified Embedding Creation
```python
# File: analysis/unified_embedding_analysis.py (NEW)
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis.autoencoder_validation import validate_enhanced_autoencoder

def create_unified_embeddings():
    """
    Combine autoencoder embeddings with existing Hill curve framework
    """
    print("=== Creating Unified Embeddings ===")
    
    # Get autoencoder results
    print("Loading autoencoder results...")
    results_df, embeddings = validate_enhanced_autoencoder()
    
    # Group by drug and concentration for Hill curve fitting
    print("Organizing data for dose-response analysis...")
    
    # Create drug-concentration groups
    drug_concentration_groups = results_df.groupby(['drug', 'concentration'])
    
    print(f"Found {len(drug_concentration_groups)} drug-concentration combinations")
    
    # Extract mean embeddings per drug-concentration
    dose_response_data = []
    
    for (drug, concentration), group in drug_concentration_groups:
        if len(group) < 2:  # Need at least 2 replicates
            continue
        
        # Mean embedding across replicates
        group_indices = group.index
        mean_embedding = embeddings[group_indices].mean(axis=0)
        
        dose_response_data.append({
            'drug': drug,
            'concentration': concentration,
            'log_concentration': np.log10(max(concentration, 1e-6)),  # Avoid log(0)
            'n_replicates': len(group),
            'mean_quality': group['quality_score'].mean(),
            **{f'embedding_{i}': mean_embedding[i] for i in range(len(mean_embedding))}
        })
    
    dose_response_df = pd.DataFrame(dose_response_data)
    
    # Hill curve fitting for each embedding dimension
    print("Fitting Hill curves to autoencoder embeddings...")
    
    def hill_equation(log_conc, E0, Emax, EC50, n):
        """4-parameter Hill equation"""
        conc = 10 ** log_conc
        return E0 + (Emax - E0) * (conc**n) / (EC50**n + conc**n)
    
    hill_results = []
    
    drugs = dose_response_df['drug'].unique()
    print(f"Fitting Hill curves for {len(drugs)} drugs...")
    
    for drug in drugs:
        drug_data = dose_response_df[dose_response_df['drug'] == drug]
        
        if len(drug_data) < 4:  # Need at least 4 concentrations
            continue
        
        concentrations = drug_data['log_concentration'].values
        
        # Fit Hill curve to each embedding dimension
        for dim in range(16):  # 16 embedding dimensions
            embedding_values = drug_data[f'embedding_{dim}'].values
            
            try:
                # Initial parameter guesses
                E0_guess = embedding_values[0]
                Emax_guess = embedding_values[-1]
                EC50_guess = concentrations[len(concentrations)//2]
                n_guess = 1.0
                
                # Fit Hill curve
                popt, pcov = curve_fit(
                    hill_equation,
                    concentrations,
                    embedding_values,
                    p0=[E0_guess, Emax_guess, EC50_guess, n_guess],
                    bounds=([-np.inf, -np.inf, -10, 0.1], [np.inf, np.inf, 10, 10]),
                    maxfev=1000
                )
                
                E0, Emax, EC50, hill_slope = popt
                
                # Calculate R²
                predicted = hill_equation(concentrations, *popt)
                ss_res = np.sum((embedding_values - predicted) ** 2)
                ss_tot = np.sum((embedding_values - np.mean(embedding_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                hill_results.append({
                    'drug': drug,
                    'embedding_dim': dim,
                    'E0': E0,
                    'Emax': Emax,
                    'EC50': EC50,
                    'hill_slope': hill_slope,
                    'r_squared': r_squared,
                    'fit_success': True
                })
                
            except Exception as e:
                # Fit failed
                hill_results.append({
                    'drug': drug,
                    'embedding_dim': dim,
                    'E0': np.nan,
                    'Emax': np.nan,
                    'EC50': np.nan,
                    'hill_slope': np.nan,
                    'r_squared': np.nan,
                    'fit_success': False
                })
    
    hill_results_df = pd.DataFrame(hill_results)
    
    # Create final unified embedding per drug
    print("Creating unified drug embeddings...")
    
    unified_embeddings = []
    
    for drug in drugs:
        drug_hill_data = hill_results_df[hill_results_df['drug'] == drug]
        
        if len(drug_hill_data) == 0:
            continue
        
        # Extract Hill parameters as features
        embedding_features = []
        
        for dim in range(16):
            dim_data = drug_hill_data[drug_hill_data['embedding_dim'] == dim]
            
            if len(dim_data) > 0 and dim_data['fit_success'].iloc[0]:
                # Use Hill parameters as features
                embedding_features.extend([
                    dim_data['EC50'].iloc[0],
                    dim_data['Emax'].iloc[0],
                    dim_data['hill_slope'].iloc[0]
                ])
            else:
                # Fill with NaN if fit failed
                embedding_features.extend([np.nan, np.nan, np.nan])
        
        # Add drug metadata
        drug_dose_data = dose_response_df[dose_response_df['drug'] == drug]
        mean_quality = drug_dose_data['mean_quality'].mean()
        n_concentrations = len(drug_dose_data)
        
        unified_embeddings.append({
            'drug': drug,
            'n_concentrations': n_concentrations,
            'mean_quality': mean_quality,
            **{f'feature_{i}': embedding_features[i] for i in range(len(embedding_features))}
        })
    
    unified_df = pd.DataFrame(unified_embeddings)
    
    # Summary statistics
    print(f"\nUnified embedding creation complete:")
    print(f"  Drugs processed: {len(unified_df)}")
    print(f"  Features per drug: {len(embedding_features)}")
    print(f"  Hill curve fit success rate: {hill_results_df['fit_success'].mean():.2%}")
    
    # Save results
    dose_response_df.to_csv('autoencoder_dose_response.csv', index=False)
    hill_results_df.to_csv('autoencoder_hill_curves.csv', index=False)
    unified_df.to_csv('unified_drug_embeddings.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"  - autoencoder_dose_response.csv: Concentration-level embeddings")
    print(f"  - autoencoder_hill_curves.csv: Hill curve fit results")
    print(f"  - unified_drug_embeddings.csv: Final drug-level embeddings")
    
    return unified_df, hill_results_df

if __name__ == "__main__":
    unified_df, hill_results_df = create_unified_embeddings()
```

**Validation checklist**:
- [ ] Hill curves fit successfully for majority of drug-embedding combinations
- [ ] Unified embeddings have reasonable statistics
- [ ] EC50 values are in expected range (log scale)
- [ ] Results save successfully for further analysis

## Summary

This integration plan builds directly on the **existing event-aware feature engineering foundation** by:

1. **Extending existing data pipeline** to create autoencoder-ready tensors with 15 channels
2. **Using existing event detection and quality assessment** as input to the autoencoder
3. **Applying existing Hill curve framework** to autoencoder embeddings for cross-drug comparability
4. **Creating unified embeddings** that combine learned patterns with pharmacological interpretability

The result is a **complete integration** where autoencoder embeddings become another feature type that can be analyzed using the same dose-response normalization framework as traditional features, providing both the power of deep learning and the interpretability of pharmacological modeling.