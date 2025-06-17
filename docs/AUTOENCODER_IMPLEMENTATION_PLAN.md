# Autoencoder Implementation Plan
## Foundation Component for Oxygen Data Interpolation

This plan provides detailed, bite-sized steps for implementing the event-aware autoencoder as the foundation for oxygen data interpolation. Each step is designed to be accomplishable by a junior engineer with guidance.

## Prerequisites

- Python environment with PyTorch, pandas, numpy, scikit-learn
- Database access via `DATABASE_URL` from project CLAUDE.md
- Basic understanding of autoencoders and time series data

## Phase 1: Basic Event-Aware Autoencoder

### Step 1: Data Pipeline Setup
**Goal**: Load and preprocess oxygen consumption data with experimental structure  
**Time estimate**: 1-2 days  
**Files to create**: `data_pipeline.py`

#### Task 1.1: Database Connection and Loading
```python
# File: data_pipeline.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

def load_oxygen_data():
    """
    Load oxygen consumption data with quality filters
    
    Returns:
        pd.DataFrame with columns: [well_id, drug, concentration, elapsed_time, oxygen_value, plate_id]
    """
    # Use DATABASE_URL from environment (already set up in project)
    engine = create_engine(os.getenv('DATABASE_URL'))
    
    # Query with quality filters from O2_REALTIME_DATA.md:
    # - is_excluded = false
    # - experiment duration >= 14 days  
    # - drugs with >= 4 concentrations
    query = """
    SELECT well_id, drug, concentration, elapsed_time, oxygen_value, plate_id
    FROM oxygen_data 
    WHERE is_excluded = false 
    AND experiment_duration >= 14 * 24  -- 14 days in hours
    AND drug IN (
        SELECT drug FROM oxygen_data 
        GROUP BY drug 
        HAVING COUNT(DISTINCT concentration) >= 4
    )
    ORDER BY plate_id, well_id, elapsed_time
    """
    
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} measurements from {df['well_id'].nunique()} wells")
    print(f"Covering {df['drug'].nunique()} drugs across {df['plate_id'].nunique()} plates")
    
    return df

# Test the function
if __name__ == "__main__":
    data = load_oxygen_data()
    print(data.head())
```

#### Task 1.2: Control Well Identification
```python
def identify_control_wells(df):
    """
    Identify control wells on each plate
    
    Args:
        df: DataFrame from load_oxygen_data()
    
    Returns:
        pd.DataFrame with control wells flagged
    """
    # Control wells: concentration = 0 or drug = 'DMSO'
    control_mask = (df['concentration'] == 0) | (df['drug'] == 'DMSO')
    df['is_control'] = control_mask
    
    # Verify expectation of ~73 control wells per plate
    controls_per_plate = df[df['is_control']].groupby('plate_id')['well_id'].nunique()
    print(f"Control wells per plate: mean={controls_per_plate.mean():.1f}, std={controls_per_plate.std():.1f}")
    
    if controls_per_plate.mean() < 50:
        print("WARNING: Fewer control wells than expected (~73). Check control identification logic.")
    
    return df

# Test the function
if __name__ == "__main__":
    data = load_oxygen_data()
    data = identify_control_wells(data)
```

#### Task 1.3: Event Detection Functions
```python
def detect_media_changes(plate_data):
    """
    Detect media change events using variance analysis
    
    Args:
        plate_data: DataFrame for single plate
    
    Returns:
        list of timestamps where media changes occurred
    """
    # Group by timepoint, calculate variance across all wells
    time_variance = plate_data.groupby('elapsed_time')['oxygen_value'].var()
    
    # Find spikes in variance (media changes cause sudden jumps)
    variance_threshold = time_variance.quantile(0.95)  # Top 5% of variance
    spike_times = time_variance[time_variance > variance_threshold].index.tolist()
    
    # Filter spikes that are too close together (within 12h)
    filtered_spikes = []
    for spike in spike_times:
        if not filtered_spikes or (spike - filtered_spikes[-1]) > 12:
            filtered_spikes.append(spike)
    
    return filtered_spikes

def detect_dosing_time(plate_data):
    """
    Detect when drugs were dosed by finding control/treatment divergence
    
    Args:
        plate_data: DataFrame for single plate
        
    Returns:
        float: estimated dosing time in hours
    """
    # Separate control and treatment wells
    controls = plate_data[plate_data['is_control']]
    treatments = plate_data[~plate_data['is_control']]
    
    # Calculate median trajectory for each group
    control_trajectory = controls.groupby('elapsed_time')['oxygen_value'].median()
    treatment_trajectory = treatments.groupby('elapsed_time')['oxygen_value'].median()
    
    # Find time when trajectories start to diverge
    common_times = control_trajectory.index.intersection(treatment_trajectory.index)
    divergence = abs(control_trajectory[common_times] - treatment_trajectory[common_times])
    
    # Dosing time = when divergence exceeds baseline (typically 24-48h)
    baseline_divergence = divergence[:48].mean()  # First 48h baseline
    divergence_threshold = baseline_divergence * 2
    
    dosing_candidates = common_times[divergence > divergence_threshold]
    dosing_time = dosing_candidates[0] if len(dosing_candidates) > 0 else 48.0
    
    return dosing_time

# Test functions
if __name__ == "__main__":
    data = load_oxygen_data()
    data = identify_control_wells(data)
    
    # Test on first plate
    first_plate = data[data['plate_id'] == data['plate_id'].iloc[0]]
    media_changes = detect_media_changes(first_plate)
    dosing_time = detect_dosing_time(first_plate)
    
    print(f"Media changes detected at: {media_changes}")
    print(f"Dosing time estimated at: {dosing_time} hours")
```

**Validation checklist for Step 1**:
- [ ] `load_oxygen_data()` returns expected number of wells and drugs
- [ ] Control wells identified (~73 per plate)
- [ ] Media change detection finds reasonable number of events (3-5 per plate)
- [ ] Dosing time detection returns values in 24-48h range

### Step 2: Data Formatting for Autoencoder
**Goal**: Convert raw timeseries into multi-channel tensor format  
**Time estimate**: 1-2 days  
**Files to create**: `data_formatter.py`

#### Task 2.1: Time Series Resampling
```python
# File: data_formatter.py
import torch
import numpy as np
from scipy import interpolate

def resample_to_hourly(well_data, target_length=336):
    """
    Resample irregular time series to regular 1-hour grid
    
    Args:
        well_data: DataFrame for single well
        target_length: Number of hours (default 336 = 2 weeks)
    
    Returns:
        dict with resampled data and metadata
    """
    # Sort by time
    well_data = well_data.sort_values('elapsed_time')
    
    # Create target time grid (0 to target_length hours)
    target_times = np.arange(0, target_length, 1.0)
    
    # Interpolate oxygen values to regular grid
    if len(well_data) < 2:
        # Handle edge case of insufficient data
        oxygen_resampled = np.full(target_length, np.nan)
        mask = np.zeros(target_length)
    else:
        # Linear interpolation
        interp_func = interpolate.interp1d(
            well_data['elapsed_time'], 
            well_data['oxygen_value'],
            kind='linear', 
            bounds_error=False, 
            fill_value=np.nan
        )
        oxygen_resampled = interp_func(target_times)
        
        # Create mask: 1 for original data points, 0 for interpolated
        mask = np.zeros(target_length)
        for orig_time in well_data['elapsed_time']:
            closest_idx = np.argmin(np.abs(target_times - orig_time))
            if np.abs(target_times[closest_idx] - orig_time) < 0.5:  # Within 30 min
                mask[closest_idx] = 1
    
    return {
        'oxygen_values': oxygen_resampled,
        'mask': mask,
        'original_times': well_data['elapsed_time'].values,
        'target_times': target_times
    }

# Test resampling
if __name__ == "__main__":
    from data_pipeline import load_oxygen_data, identify_control_wells
    
    data = load_oxygen_data()
    data = identify_control_wells(data)
    
    # Test on first well
    first_well = data[data['well_id'] == data['well_id'].iloc[0]]
    resampled = resample_to_hourly(first_well)
    
    print(f"Original points: {len(first_well)}")
    print(f"Resampled points: {len(resampled['oxygen_values'])}")
    print(f"Valid data fraction: {resampled['mask'].mean():.3f}")
```

#### Task 2.2: Multi-Channel Tensor Creation
```python
def create_phase_indicators(target_times, dosing_time, media_changes):
    """
    Create phase indicator channels
    
    Args:
        target_times: Array of time points
        dosing_time: When dosing occurred
        media_changes: List of media change times
        
    Returns:
        np.array of shape [len(target_times), 4] for phases
    """
    phases = np.zeros((len(target_times), 4))
    
    for i, t in enumerate(target_times):
        if t < dosing_time:
            phases[i, 0] = 1  # pre_dosing
        elif t < dosing_time + 72:  # 3 days post-dosing
            phases[i, 1] = 1  # early_response
        elif t < dosing_time + 240:  # 10 days post-dosing  
            phases[i, 2] = 1  # sustained
        else:
            phases[i, 3] = 1  # late
    
    return phases

def create_event_flags(target_times, dosing_time, media_changes):
    """
    Create event flag channels
    
    Returns:
        tuple of (media_flags, dosing_flags)
    """
    media_flags = np.zeros(len(target_times))
    dosing_flags = np.zeros(len(target_times))
    
    # Mark media change times
    for media_time in media_changes:
        closest_idx = np.argmin(np.abs(target_times - media_time))
        if np.abs(target_times[closest_idx] - media_time) < 1.0:  # Within 1 hour
            media_flags[closest_idx] = 1
    
    # Mark dosing time
    closest_idx = np.argmin(np.abs(target_times - dosing_time))
    if np.abs(target_times[closest_idx] - dosing_time) < 1.0:
        dosing_flags[closest_idx] = 1
    
    return media_flags, dosing_flags

def create_multichannel_input(well_data, dosing_time, media_changes, controls_trajectory):
    """
    Convert single well to multi-channel tensor
    
    Args:
        well_data: DataFrame for single well
        dosing_time: Dosing time for this plate
        media_changes: List of media change times
        controls_trajectory: Median control trajectory for normalization
        
    Returns:
        torch.Tensor of shape [time_points, 7]
    """
    # Resample to hourly grid
    resampled = resample_to_hourly(well_data)
    target_times = resampled['target_times']
    
    # Normalize oxygen values to [-1, 1] range and subtract controls
    oxygen_raw = resampled['oxygen_values']
    oxygen_normalized = (oxygen_raw - controls_trajectory) / 50.0  # Rough normalization
    oxygen_normalized = np.clip(oxygen_normalized, -1, 1)
    
    # Create event flags
    media_flags, dosing_flags = create_event_flags(target_times, dosing_time, media_changes)
    
    # Create phase indicators
    phase_indicators = create_phase_indicators(target_times, dosing_time, media_changes)
    
    # Stack all channels
    channels = np.column_stack([
        oxygen_normalized,           # Channel 0: normalized oxygen
        resampled['mask'],          # Channel 1: data validity mask
        media_flags,                # Channel 2: media change events
        dosing_flags,               # Channel 3: dosing event
        phase_indicators            # Channels 4-7: phase indicators
    ])
    
    # Handle NaN values
    channels = np.nan_to_num(channels, nan=0.0)
    
    return torch.FloatTensor(channels)

# Test multi-channel creation
if __name__ == "__main__":
    from data_pipeline import load_oxygen_data, identify_control_wells, detect_media_changes, detect_dosing_time
    
    data = load_oxygen_data()
    data = identify_control_wells(data)
    
    # Process first plate
    first_plate_id = data['plate_id'].iloc[0]
    plate_data = data[data['plate_id'] == first_plate_id]
    
    # Get events for this plate
    media_changes = detect_media_changes(plate_data)
    dosing_time = detect_dosing_time(plate_data)
    
    # Get control trajectory for normalization
    controls = plate_data[plate_data['is_control']]
    controls_resampled = controls.groupby('elapsed_time')['oxygen_value'].median()
    
    # Process first well
    first_well = plate_data[plate_data['well_id'] == plate_data['well_id'].iloc[0]]
    multichannel = create_multichannel_input(first_well, dosing_time, media_changes, controls_resampled)
    
    print(f"Multi-channel tensor shape: {multichannel.shape}")
    print(f"Channels: oxygen, mask, media_changes, dosing, pre_dosing, early, sustained, late")
    print(f"Non-zero values per channel: {(multichannel != 0).sum(dim=0)}")
```

**Validation checklist for Step 2**:
- [ ] Resampling produces regular 1-hour grid
- [ ] Multi-channel tensor has shape [336, 7]
- [ ] Event flags mark reasonable number of timepoints
- [ ] Phase indicators sum to 1 at each timepoint
- [ ] Oxygen values are normalized to [-1, 1] range

### Step 3: Basic Autoencoder Implementation
**Goal**: Implement and test basic autoencoder architecture  
**Time estimate**: 2-3 days  
**Files to create**: `autoencoder.py`

#### Task 3.1: Model Architecture
```python
# File: autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExperimentalEventAwareAutoencoder(nn.Module):
    """
    Multi-channel autoencoder for experimental oxygen consumption data
    
    Input: [batch_size, time_points, 7_channels]
    Output: [batch_size, time_points, 1] + [batch_size, latent_dim]
    """
    
    def __init__(self, latent_dim=16, input_length=336):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_length = input_length
        
        # Encoder: 1D convolutions for temporal patterns
        self.encoder = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv1d(7, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            # Block 2: Downsample + increase channels  
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            # Block 3: Further compression
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            # Global pooling + dense layers
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder: Mirror architecture with upsampling
        self.decoder_start = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        self.decoder_conv = nn.Sequential(
            # Start with small spatial dimension
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Final reconstruction (only oxygen channel)
            nn.Conv1d(64, 1, kernel_size=7, padding=3)
        )
    
    def encode(self, x):
        """
        Encode input to latent representation
        
        Args:
            x: [batch_size, time_points, channels]
        
        Returns:
            latent: [batch_size, latent_dim]
        """
        # Conv1d expects [batch, channels, time]
        x = x.transpose(1, 2)
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to oxygen timeseries
        
        Args:
            z: [batch_size, latent_dim]
            
        Returns:
            reconstruction: [batch_size, time_points, 1]
        """
        # Dense layers
        h = self.decoder_start(z)  # [batch, 256]
        
        # Reshape for convolution [batch, channels, small_time]
        h = h.unsqueeze(-1)  # [batch, 256, 1]
        
        # Transpose convolutions
        recon = self.decoder_conv(h)  # [batch, 1, time]
        
        # Resize to exact input length and transpose back
        recon = F.interpolate(recon, size=self.input_length, mode='linear', align_corners=False)
        return recon.transpose(1, 2)  # [batch, time, 1]
    
    def forward(self, x):
        """
        Full forward pass
        
        Args:
            x: [batch_size, time_points, 7_channels]
            
        Returns:
            reconstruction: [batch_size, time_points, 1]
            latent: [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def quality_score(self, x, mask):
        """
        Calculate reconstruction error for quality assessment
        
        Args:
            x: [batch_size, time_points, 7_channels]
            mask: [batch_size, time_points] - data validity mask
            
        Returns:
            error: [batch_size] - per-sample reconstruction error
        """
        reconstruction, _ = self.forward(x)
        
        # Extract original oxygen values (channel 0)
        original_oxygen = x[:, :, 0:1]  # [batch, time, 1]
        
        # Apply mask to ignore missing data
        mask_3d = mask.unsqueeze(-1)  # [batch, time, 1]
        masked_recon = reconstruction * mask_3d
        masked_original = original_oxygen * mask_3d
        
        # Calculate MSE per sample
        mse = F.mse_loss(masked_recon, masked_original, reduction='none')
        error = mse.mean(dim=(1, 2))  # [batch_size]
        
        return error

# Test model architecture
if __name__ == "__main__":
    # Create dummy data
    batch_size = 4
    time_points = 336
    channels = 7
    
    dummy_input = torch.randn(batch_size, time_points, channels)
    dummy_mask = torch.ones(batch_size, time_points)
    
    # Test model
    model = ExperimentalEventAwareAutoencoder(latent_dim=16)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    reconstruction, latent = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Quality score
    quality = model.quality_score(dummy_input, dummy_mask)
    print(f"Quality scores shape: {quality.shape}")
```

#### Task 3.2: Training Functions
```python
def train_autoencoder(model, train_loader, val_loader, epochs=50, device='cpu'):
    """
    Training loop for autoencoder
    
    Args:
        model: ExperimentalEventAwareAutoencoder instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, mask) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, latent = model(data)
            
            # Loss calculation with mask weighting
            original_oxygen = data[:, :, 0:1]  # Extract oxygen channel
            mask_3d = mask.unsqueeze(-1)
            
            # Reconstruction loss (only on valid data points)
            recon_loss = F.mse_loss(reconstruction * mask_3d, original_oxygen * mask_3d)
            
            # L2 regularization on latent embeddings
            l2_reg = 1e-4 * torch.norm(latent, p=2, dim=1).mean()
            
            total_loss = recon_loss + l2_reg
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, mask in val_loader:
                data, mask = data.to(device), mask.to(device)
                
                reconstruction, latent = model(data)
                original_oxygen = data[:, :, 0:1]
                mask_3d = mask.unsqueeze(-1)
                
                recon_loss = F.mse_loss(reconstruction * mask_3d, original_oxygen * mask_3d)
                l2_reg = 1e-4 * torch.norm(latent, p=2, dim=1).mean()
                
                val_loss += (recon_loss + l2_reg).item()
        
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
            # Save best model
            torch.save(model.state_dict(), 'best_autoencoder.pth')
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

# Test training function with dummy data
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy dataset
    n_samples = 1000
    dummy_data = torch.randn(n_samples, 336, 7)
    dummy_masks = torch.ones(n_samples, 336)
    
    dataset = TensorDataset(dummy_data, dummy_masks)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Test training
    model = ExperimentalEventAwareAutoencoder()
    history = train_autoencoder(model, train_loader, val_loader, epochs=5)
    
    print(f"Training completed. Best val loss: {history['best_val_loss']:.6f}")
```

**Validation checklist for Step 3**:
- [ ] Model architecture runs without errors on dummy data
- [ ] Input/output tensor shapes are correct
- [ ] Training loop completes without crashes
- [ ] Loss decreases over training epochs
- [ ] Model checkpoints are saved correctly

### Step 4: Data Loading and Training Pipeline
**Goal**: Create end-to-end pipeline from database to trained model  
**Time estimate**: 2-3 days  
**Files to create**: `train_pipeline.py`, update `data_formatter.py`

#### Task 4.1: Dataset Class
```python
# File: train_pipeline.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from data_pipeline import load_oxygen_data, identify_control_wells, detect_media_changes, detect_dosing_time
from data_formatter import create_multichannel_input, resample_to_hourly

class OxygenDataset(Dataset):
    """
    PyTorch Dataset for oxygen consumption data
    """
    
    def __init__(self, max_wells_per_drug=None, target_length=336):
        """
        Args:
            max_wells_per_drug: Limit wells per drug for testing (None = all wells)
            target_length: Length of resampled timeseries
        """
        print("Loading oxygen consumption data...")
        
        # Load and process data
        self.raw_data = load_oxygen_data()
        self.raw_data = identify_control_wells(self.raw_data)
        
        print(f"Loaded {len(self.raw_data)} measurements")
        print(f"Drugs: {self.raw_data['drug'].nunique()}")
        print(f"Plates: {self.raw_data['plate_id'].nunique()}")
        print(f"Wells: {self.raw_data['well_id'].nunique()}")
        
        # Process each plate to extract events and create tensors
        self.processed_wells = []
        self.well_metadata = []
        
        for plate_id in self.raw_data['plate_id'].unique():
            print(f"Processing plate {plate_id}...")
            plate_data = self.raw_data[self.raw_data['plate_id'] == plate_id]
            
            # Detect events for this plate
            media_changes = detect_media_changes(plate_data)
            dosing_time = detect_dosing_time(plate_data)
            
            # Get control trajectory for normalization
            controls = plate_data[plate_data['is_control']]
            if len(controls) > 0:
                # Resample control trajectory
                control_trajectory = self._get_control_trajectory(controls, target_length)
            else:
                # Fallback if no controls
                control_trajectory = np.zeros(target_length)
            
            # Process each well on this plate
            wells_on_plate = plate_data['well_id'].unique()
            
            # Limit wells per drug if specified (for testing)
            if max_wells_per_drug is not None:
                drug_counts = {}
                filtered_wells = []
                for well_id in wells_on_plate:
                    well_data = plate_data[plate_data['well_id'] == well_id]
                    drug = well_data['drug'].iloc[0]
                    
                    if drug_counts.get(drug, 0) < max_wells_per_drug:
                        filtered_wells.append(well_id)
                        drug_counts[drug] = drug_counts.get(drug, 0) + 1
                
                wells_on_plate = filtered_wells
            
            for well_id in wells_on_plate:
                well_data = plate_data[plate_data['well_id'] == well_id]
                
                if len(well_data) < 10:  # Skip wells with too little data
                    continue
                
                # Create multi-channel tensor
                try:
                    tensor = create_multichannel_input(
                        well_data, dosing_time, media_changes, control_trajectory
                    )
                    
                    # Extract mask (channel 1)
                    mask = tensor[:, 1]
                    
                    self.processed_wells.append(tensor)
                    self.well_metadata.append({
                        'well_id': well_id,
                        'plate_id': plate_id,
                        'drug': well_data['drug'].iloc[0],
                        'concentration': well_data['concentration'].iloc[0],
                        'is_control': well_data['is_control'].iloc[0],
                        'data_points': len(well_data),
                        'valid_fraction': mask.mean().item()
                    })
                    
                except Exception as e:
                    print(f"Error processing well {well_id}: {e}")
                    continue
        
        print(f"Successfully processed {len(self.processed_wells)} wells")
        
        # Create metadata DataFrame for analysis
        self.metadata_df = pd.DataFrame(self.well_metadata)
        print("\nDataset summary:")
        print(f"Wells per drug: {self.metadata_df.groupby('drug').size().describe()}")
        print(f"Valid data fraction: {self.metadata_df['valid_fraction'].describe()}")
    
    def _get_control_trajectory(self, controls, target_length):
        """Helper to create median control trajectory"""
        # Combine all control wells and resample
        all_controls = []
        for well_id in controls['well_id'].unique():
            well_controls = controls[controls['well_id'] == well_id]
            resampled = resample_to_hourly(well_controls, target_length)
            if not np.isnan(resampled['oxygen_values']).all():
                all_controls.append(resampled['oxygen_values'])
        
        if all_controls:
            # Calculate median across control wells
            control_matrix = np.column_stack(all_controls)
            return np.nanmedian(control_matrix, axis=1)
        else:
            return np.zeros(target_length)
    
    def __len__(self):
        return len(self.processed_wells)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (input_tensor, mask_tensor)
        """
        tensor = self.processed_wells[idx]
        
        # Input is all 7 channels
        input_tensor = tensor  # [time_points, 7_channels]
        
        # Mask is channel 1
        mask_tensor = tensor[:, 1]  # [time_points]
        
        return input_tensor, mask_tensor
    
    def get_metadata(self):
        """Return metadata DataFrame"""
        return self.metadata_df
    
    def train_val_split(self, val_fraction=0.2, random_state=42):
        """
        Split by drug to prevent leakage
        
        Returns:
            tuple: (train_indices, val_indices)
        """
        np.random.seed(random_state)
        
        # Get unique drugs
        drugs = self.metadata_df['drug'].unique()
        n_val_drugs = int(len(drugs) * val_fraction)
        
        # Randomly select validation drugs
        val_drugs = np.random.choice(drugs, size=n_val_drugs, replace=False)
        
        # Split indices
        val_mask = self.metadata_df['drug'].isin(val_drugs)
        val_indices = self.metadata_df[val_mask].index.tolist()
        train_indices = self.metadata_df[~val_mask].index.tolist()
        
        print(f"Train drugs: {len(drugs) - n_val_drugs}")
        print(f"Val drugs: {n_val_drugs}")
        print(f"Train wells: {len(train_indices)}")
        print(f"Val wells: {len(val_indices)}")
        
        return train_indices, val_indices

# Test dataset creation
if __name__ == "__main__":
    # Create small dataset for testing
    dataset = OxygenDataset(max_wells_per_drug=10)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    sample_input, sample_mask = dataset[0]
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")
    
    # Test train/val split
    train_idx, val_idx = dataset.train_val_split()
    print(f"Split created successfully")
```

#### Task 4.2: End-to-End Training Script
```python
# File: train_full_pipeline.py
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from train_pipeline import OxygenDataset
from autoencoder import ExperimentalEventAwareAutoencoder, train_autoencoder

def main():
    """
    Complete training pipeline
    """
    print("Starting full autoencoder training pipeline...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("\n=== Creating Dataset ===")
    dataset = OxygenDataset(max_wells_per_drug=50)  # Limit for faster testing
    
    # Train/validation split
    train_indices, val_indices = dataset.train_val_split(val_fraction=0.2)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = ExperimentalEventAwareAutoencoder(latent_dim=16)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n=== Training Model ===")
    history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['val_losses'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    # Load best model and analyze
    model.load_state_dict(torch.load('best_autoencoder.pth'))
    model = model.to(device)
    model.eval()
    
    # Extract embeddings for analysis
    print("\n=== Analyzing Results ===")
    all_embeddings = []
    all_metadata = []
    
    with torch.no_grad():
        for data, mask in val_loader:
            data = data.to(device)
            _, embeddings = model(data)
            all_embeddings.append(embeddings.cpu())
            
            # Get metadata for these samples (this is simplified)
            batch_size = data.shape[0]
            all_metadata.extend(['unknown'] * batch_size)
    
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Extracted {embeddings_tensor.shape[0]} embeddings of dimension {embeddings_tensor.shape[1]}")
    
    # Basic analysis
    print(f"Embedding statistics:")
    print(f"  Mean: {embeddings_tensor.mean(dim=0)}")
    print(f"  Std: {embeddings_tensor.std(dim=0)}")
    print(f"  Range: [{embeddings_tensor.min():.3f}, {embeddings_tensor.max():.3f}]")
    
    # Quality analysis
    quality_scores = []
    with torch.no_grad():
        for data, mask in val_loader:
            data, mask = data.to(device), mask.to(device)
            scores = model.quality_score(data, mask)
            quality_scores.append(scores.cpu())
    
    quality_tensor = torch.cat(quality_scores, dim=0)
    
    plt.subplot(1, 2, 2)
    plt.hist(quality_tensor.numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Quality Score Distribution')
    plt.axvline(quality_tensor.median(), color='red', linestyle='--', label=f'Median: {quality_tensor.median():.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Model saved as: best_autoencoder.pth")
    print(f"Analysis plot saved as: training_analysis.png")
    
    return model, history, dataset

if __name__ == "__main__":
    model, history, dataset = main()
```

**Validation checklist for Step 4**:
- [ ] Dataset loads without errors and reports expected number of wells
- [ ] Train/validation split separates drugs properly (no leakage)
- [ ] Training completes successfully and saves model checkpoint
- [ ] Loss curves show decreasing trend
- [ ] Embeddings have reasonable statistics (not all zeros/extreme values)

### Step 5: Quality Control and Validation
**Goal**: Validate autoencoder learns meaningful patterns and detects outliers  
**Time estimate**: 1-2 days  
**Files to create**: `quality_analysis.py`

#### Task 5.1: Comprehensive Quality Analysis
```python
# File: quality_analysis.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from autoencoder import ExperimentalEventAwareAutoencoder
from train_pipeline import OxygenDataset

def analyze_reconstruction_quality(model, dataset, device='cpu'):
    """
    Analyze reconstruction quality across all wells
    """
    model = model.to(device)
    model.eval()
    
    reconstruction_errors = []
    metadata = dataset.get_metadata()
    
    print("Calculating reconstruction errors...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data, mask = dataset[i]
            data = data.unsqueeze(0).to(device)  # Add batch dimension
            mask = mask.unsqueeze(0).to(device)
            
            error = model.quality_score(data, mask)
            reconstruction_errors.append(error.item())
    
    # Add errors to metadata
    metadata['reconstruction_error'] = reconstruction_errors
    
    # Calculate outlier threshold (3 * MAD)
    median_error = np.median(reconstruction_errors)
    mad = np.median(np.abs(reconstruction_errors - median_error))
    outlier_threshold = median_error + 3 * mad
    
    metadata['is_outlier'] = metadata['reconstruction_error'] > outlier_threshold
    
    print(f"Reconstruction error statistics:")
    print(f"  Median: {median_error:.6f}")
    print(f"  MAD: {mad:.6f}")
    print(f"  Outlier threshold: {outlier_threshold:.6f}")
    print(f"  Outliers detected: {metadata['is_outlier'].sum()} / {len(metadata)} ({metadata['is_outlier'].mean()*100:.1f}%)")
    
    # Analyze outliers by drug and other factors
    print(f"\nOutlier analysis:")
    outlier_by_drug = metadata.groupby('drug')['is_outlier'].agg(['count', 'sum', 'mean'])
    outlier_by_drug['outlier_rate'] = outlier_by_drug['mean']
    outlier_by_drug = outlier_by_drug.sort_values('outlier_rate', ascending=False)
    
    print(f"Top 10 drugs by outlier rate:")
    print(outlier_by_drug.head(10))
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(reconstruction_errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(median_error, color='blue', linestyle='--', label=f'Median: {median_error:.4f}')
    plt.axvline(outlier_threshold, color='red', linestyle='--', label=f'Outlier threshold: {outlier_threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    # Error vs data quality
    plt.scatter(metadata['valid_fraction'], metadata['reconstruction_error'], alpha=0.6)
    plt.xlabel('Valid Data Fraction')
    plt.ylabel('Reconstruction Error')
    plt.title('Error vs Data Quality')
    
    plt.subplot(1, 3, 3)
    # Error by control vs treatment
    control_errors = metadata[metadata['is_control']]['reconstruction_error']
    treatment_errors = metadata[~metadata['is_control']]['reconstruction_error']
    
    plt.boxplot([control_errors, treatment_errors], labels=['Control', 'Treatment'])
    plt.ylabel('Reconstruction Error')
    plt.title('Error by Well Type')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('reconstruction_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return metadata

def analyze_embeddings(model, dataset, device='cpu'):
    """
    Analyze learned embeddings
    """
    model = model.to(device)
    model.eval()
    
    embeddings = []
    metadata = dataset.get_metadata()
    
    print("Extracting embeddings...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data, mask = dataset[i]
            data = data.unsqueeze(0).to(device)
            
            _, embedding = model(data)
            embeddings.append(embedding.squeeze().cpu().numpy())
    
    embeddings = np.array(embeddings)  # [n_wells, latent_dim]
    
    print(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"Embedding statistics:")
    print(f"  Mean: {embeddings.mean(axis=0)}")
    print(f"  Std: {embeddings.std(axis=0)}")
    
    # PCA analysis
    pca = PCA()
    pca_embeddings = pca.fit_transform(embeddings)
    
    print(f"PCA explained variance ratio (first 8 components): {pca.explained_variance_ratio_[:8]}")
    print(f"Cumulative variance (first 8 components): {pca.explained_variance_ratio_[:8].cumsum()}")
    
    # t-SNE visualization
    print("Computing t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Clustering analysis
    if len(metadata['drug'].unique()) > 1:
        # Group by drug for clustering
        drug_labels = metadata['drug'].astype('category').cat.codes
        silhouette = silhouette_score(embeddings, drug_labels)
        print(f"Silhouette score (grouped by drug): {silhouette:.4f}")
    
    # Visualization
    plt.figure(figsize=(20, 5))
    
    # PCA plot
    plt.subplot(1, 4, 1)
    scatter = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
                         c=metadata['is_control'], cmap='coolwarm', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Control vs Treatment')
    plt.colorbar(scatter, label='Is Control')
    
    # t-SNE by control status
    plt.subplot(1, 4, 2)
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                         c=metadata['is_control'], cmap='coolwarm', alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE: Control vs Treatment')
    plt.colorbar(scatter, label='Is Control')
    
    # t-SNE by concentration (log scale)
    plt.subplot(1, 4, 3)
    conc_log = np.log10(metadata['concentration'] + 1e-6)  # Add small value for log
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                         c=conc_log, cmap='viridis', alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE: Log Concentration')
    plt.colorbar(scatter, label='Log Concentration')
    
    # Embedding dimension analysis
    plt.subplot(1, 4, 4)
    embedding_stds = embeddings.std(axis=0)
    plt.bar(range(len(embedding_stds)), embedding_stds)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Embedding Dimension Usage')
    
    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return embeddings, pca_embeddings, tsne_embeddings

def validate_event_responses(model, dataset, device='cpu'):
    """
    Validate that model captures event-specific patterns
    """
    model = model.to(device)
    model.eval()
    
    print("Analyzing event response patterns...")
    
    # Sample a few wells for detailed analysis
    sample_indices = np.random.choice(len(dataset), size=min(20, len(dataset)), replace=False)
    
    plt.figure(figsize=(20, 10))
    
    for i, idx in enumerate(sample_indices[:12]):  # Show first 12
        data, mask = dataset[idx]
        metadata = dataset.get_metadata().iloc[idx]
        
        # Get reconstruction
        with torch.no_grad():
            data_batch = data.unsqueeze(0).to(device)
            reconstruction, embedding = model(data_batch)
            reconstruction = reconstruction.squeeze().cpu().numpy()
        
        data_np = data.numpy()
        
        plt.subplot(3, 4, i + 1)
        
        # Plot original vs reconstruction (oxygen channel only)
        time_points = np.arange(len(data_np))
        original_oxygen = data_np[:, 0]  # Channel 0 = oxygen
        mask_np = data_np[:, 1]  # Channel 1 = mask
        media_flags = data_np[:, 2]  # Channel 2 = media changes
        
        # Plot original data (only valid points)
        valid_mask = mask_np > 0.5
        plt.plot(time_points[valid_mask], original_oxygen[valid_mask], 'b-', alpha=0.7, label='Original')
        
        # Plot reconstruction
        plt.plot(time_points, reconstruction.flatten(), 'r--', alpha=0.7, label='Reconstruction')
        
        # Mark media change events
        media_events = time_points[media_flags > 0.5]
        for event_time in media_events:
            plt.axvline(event_time, color='green', alpha=0.5, linestyle=':')
        
        plt.title(f"Well {metadata['well_id']}\n{metadata['drug']}, {metadata['concentration']:.3f} μM")
        plt.xlabel('Time (hours)')
        plt.ylabel('Oxygen (normalized)')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('event_response_validation.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Complete quality analysis pipeline
    """
    print("Starting quality analysis...")
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExperimentalEventAwareAutoencoder(latent_dim=16)
    
    try:
        model.load_state_dict(torch.load('best_autoencoder.pth', map_location=device))
        print("Loaded trained model successfully")
    except FileNotFoundError:
        print("ERROR: best_autoencoder.pth not found. Please train model first.")
        return
    
    # Load dataset
    dataset = OxygenDataset(max_wells_per_drug=100)  # Limit for analysis
    
    # Run analyses
    print("\n=== 1. Reconstruction Quality Analysis ===")
    metadata_with_errors = analyze_reconstruction_quality(model, dataset, device)
    
    print("\n=== 2. Embedding Analysis ===")
    embeddings, pca_embeddings, tsne_embeddings = analyze_embeddings(model, dataset, device)
    
    print("\n=== 3. Event Response Validation ===")
    validate_event_responses(model, dataset, device)
    
    # Save results
    metadata_with_errors.to_csv('quality_analysis_results.csv', index=False)
    np.save('embeddings.npy', embeddings)
    
    print("\n=== Analysis Complete ===")
    print("Results saved:")
    print("  - quality_analysis_results.csv: Metadata with quality scores")
    print("  - embeddings.npy: Extracted embeddings")
    print("  - reconstruction_quality_analysis.png: Quality analysis plots")
    print("  - embedding_analysis.png: Embedding visualization")
    print("  - event_response_validation.png: Event response examples")

if __name__ == "__main__":
    main()
```

**Validation checklist for Step 5**:
- [ ] Reconstruction errors show reasonable distribution (not all zeros or very high)
- [ ] Outlier detection identifies ~1-5% of wells as problematic
- [ ] t-SNE visualization shows some clustering by experimental factors
- [ ] Model reconstructions capture major patterns and event responses
- [ ] Embeddings have non-zero variance across dimensions

## Success Criteria

After completing Phase 1, you should have:

1. **Working autoencoder model** that:
   - Takes multi-channel experimental data as input
   - Produces 16D embeddings capturing experimental patterns
   - Reconstructs oxygen timeseries with reasonable accuracy
   - Automatically detects problematic wells

2. **Quality control system** that:
   - Flags outlier wells based on reconstruction error
   - Validates embeddings cluster by experimental factors
   - Demonstrates event-aware pattern recognition

3. **Foundation for diffusion model** with:
   - Standardized data pipeline
   - Event detection and formatting
   - 16D embeddings ready for use as conditioning vectors

This autoencoder will serve as the foundation for the more advanced diffusion model while providing immediate value for quality control and pattern analysis.