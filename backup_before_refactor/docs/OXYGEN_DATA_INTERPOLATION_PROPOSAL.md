# Oxygen Data Interpolation and Synthesis Proposal

## Overview

This document proposes methods for interpolating missing oxygen consumption data and synthesizing complete concentration/time/O₂ curves from sparse observations. The ultimate goal is to generate smooth, continuous oxygen curves spanning 0-100 μM concentrations over two weeks, enabling both data gap filling and future drug structure → simulated curve prediction.

**Critical**: This proposal incorporates the complex experimental structure including pre-dosing baselines, media change artifacts, control wells, and hierarchical organization (wells → concentrations → drugs).

## Current Data Characteristics and Constraints

Based on the existing dataset:
- **Temporal coverage**: 17.8 ± 9.3 days (up to 46.9 days)
- **Sampling**: ~1.6 hour intervals (irregular), 287 ± 126 timepoints per well
- **Concentration range**: 0.0003 - 3333 μM (>10⁶ fold range, requires log-scale)
- **Common concentrations**: 0.03, 0.09, 0.28, 2.5, 7.5 μM (5 main concentrations)
- **Hierarchical structure**: 4 replicates × ~8 concentrations × 248 drugs
- **Missing data**: Currently 0%, but sparse concentration coverage and irregular sampling
- **Artifacts**: Media change spikes (100% of experiments), measurement noise, outliers (6.95%)

### Critical Experimental Structure
1. **Pre-dosing baseline**: 24-48h control period (ALL wells undosed)
2. **Media changes**: 100% of experiments, causing systematic spikes
3. **Control wells**: ~73 wells per plate (22%), continuous monitoring
4. **Dosing timing**: Not explicitly marked, inferred from variance/divergence
5. **Plate effects**: Spatial distribution, experimental batch effects

## Proposed Approaches

### 1. Event-Aware Autoencoder (Foundation Component) ⭐ **IMPLEMENT FIRST**

#### Rationale
An autoencoder provides immediate value for quality control, pattern detection, and data compression while serving as the foundation for more complex interpolation models. The 16-dimensional "barcode" embeddings capture experimental patterns and serve as powerful conditioning vectors.

#### Key Benefits
- **Auto-compressor**: Shrinks each curve to 16-number "barcode" for quick similarity analysis
- **Noise detector**: High reconstruction error automatically flags problematic wells
- **Joint view**: Same encoder handles full experiments and post-event windows
- **Plug-and-play**: Embeddings integrate with PCA/UMAP or stack with catch22/SAX features

#### Architecture Design

```python
class ExperimentalEventAwareAutoencoder:
    """
    Multi-channel autoencoder for experimental oxygen consumption data
    
    Input shape: [T, 7] channels:
    - oxygen_values: O₂ consumption measurements
    - mask: 0=missing data, 1=valid data  
    - media_change_flag: 1 at media change timepoints
    - dosing_flag: 1 at dosing timepoint
    - phase_indicators: [pre_dosing, early, sustained, late] one-hot [4D]
    
    Output: 16-dimensional embedding capturing experimental patterns
    """
    
    def __init__(self, latent_dim=16):
        # Encoder: 1D convolutions for temporal patterns
        self.encoder = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv1d(7, 64, kernel_size=7, padding=3),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            
            # Block 2: Downsample + increase channels
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            
            # Block 3: Further compression
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
            
            # Global pooling + dense layers
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder: Mirror architecture with upsampling
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Unflatten(1, (256, 1)),
            
            # Transpose convolutions for upsampling
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(), nn.BatchNorm1d(64),
            
            # Final reconstruction (only oxygen channel)
            nn.Conv1d(64, 1, kernel_size=7, padding=3)
        )
    
    def encode(self, x):
        return self.encoder(x.transpose(1, 2))  # [batch, channels, time]
    
    def decode(self, z, original_length):
        decoded = self.decoder(z)
        # Resize to match original length and apply mask
        return F.interpolate(decoded, size=original_length, mode='linear')
    
    def forward(self, x, mask):
        z = self.encode(x)
        recon = self.decode(z, x.shape[1])
        
        # Apply mask to ignore missing data in loss
        masked_recon = recon * mask.unsqueeze(1)
        return masked_recon, z

    def quality_score(self, x, mask):
        """Calculate reconstruction error for quality assessment"""
        recon, _ = self.forward(x, mask)
        mse = F.mse_loss(recon * mask.unsqueeze(1), 
                        x[:, :, 0:1] * mask.unsqueeze(1), 
                        reduction='none')
        return mse.mean(dim=(1, 2))  # Per-sample error
```

#### Enhanced Variants

```python
class MultiResolutionExperimentalAE:
    """
    Captures both local (event-specific) and global (full experiment) patterns
    """
    def __init__(self):
        # Short-term: 6h windows around events
        self.short_ae = ExperimentalEventAwareAutoencoder(latent_dim=8)
        
        # Long-term: Full experiment timeline  
        self.long_ae = ExperimentalEventAwareAutoencoder(latent_dim=16)
        
        # Fusion layer
        self.fusion = nn.Linear(8 + 16, 16)
    
    def encode(self, full_timeseries, event_windows):
        # Encode full experiment
        long_embedding = self.long_ae.encode(full_timeseries)
        
        # Encode each event window and average
        short_embeddings = []
        for window in event_windows:
            short_emb = self.short_ae.encode(window)
            short_embeddings.append(short_emb)
        
        avg_short = torch.stack(short_embeddings).mean(dim=0)
        
        # Combine multi-resolution embeddings
        combined = torch.cat([avg_short, long_embedding], dim=-1)
        return self.fusion(combined)

class ExperimentalBetaVAE(ExperimentalEventAwareAutoencoder):
    """
    Disentangled representation learning for experimental factors
    
    Expected disentangled factors:
    - Dims 0-3: Baseline consumption patterns
    - Dims 4-7: Media change response characteristics  
    - Dims 8-11: Dose-response sensitivity
    - Dims 12-15: Temporal dynamics (adaptation, recovery)
    """
    def __init__(self, latent_dim=16, beta=4.0):
        super().__init__(latent_dim)
        self.beta = beta
        
        # VAE-specific layers
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
    
    def encode(self, x):
        h = self.encoder[:-1](x.transpose(1, 2))  # Stop before final linear
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss(self, recon, original, mu, logvar, mask):
        # Reconstruction loss (masked)
        recon_loss = F.mse_loss(
            recon * mask.unsqueeze(1), 
            original[:, :, 0:1] * mask.unsqueeze(1)
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss
```

### 2. Event-Aware Diffusion Model (Advanced Component)

#### Rationale
Diffusion models excel at learning complex data distributions, but must be adapted for this experimental structure:
- **Event-driven dynamics**: Media changes create systematic artifacts that are informative, not noise
- **Multi-phase modeling**: Pre-dosing baseline vs. post-dosing response require different dynamics
- **Control-aware normalization**: Model must understand plate-level control behavior
- **Hierarchical structure**: 4 replicates provide uncertainty estimates for interpolation

#### Architecture Design

```python
class EventAwareOxygenDiffusionModel:
    """
    Conditional diffusion model for oxygen consumption curves with experimental events
    
    Inputs:
    - concentration: log₁₀(μM) [-3.5 to 3.5] (ESSENTIAL for 10⁶ fold range)
    - time_relative: hours since dosing [0 to 300h] 
    - time_absolute: hours from experiment start [0 to 400h]
    - media_change_times: list of media change timestamps
    - drug_embedding: molecular descriptors/fingerprints [256D]
    - plate_context: experimental conditions + control well behavior [32D]
    - phase_indicator: [pre_dosing, early_response, sustained, late] one-hot [4D]
    
    Output:
    - oxygen_consumption: predicted O₂ level [-27 to 120 range, from actual data]
    """
    
    def __init__(self):
        # Event-aware temporal encoder (handles media changes explicitly)
        self.event_encoder = EventTransformerEncoder(
            d_model=256, n_heads=8, n_layers=6,
            event_types=['dosing', 'media_change', 'measurement_artifact']
        )
        
        # Log-concentration embedding (critical for dose-response)
        self.conc_encoder = MLP([1, 64, 128, 256])
        
        # Drug molecular embedding 
        self.drug_encoder = MLP([256, 512, 256])
        
        # Plate/control context (captures control well dynamics)
        self.plate_encoder = MLP([32, 128, 256])
        
        # Phase-specific dynamics (pre-dosing vs post-dosing behavior)
        self.phase_encoder = MLP([4, 64, 128])
        
        # Diffusion denoising network with event attention
        self.denoiser = EventAwareUNet1D(
            in_channels=1,  # O₂ values
            condition_dim=256 + 256 + 256 + 256 + 128,  # all embeddings
            time_embed_dim=128,
            event_attention=True  # Attend to media change timing
        )
```

#### Training Strategy

1. **Data preparation with experimental structure**:
   - **Phase segmentation**: Split each well into pre-dosing (24-48h) and post-dosing phases
   - **Event detection**: Automatically detect media changes via variance analysis
   - **Control normalization**: Normalize each well to its plate's control wells behavior
   - **Log-concentration scaling**: Essential for 0.0003-3333 μM range (factor of 10⁶)
   - **Replicate handling**: Use 4 replicates for uncertainty-aware training

2. **Hierarchical training approach**:
   - **Level 1**: Train on individual wells (preserve replicate variability)
   - **Level 2**: Train on concentration-level aggregates (reduce noise)
   - **Level 3**: Learn drug-level dose-response relationships (Hill curves)

3. **Event-aware conditioning**:
   - **Media change timing**: Explicit temporal markers for spikes
   - **Plate context**: Include control well dynamics as conditioning
   - **Phase indicators**: Different model behavior for pre/post-dosing
   - **Dosing detection**: Learn to identify dosing time from trajectory changes

4. **Quality-aware training**:
   - **Exclusion handling**: Train only on is_excluded=false wells
   - **Outlier robustness**: Use median-based loss functions (MAE vs MSE)
   - **Missing data simulation**: Randomly mask timepoints to simulate gaps

#### Inference Process

```python
def interpolate_event_aware_oxygen_curve(model, drug_embedding, concentration_range, 
                                       experiment_timeline, plate_controls):
    """
    Generate complete O₂ curve respecting experimental structure
    
    Args:
        drug_embedding: [256] molecular descriptors
        concentration_range: [N] concentrations in log₁₀ μM
        experiment_timeline: dict with dosing_time, media_changes, duration
        plate_controls: [T] control well oxygen trajectory for normalization
    
    Returns:
        oxygen_curves: [N, T] predicted O₂ consumption matrix
        uncertainty: [N, T] prediction uncertainty (from 4-replicate structure)
    """
    
    # Extract experimental structure
    dosing_time = experiment_timeline['dosing_time']  # e.g., 48h
    media_changes = experiment_timeline['media_changes']  # e.g., [72h, 120h, 168h]
    total_duration = experiment_timeline['duration']  # e.g., 336h
    
    # Create phase indicators
    time_points = torch.arange(0, total_duration, 1.6)  # 1.6h sampling
    phase_indicators = create_phase_indicators(time_points, dosing_time, media_changes)
    
    # Create conditioning tensors
    conc_embed = model.conc_encoder(concentration_range)
    drug_embed = model.drug_encoder(drug_embedding)
    plate_embed = model.plate_encoder(plate_controls)
    phase_embed = model.phase_encoder(phase_indicators)
    
    # Event-aware temporal embedding
    event_context = model.event_encoder(time_points, media_changes, dosing_time)
    
    # Sample from diffusion model with replicate-based uncertainty
    curves = []
    for replicate in range(4):  # Generate 4 replicates
        noise = torch.randn(len(concentration_range), len(time_points))
        
        # Iterative denoising with event awareness
        for t in reversed(range(model.num_timesteps)):
            # Combine all conditioning
            condition = torch.cat([
                conc_embed, 
                drug_embed.expand(len(concentration_range), -1),
                plate_embed.expand(len(concentration_range), -1),
                event_context,
                phase_embed
            ], dim=-1)
            
            # Event-aware denoise step
            noise = model.event_aware_denoise_step(
                noise, t, condition, media_changes, dosing_time
            )
        
        curves.append(noise)
    
    # Aggregate replicates
    curves = torch.stack(curves)  # [4, N, T]
    mean_curve = curves.mean(dim=0)  # [N, T]
    uncertainty = curves.std(dim=0)  # [N, T] - replicate-based uncertainty
    
    return mean_curve, uncertainty

def create_phase_indicators(time_points, dosing_time, media_changes):
    """Create phase indicators for pre-dosing, early, sustained, late response"""
    phases = torch.zeros(len(time_points), 4)  # one-hot encoding
    
    for i, t in enumerate(time_points):
        if t < dosing_time:
            phases[i, 0] = 1  # pre_dosing
        elif t < dosing_time + 72:  # 3 days post-dosing
            phases[i, 1] = 1  # early_response  
        elif t < dosing_time + 240:  # 10 days post-dosing
            phases[i, 2] = 1  # sustained
        else:
            phases[i, 3] = 1  # late
    
    return phases
```

### 2. Event-Driven Neural ODE (Alternative)

For cases requiring interpretable dynamics with experimental events:

```python
class EventDrivenOxygenODE:
    """
    Learn event-aware oxygen dynamics with interpretable parameters
    """
    
    def __init__(self):
        # Base physiological parameters (pre-dosing)
        self.baseline_network = MLP([1, 32, 64, 2])  # → [baseline_consumption, adaptation_rate]
        
        # Concentration-dependent toxicity parameters  
        self.toxicity_network = MLP([1, 64, 128, 3])  # → [EC50, Emax, hill_slope]
        
        # Drug-specific parameter modulation
        self.drug_modulator = MLP([256, 128, 8])  # molecular → all rate modifiers
        
        # Media change response parameters
        self.media_response_network = MLP([2, 32, 64, 2])  # [time_since_media, conc] → [spike_mag, recovery_rate]
    
    def ode_func(self, t, y, concentration, drug_embedding, media_change_times, dosing_time):
        """
        dy/dt = f(y, concentration, drug_properties, t, events)
        
        Incorporates:
        - Pre-dosing baseline dynamics
        - Concentration-dependent toxicity (Hill curves)
        - Media change artifacts with recovery
        - Drug-specific parameter modulation
        """
        log_conc = torch.log10(torch.clamp(concentration, 1e-6, 1e6))
        
        # Determine current phase
        if t < dosing_time:
            # Pre-dosing: only baseline dynamics
            baseline_params = self.baseline_network(torch.tensor([0.0]))  # No concentration
            baseline_consumption, adaptation_rate = baseline_params
            
            dydt = -baseline_consumption + adaptation_rate * (5.0 - y)  # Adapt toward 5 μM baseline
            
        else:
            # Post-dosing: full dynamics
            
            # 1. Baseline consumption
            baseline_params = self.baseline_network(log_conc)
            baseline_consumption, _ = baseline_params
            
            # 2. Concentration-dependent toxicity (Hill equation in rate space)
            tox_params = self.toxicity_network(log_conc)
            EC50, Emax, hill_slope = tox_params
            
            # Hill equation for toxicity rate
            tox_rate = Emax * (concentration ** hill_slope) / (EC50 ** hill_slope + concentration ** hill_slope)
            
            # 3. Drug-specific modulation
            drug_effects = self.drug_modulator(drug_embedding)
            baseline_mod, tox_mod, recovery_mod = drug_effects[:3], drug_effects[3:6], drug_effects[6:]
            
            # Apply drug modulation
            baseline_consumption = baseline_consumption * (1 + baseline_mod.mean())
            tox_rate = tox_rate * (1 + tox_mod.mean())
            
            # 4. Media change artifacts
            media_effect = 0.0
            for media_time in media_change_times:
                if abs(t - media_time) < 12:  # Within 12h of media change
                    time_since_media = t - media_time
                    media_params = self.media_response_network(
                        torch.tensor([time_since_media, log_conc])
                    )
                    spike_magnitude, recovery_rate = media_params
                    
                    # Exponential decay from spike
                    media_effect += spike_magnitude * torch.exp(-recovery_rate * time_since_media)
            
            # Combined dynamics
            dydt = (
                -baseline_consumption * (1 + 0.1 * y)  # Baseline consumption
                - tox_rate * y                          # Toxicity-induced consumption  
                + media_effect                          # Media change artifacts
                + 0.01 * (5.0 - y)                     # Slow recovery toward baseline
            )
        
        return dydt
    
    def get_hill_parameters(self, drug_embedding, feature='oxygen_consumption'):
        """
        Extract Hill curve parameters for dose-response analysis
        Compatible with existing Hill curve fitting approach
        """
        concentrations = torch.logspace(-3.5, 3.5, 100)  # 0.0003 to 3333 μM
        
        responses = []
        for conc in concentrations:
            # Simulate to steady state (no media changes, post-dosing)
            steady_state = self.solve_to_steady_state(
                concentration=conc, 
                drug_embedding=drug_embedding,
                phase='post_dosing'
            )
            responses.append(steady_state)
        
        responses = torch.stack(responses)
        
        # Fit Hill equation to extract EC50, Emax, etc.
        hill_params = fit_hill_curve(concentrations, responses)
        
        return hill_params
```

### 3. Hybrid Physics-Informed Approach

Combine mechanistic understanding with neural networks:

```python
class PhysicsInformedOxygenModel:
    """
    Incorporate known biological constraints
    """
    
    def __init__(self):
        self.base_consumption = 5.0  # Baseline O₂ consumption (μM/h)
        self.max_consumption = 50.0  # Maximum possible consumption
        
        # Neural network predicts deviations from physics
        self.deviation_net = ConditionalTransformer()
    
    def forward(self, concentration, time, drug_embedding):
        # Physics-based baseline
        baseline_curve = self.michaelis_menten_dynamics(concentration, time)
        
        # Neural correction
        deviation = self.deviation_net(concentration, time, drug_embedding)
        
        # Constrained output
        return torch.clamp(baseline_curve + deviation, -50, 150)
    
    def michaelis_menten_dynamics(self, concentration, time):
        """Simple Michaelis-Menten-like consumption model"""
        V_max = self.max_consumption
        K_m = 1.0  # Half-saturation concentration
        
        rate = V_max * concentration / (K_m + concentration)
        
        # Integrate over time with media change artifacts
        consumption = self.integrate_with_media_changes(rate, time)
        
        return consumption
```

## Detailed Implementation Plan

### Phase 1: Foundation - Event-Aware Autoencoder

#### Step 1.1: Data Pipeline Setup
**Goal**: Load and preprocess oxygen consumption data with experimental structure
**Deliverable**: `data_pipeline.py` with functions to load, filter, and format data

**Tasks**:
1. **Database connection setup**
   - Use existing DATABASE_URL from CLAUDE.md
   - Create `load_oxygen_data()` function with quality filters (is_excluded=false, ≥14 days, ≥4 concentrations)
   - Return DataFrame with columns: [well_id, drug, concentration, elapsed_time, oxygen_value, plate_id]

2. **Control well identification**
   - Create `identify_control_wells()` function
   - Flag wells with concentration=0 or drug='DMSO' as controls
   - Verify ~73 control wells per plate expectation

3. **Event detection implementation**
   - Create `detect_media_changes()` function using variance analysis
   - Look for sudden spikes in oxygen consumption across all wells on a plate
   - Return list of media change timestamps per plate

4. **Dosing time detection**
   - Create `detect_dosing_time()` function 
   - Find timepoint where treatment wells diverge from controls (typically 24-48h)
   - Use variance increase + trajectory divergence metrics

**Helper code**:
```python
def load_oxygen_data(filters=None):
    """Load and filter oxygen consumption data"""
    # Load from database using existing connection
    # Apply quality filters: is_excluded=false, ≥14 days, ≥4 concentrations
    # Return clean DataFrame
    pass

def detect_events(plate_data):
    """Detect media changes and dosing events for a plate"""
    # Variance analysis for media changes
    # Trajectory divergence for dosing time
    # Return event timestamps
    pass
```

#### Step 1.2: Data Formatting for Autoencoder
**Goal**: Convert raw timeseries into multi-channel tensor format
**Deliverable**: `data_formatter.py` with tensor conversion functions

**Tasks**:
1. **Resampling to regular grid**
   - Create `resample_to_hourly()` function
   - Convert irregular ~1.6h sampling to 1h grid
   - Use linear interpolation for missing timepoints
   - Pad/truncate to standard length (e.g., 336h = 2 weeks)

2. **Multi-channel tensor creation**
   - Create `create_multichannel_input()` function
   - Channel 0: oxygen_values (normalized to [-1, 1] range)
   - Channel 1: mask (1=valid data, 0=missing/interpolated)
   - Channel 2: media_change_flag (1 at media change times, 0 elsewhere)
   - Channel 3: dosing_flag (1 at dosing time, 0 elsewhere)  
   - Channels 4-7: phase_indicators (one-hot: pre_dosing, early, sustained, late)

3. **Control normalization**
   - Create `normalize_to_controls()` function
   - For each plate, subtract median control well trajectory
   - Preserve original scale for reconstruction

**Helper code**:
```python
def create_multichannel_input(well_data, events, controls):
    """Convert single well timeseries to [T, 7] tensor"""
    # Resample to 1h grid
    # Create mask channel
    # Add event flags  
    # Add phase indicators
    # Return tensor [time_points, 7_channels]
    pass
```

#### Step 1.3: Basic Autoencoder Implementation
**Goal**: Implement and test basic autoencoder architecture
**Deliverable**: `autoencoder.py` with working model class

**Tasks**:
1. **Model architecture implementation**
   - Copy `ExperimentalEventAwareAutoencoder` class from proposal
   - Add proper imports (torch, torch.nn, torch.nn.functional)
   - Test with dummy data to ensure tensor shapes work

2. **Training loop setup**
   - Create `train_autoencoder()` function
   - Use MSE loss with mask weighting: `loss = F.mse_loss(recon * mask, target * mask)`
   - Add L2 regularization: `loss += 1e-4 * torch.norm(latent_embedding)`
   - Adam optimizer with lr=1e-3

3. **Quality scoring function**
   - Implement `quality_score()` method from proposal
   - Calculate per-well reconstruction error
   - Flag wells with error > 3 * MAD as outliers

**Helper code**:
```python
def train_autoencoder(model, train_loader, val_loader, epochs=50):
    """Training loop for autoencoder"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # Training phase
        for batch in train_loader:
            # Forward pass, loss calculation, backprop
            pass
        
        # Validation phase 
        # Early stopping if val loss stops improving
        pass
```

#### Step 1.4: Data Loading and Training Pipeline
**Goal**: Create end-to-end pipeline from database to trained model
**Deliverable**: `train_pipeline.py` that loads data and trains autoencoder

**Tasks**:
1. **Dataset class creation**
   - Create `OxygenDataset(torch.utils.data.Dataset)` 
   - Load all wells, apply filtering, format as tensors
   - Return (multichannel_input, mask) pairs

2. **Train/validation split**
   - Split by drug (not by wells) to prevent leakage
   - 80% drugs for training, 20% for validation
   - Ensure no drug appears in both sets

3. **Training execution**
   - Load data using steps 1.1-1.2
   - Create train/val DataLoaders with batch_size=256
   - Train autoencoder for 50 epochs with early stopping
   - Save best model checkpoint

**Helper code**:
```python
class OxygenDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Load and process all wells
        self.wells = load_oxygen_data()
        self.events = detect_events(self.wells)
        # Format as tensors
        
    def __getitem__(self, idx):
        # Return (input_tensor, mask) for one well
        pass
```

#### Step 1.5: Quality Control and Validation
**Goal**: Validate autoencoder learns meaningful patterns and detects outliers
**Deliverable**: `quality_analysis.py` with validation metrics and visualizations

**Tasks**:
1. **Reconstruction quality metrics**
   - Calculate reconstruction MSE for all wells
   - Identify outlier wells (high reconstruction error)
   - Compare outliers with existing is_excluded flags

2. **Embedding analysis**
   - Extract 16D embeddings for all wells
   - Plot t-SNE/UMAP of embeddings colored by drug, concentration
   - Verify that similar conditions cluster together

3. **Event response validation**
   - Check that wells with media changes have different embeddings than stable wells
   - Verify that pre-dosing vs post-dosing periods have distinct representations

**Helper code**:
```python
def analyze_embeddings(model, dataset):
    """Analyze learned embeddings"""
    # Extract embeddings for all wells
    # Create UMAP visualization
    # Calculate clustering metrics
    pass

def validate_outlier_detection(model, dataset):
    """Validate quality control"""
    # Calculate reconstruction errors
    # Compare with manual quality flags
    # Generate outlier report
    pass
```

### Phase 2: Enhanced Autoencoder Variants

#### Step 2.1: Multi-Resolution Architecture
**Goal**: Implement short-term + long-term pattern capture
**Deliverable**: `multi_resolution_ae.py` with combined local/global modeling

**Tasks**:
1. **Event window extraction**
   - Create `extract_event_windows()` function
   - Extract 6h windows around each media change
   - Handle variable numbers of media changes per experiment

2. **Multi-resolution model implementation** 
   - Implement `MultiResolutionExperimentalAE` from proposal
   - Train separate autoencoders for 6h windows vs full experiments
   - Combine 8D + 16D embeddings via fusion layer

3. **Comparative evaluation**
   - Compare single-resolution vs multi-resolution embeddings
   - Test clustering quality and reconstruction accuracy
   - Validate that short-term patterns capture event responses

#### Step 2.2: β-VAE for Disentanglement
**Goal**: Learn disentangled factors (baseline, events, dose-response, dynamics)
**Deliverable**: `beta_vae.py` with interpretable latent dimensions

**Tasks**:
1. **β-VAE implementation**
   - Implement `ExperimentalBetaVAE` from proposal
   - Add KL divergence loss with β=4.0 weighting
   - Train with same data pipeline as basic autoencoder

2. **Disentanglement analysis**
   - Analyze each latent dimension separately
   - Plot latent values vs experimental factors (concentration, time, events)
   - Verify expected disentanglement: dims 0-3 (baseline), 4-7 (events), etc.

3. **Interpretability validation**
   - Generate reconstructions by varying single latent dimensions
   - Confirm that dimension changes correspond to expected experimental factors

### Phase 3: Integration Preparation for Diffusion Model

#### Step 3.1: Autoencoder Embedding Integration
**Goal**: Prepare autoencoder embeddings as conditioning vectors
**Deliverable**: `embedding_pipeline.py` for diffusion model conditioning

**Tasks**:
1. **Hierarchical embedding creation**
   - Well-level: 16D embeddings from trained autoencoder
   - Concentration-level: Average across 4 replicates  
   - Drug-level: Aggregate across concentrations

2. **Conditioning vector preparation**
   - Combine autoencoder embeddings (16D) with molecular descriptors (256D)
   - Add concentration (1D) and experimental context (32D)
   - Create standardized conditioning pipeline

3. **Validation for diffusion integration**
   - Verify embeddings capture dose-response relationships
   - Test that similar drugs have similar embeddings
   - Confirm embeddings respect experimental structure (controls, events)

### Phase 2: Event-Aware Diffusion Model Development (5-7 weeks)

1. **Event-aware model architecture**:
   - **EventAwareUNet1D**: Implement attention mechanisms for media change timing
   - **Multi-phase conditioning**: Separate embeddings for pre-dosing vs post-dosing dynamics
   - **Control-aware normalization**: Integrate plate control trajectories into conditioning
   - **Replicate uncertainty modeling**: Generate 4 replicates to match experimental structure

2. **Experimental-structure-aware training**:
   - **Phase-stratified sampling**: Balance pre-dosing, early, sustained, late response samples
   - **Event-synchronized batching**: Group samples by media change timing for consistent learning
   - **Control-referenced loss**: Weight losses by deviation from control well behavior
   - **Hill-curve consistency loss**: Penalize violations of dose-response monotonicity

3. **Evaluation metrics with biological constraints**:
   - **Event response fidelity**: Accurate media change spike prediction
   - **Dose-response preservation**: EC₅₀, Eₘₐₓ, hill slope accuracy across concentration ranges
   - **Control consistency**: Model predictions should match control well behavior at zero concentration
   - **Replicate concordance**: Generated replicates should have realistic inter-replicate variance
   - **Phase transition accuracy**: Smooth transitions between pre/post-dosing dynamics

### Phase 3: Experimental-Structure-Aware Applications and Validation (4-5 weeks)

1. **Event-aware gap filling validation**:
   - **Missing timepoints**: Remove data around media changes, during critical response phases
   - **Missing replicates**: Simulate well failures, test replicate imputation quality
   - **Missing concentration levels**: Hold out 1-2 concentrations per drug, test dose-response interpolation
   - **Validation across experimental phases**: Separate evaluation for pre-dosing, early response, sustained response

2. **Dose-response curve completion**:
   - **Sparse concentration training**: Train on 3-4 concentrations (e.g., 0.03, 0.28, 2.5, 7.5 μM)
   - **Full curve prediction**: Generate smooth curves across 0.001-100 μM range (100+ points)
   - **Hill parameter preservation**: Validate that EC₅₀, Eₘₐₓ, hill slope match experimental values
   - **Control-referenced validation**: Ensure zero-concentration predictions match control wells

3. **Cross-drug generalization with experimental constraints**:
   - **Drug family validation**: Test on held-out drugs within same chemical classes
   - **Novel drug prediction**: Predict curves for drugs with only molecular descriptors
   - **Experimental protocol transfer**: Train on one experimental setup, test on different plate layouts/timing
   - **Quality flag integration**: Validate that model respects is_excluded flags and quality metrics

## Expected Outputs

### Immediate Applications with Experimental Structure

1. **Complete concentration-time matrices**: For each drug, generate smooth O₂ curves at 20+ concentrations from 0.001-100 μM, respecting:
   - Pre-dosing baseline periods (24-48h)
   - Media change artifacts and recovery patterns
   - Dose-response relationships (Hill curves)
   - Replicate variability (4-replicate uncertainty estimates)

2. **Event-aware gap filling**: Replace missing/corrupted time points while preserving:
   - Media change spike characteristics
   - Control well normalization
   - Phase-appropriate dynamics (pre vs post-dosing)

3. **Quality assessment with experimental context**: 
   - Identify anomalous measurements relative to plate controls
   - Detect wells that violate expected dose-response patterns
   - Flag inconsistent replicate behavior

### Future Applications with Experimental Constraints

1. **Virtual screening with experimental realism**: 
   - Predict complete oxygen curves including media change responses
   - Generate realistic experimental timelines with dosing/media change timing
   - Provide uncertainty estimates based on 4-replicate structure

2. **Dose optimization with Hill curve constraints**:
   - Identify optimal concentration ranges while respecting dose-response monotonicity
   - Account for media change artifacts in toxicity assessment
   - Consider replicate variability in safety margins

3. **Mechanism understanding through event analysis**:
   - Analyze drug-specific media change responses (stress response)
   - Identify molecular features that predict baseline vs toxicity-induced changes
   - Understand concentration-dependent recovery patterns

## Technical Considerations with Experimental Constraints

### Computational Requirements

- **Training**: 6-10 GPUs, ~4-5 days for event-aware diffusion model (more complex than standard)
- **Inference**: Single GPU, ~2-3 seconds per drug/concentration (includes event processing)
- **Storage**: ~20GB for model weights, ~100GB for training data (includes event annotations)

### Data Requirements with Experimental Structure

- **Minimum**: 50+ drugs with complete concentration series AND identified experimental events
- **Optimal**: 200+ drugs across diverse chemical space with annotated media changes, dosing times
- **Validation**: 20-30 drugs held out completely, ensuring no drug family leakage
- **Control wells**: Must include ~73 control wells per plate for proper normalization

### Quality Assurance with Biological Constraints

1. **Physical constraints**: Ensure predictions respect O₂ bounds [-27, 120] (actual data range)
2. **Dose-response monotonicity**: Enforce Hill curve constraints in log-concentration space
3. **Event consistency**: Media change responses should match control well patterns
4. **Control consistency**: Zero-concentration predictions must match control wells
5. **Replicate realism**: Generated replicates should have CV similar to experimental data (typically 10-30%)
6. **Phase transitions**: Smooth transitions between pre-dosing and post-dosing dynamics
7. **Temporal constraints**: Respect 1.6h ± 6.2h irregular sampling patterns

## Implementation Priority

**Highest priority**: Event-aware diffusion model approach due to:
- Superior handling of sparse, irregular data with experimental events
- Natural integration of multiple conditioning variables (concentration, drug, plate, events)
- Ability to model complex experimental structure (pre-dosing, media changes, controls)
- State-of-the-art performance on similar time series tasks
- Potential for future drug discovery applications with experimental realism

**Secondary approaches**: Event-driven Neural ODE and physics-informed models for interpretability and mechanism understanding of experimental dynamics.

## Critical Success Factors

This proposal **must** incorporate the experimental nuances identified in O2_REALTIME_DATA.md:

1. **Event-driven modeling**: Media changes are not noise but informative experimental events
2. **Control-aware normalization**: All predictions must be referenced to plate control behavior  
3. **Phase-specific dynamics**: Pre-dosing vs post-dosing require different model behaviors
4. **Log-concentration scaling**: Essential for the 10⁶-fold concentration range (0.0003-3333 μM)
5. **Replicate structure**: Must generate 4 replicates with realistic inter-replicate variance
6. **Hill curve consistency**: Dose-response relationships must be monotonic and pharmacologically interpretable
7. **Quality integration**: Respect is_excluded flags and experimental quality metrics

This proposal provides a roadmap for transforming sparse oxygen consumption data into rich, continuous curves that preserve experimental structure and biological constraints, suitable for both analytical gap-filling and predictive drug discovery applications.