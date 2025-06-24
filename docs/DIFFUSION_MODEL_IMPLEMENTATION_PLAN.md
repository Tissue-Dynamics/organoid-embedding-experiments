# Diffusion Model Implementation Plan for Oxygen Data Interpolation
## Flexible Replicate Generation with Drug-Aware Media Change Dynamics

This document presents a comprehensive plan for implementing a diffusion model that generates realistic oxygen consumption curves with flexible replicate handling and drug-specific media change responses.

## Executive Summary

We will implement a **flexible replicate diffusion model** that can generate 1-4 oxygen consumption curves based on available data, while learning drug-specific media change dynamics. The model adapts to the actual number of replicates available and learns how different drugs affect both the magnitude and timing of media change responses.

## Core Innovation: Adaptive Architecture

The model dynamically adjusts to handle any number of replicates (1-4) and learns drug-specific media change recovery patterns rather than using fixed timescales.

## Architecture Overview

### 1. Flexible Replicate Diffusion Model

```python
class FlexibleReplicateDiffusionModel(nn.Module):
    """
    Generates 1-4 replicate curves based on available data.
    """
    def __init__(self, max_replicates=4):
        super().__init__()
        self.max_replicates = max_replicates
        
        # Core denoising network - handles variable replicate counts
        self.denoiser = AdaptiveReplicateUNet(
            max_channels=max_replicates,
            time_dim=256,
            hidden_dims=[256, 512, 1024, 2048]
        )
        
        # Drug-specific media change dynamics
        self.media_dynamics = DrugAwareMediaChangeModule(
            drug_embed_dim=256,
            dynamics_dim=128
        )
        
        # Replicate count encoder
        self.replicate_count_encoder = nn.Embedding(max_replicates + 1, 64)
```

### 2. Adaptive Replicate Handling

```python
class AdaptiveReplicateUNet(nn.Module):
    """
    UNet that adapts to variable numbers of replicates.
    """
    def __init__(self, max_channels=4):
        super().__init__()
        # Flexible input convolution
        self.input_conv = nn.Conv1d(max_channels, 256, kernel_size=7, padding=3)
        
        # Mask-aware processing
        self.replicate_attention = ReplicateMaskAttention()
        
    def forward(self, x_t, t, media_context, replicate_mask, replicate_latents):
        # x_t: [batch, time, max_replicates] - padded with zeros for missing replicates
        # replicate_mask: [batch, max_replicates] - 1 for valid replicates, 0 for missing
        
        # Apply replicate mask
        x_masked = x_t * replicate_mask.unsqueeze(1)
        
        # Encode number of replicates as additional context
        n_reps = replicate_mask.sum(dim=1)
        rep_count_emb = self.replicate_count_encoder(n_reps)
        
        # Process with attention to valid replicates only
        h = self.input_conv(x_masked.transpose(1, 2))
        h = self.replicate_attention(h, replicate_mask)
        
        # Continue with standard UNet processing
        # ...
```

### 3. Drug-Aware Media Change Dynamics

```python
class DrugAwareMediaChangeModule(nn.Module):
    """
    Learns drug-specific media change response patterns.
    """
    def __init__(self, drug_embed_dim=256, dynamics_dim=128):
        super().__init__()
        
        # Drug-specific response parameters
        self.drug_response_net = nn.Sequential(
            nn.Linear(drug_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, dynamics_dim)
        )
        
        # Learn response characteristics per drug
        self.response_decoder = nn.ModuleDict({
            'spike_magnitude': nn.Linear(dynamics_dim, 1),
            'recovery_rate': nn.Linear(dynamics_dim, 1),
            'adaptation_rate': nn.Linear(dynamics_dim, 1),
            'secondary_effects': nn.Linear(dynamics_dim, 1)
        })
    
    def get_media_response_profile(self, drug_embedding, concentration):
        """
        Generate drug-specific media change response parameters.
        """
        # Base drug response
        drug_dynamics = self.drug_response_net(drug_embedding)
        
        # Concentration modulation
        conc_scale = torch.log(concentration + 1e-6) / 10.0  # Log scale normalization
        drug_dynamics = drug_dynamics * (1 + conc_scale)
        
        # Decode specific response characteristics
        response_profile = {
            'spike_magnitude': self.response_decoder['spike_magnitude'](drug_dynamics),
            'recovery_tau': torch.exp(self.response_decoder['recovery_rate'](drug_dynamics)),  # Ensure positive
            'adaptation_tau': torch.exp(self.response_decoder['adaptation_rate'](drug_dynamics)),
            'secondary_magnitude': self.response_decoder['secondary_effects'](drug_dynamics)
        }
        
        return response_profile
    
    def compute_media_effect(self, time_since_change, drug_embedding, concentration):
        """
        Compute the effect of a media change at a given time offset.
        """
        profile = self.get_media_response_profile(drug_embedding, concentration)
        
        # Multi-phase response model
        # Phase 1: Immediate spike
        spike = profile['spike_magnitude'] * torch.exp(-time_since_change / 2.0)
        
        # Phase 2: Recovery (drug-dependent rate)
        recovery = -profile['spike_magnitude'] * 0.5 * (
            torch.exp(-time_since_change / profile['recovery_tau']) - 
            torch.exp(-time_since_change / 2.0)
        )
        
        # Phase 3: Long-term adaptation
        adaptation = profile['secondary_magnitude'] * (
            1 - torch.exp(-time_since_change / profile['adaptation_tau'])
        )
        
        total_effect = spike + recovery + adaptation
        return total_effect, profile
```

### 4. Flexible Replicate Generation

```python
class ReplicateGenerator(nn.Module):
    """
    Generates appropriate number of replicates based on training data.
    """
    def __init__(self):
        super().__init__()
        # Learn replicate correlation patterns
        self.correlation_predictor = nn.Sequential(
            nn.Linear(256 + 1, 128),  # drug_embedding + concentration
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output correlation coefficient
        )
    
    def generate_replicates(self, base_curve, n_replicates, drug_embedding, concentration):
        """
        Generate n_replicates from a base curve with appropriate variability.
        """
        if n_replicates == 1:
            return base_curve.unsqueeze(-1)
        
        # Predict correlation for this drug/concentration
        drug_conc = torch.cat([drug_embedding, torch.log(concentration + 1e-6).unsqueeze(-1)], dim=-1)
        correlation = self.correlation_predictor(drug_conc)
        
        # Generate correlated variations
        replicates = [base_curve]
        for i in range(1, n_replicates):
            # Replicate-specific noise
            noise = torch.randn_like(base_curve) * 0.1  # 10% variation
            
            # Correlate with base curve
            replicate_i = base_curve + torch.sqrt(1 - correlation) * noise
            
            # Add replicate-specific drift
            drift = self.generate_replicate_drift(i, drug_embedding)
            replicate_i = replicate_i + drift
            
            replicates.append(replicate_i)
        
        return torch.stack(replicates, dim=-1)
```

### 5. Training with Variable Replicates

```python
def train_with_flexible_replicates(model, data_loader):
    """
    Train on data with varying numbers of replicates.
    """
    for batch in data_loader:
        # batch.oxygen: [batch, time, max_replicates] - padded with NaN
        # batch.replicate_mask: [batch, max_replicates] - indicates valid replicates
        
        # Count actual replicates per sample
        n_reps = batch.replicate_mask.sum(dim=1)
        
        # Replace NaN with 0 for processing
        oxygen_data = torch.nan_to_num(batch.oxygen, nan=0.0)
        
        # Forward diffusion with replicate-aware noise
        noise = generate_replicate_aware_noise(
            oxygen_data.shape, 
            batch.replicate_mask,
            correlation=0.7
        )
        
        t = torch.randint(0, num_timesteps, (batch_size,))
        x_t = forward_diffusion(oxygen_data, noise, t)
        
        # Get drug-specific media dynamics
        media_context = model.media_dynamics(
            batch.media_schedule,
            batch.drug_embedding,
            batch.concentration
        )
        
        # Predict noise with flexible replicate handling
        noise_pred = model.denoiser(
            x_t, t, media_context, 
            batch.replicate_mask,
            replicate_latents=None  # Can be None for existing data
        )
        
        # Masked loss - only compute on valid replicates
        loss = masked_replicate_loss(
            noise_pred, noise, 
            batch.replicate_mask,
            media_proximity=batch.media_proximity
        )
```

### 6. Loss Functions for Variable Replicates

```python
class MaskedReplicateLoss(nn.Module):
    """
    Loss function that handles missing replicates properly.
    """
    def forward(self, pred, target, replicate_mask, media_proximity):
        # Expand mask to match data dimensions
        mask_expanded = replicate_mask.unsqueeze(1)  # [batch, 1, max_reps]
        
        # Individual replicate loss (only on valid replicates)
        individual_loss = F.mse_loss(pred, target, reduction='none')
        masked_loss = individual_loss * mask_expanded
        
        # Normalize by actual number of replicates
        n_valid = replicate_mask.sum(dim=1, keepdim=True).unsqueeze(1)
        normalized_loss = masked_loss.sum(dim=2) / (n_valid + 1e-8)
        
        # Media proximity weighting (drug-agnostic for now)
        media_weights = self.compute_media_weights(media_proximity)
        weighted_loss = normalized_loss * media_weights
        
        # Coherence loss (only when >1 replicate)
        coherence_loss = 0
        for i in range(len(replicate_mask)):
            n_reps = int(replicate_mask[i].sum())
            if n_reps > 1:
                # Compute variance only among valid replicates
                valid_preds = pred[i, :, :n_reps]
                valid_targets = target[i, :, :n_reps]
                
                pred_var = valid_preds.var(dim=1)
                target_var = valid_targets.var(dim=1)
                coherence_loss += F.mse_loss(pred_var, target_var)
        
        coherence_loss = coherence_loss / len(replicate_mask)
        
        return weighted_loss.mean() + 0.1 * coherence_loss
```

### 7. Generation for Different Replicate Counts

```python
def generate_curves(
    model,
    drug,
    concentration,
    media_schedule,
    n_replicates=4,  # Can be 1, 2, 3, or 4
    experiment_duration=400
):
    """
    Generate specified number of replicate curves.
    """
    # Create replicate mask
    replicate_mask = torch.zeros(1, 4)
    replicate_mask[0, :n_replicates] = 1
    
    # Initialize with appropriate noise
    if n_replicates == 1:
        # Single curve - no correlation needed
        x_T = torch.randn(1, experiment_duration, 4)
        x_T[:, :, 1:] = 0  # Zero out unused channels
    else:
        # Multiple curves - use correlated noise
        x_T = torch.zeros(1, experiment_duration, 4)
        
        # Generate base noise
        base_noise = torch.randn(1, experiment_duration, 1)
        
        # Create correlated noise for each replicate
        for i in range(n_replicates):
            correlation = 0.7  # Could be drug-specific
            independent = torch.randn(1, experiment_duration, 1)
            x_T[:, :, i] = (
                np.sqrt(correlation) * base_noise.squeeze() + 
                np.sqrt(1 - correlation) * independent.squeeze()
            )
    
    # Get drug-specific media dynamics
    drug_embedding = model.drug_encoder(drug)
    media_profile = model.media_dynamics.get_media_response_profile(
        drug_embedding, concentration
    )
    
    print(f"Drug {drug} media response profile:")
    print(f"  Recovery tau: {media_profile['recovery_tau'].item():.1f} hours")
    print(f"  Adaptation tau: {media_profile['adaptation_tau'].item():.1f} hours")
    
    # Reverse diffusion
    for t in reversed(range(num_timesteps)):
        # Media context with drug-specific dynamics
        media_context = model.media_dynamics(
            media_schedule,
            drug_embedding,
            concentration
        )
        
        # Denoise with replicate mask
        noise_pred = model.denoiser(
            x_T, t, media_context,
            replicate_mask,
            replicate_latents=None
        )
        
        x_T = reverse_diffusion_step(x_T, noise_pred, t)
    
    # Return only the requested number of replicates
    return x_T[:, :, :n_replicates]
```

### 8. Drug-Specific Media Response Learning

```python
class MediaResponseDataset(Dataset):
    """
    Dataset that emphasizes media change responses for learning drug-specific patterns.
    """
    def __getitem__(self, idx):
        data = self.data[idx]
        
        # Extract windows around media changes
        media_windows = []
        for change_time in data.media_changes:
            # Get 24-hour window: 2h before to 22h after
            start = max(0, change_time - 2)
            end = min(len(data.oxygen), change_time + 22)
            
            window = {
                'oxygen': data.oxygen[start:end],
                'time_relative': np.arange(start - change_time, end - change_time),
                'drug': data.drug,
                'concentration': data.concentration
            }
            media_windows.append(window)
        
        return {
            'full_series': data.oxygen,
            'media_windows': media_windows,
            'drug': data.drug,
            'concentration': data.concentration,
            'n_replicates': data.n_replicates
        }
```

### 9. Validation for Drug-Specific Responses

```python
def validate_drug_specific_media_responses(model, test_drugs):
    """
    Validate that model learns drug-specific media change patterns.
    """
    response_profiles = {}
    
    for drug in test_drugs:
        # Generate curves with media changes
        media_schedule = MediaChangeSchedule()
        media_schedule.add_change(48, drug_containing=True)  # Initial dosing
        media_schedule.add_change(120, drug_containing=True)  # First media change
        
        # Generate at multiple concentrations
        concentrations = [0.1, 1.0, 10.0]
        
        drug_responses = []
        for conc in concentrations:
            curves = generate_curves(
                model, drug, conc, media_schedule, n_replicates=4
            )
            
            # Analyze media change response
            pre_change = curves[:, 118:120, :].mean()
            peak = curves[:, 120:122, :].max()
            recovery_point = curves[:, 130:132, :].mean()
            
            response = {
                'spike_magnitude': (peak - pre_change).mean().item(),
                'recovery_percent': ((recovery_point - pre_change) / (peak - pre_change)).mean().item(),
                'concentration': conc
            }
            drug_responses.append(response)
        
        response_profiles[drug] = drug_responses
    
    # Verify drug-specific differences
    assert len(set([str(v) for v in response_profiles.values()])) > 1, \
        "All drugs showing identical responses!"
    
    return response_profiles
```

## Implementation Timeline

### Week 1: Flexible Architecture
- Implement adaptive replicate handling
- Create masked attention mechanisms
- Test with varying replicate counts

### Week 2: Drug-Specific Dynamics
- Implement DrugAwareMediaChangeModule
- Learn drug-specific recovery patterns
- Extract media response windows from data

### Week 3: Training Pipeline
- Handle missing replicate data
- Implement masked loss functions
- Train on full dataset with all variations

### Week 4: Validation & Analysis
- Validate drug-specific responses
- Test generation with 1-4 replicates
- Compare with real experimental data

## Why This Approach is Superior

1. **Flexible Replicate Handling**: Works with any number of replicates (1-4), matching real experimental conditions
2. **Drug-Specific Dynamics**: Learns unique media change responses for each drug rather than fixed timescales
3. **Adaptive Architecture**: Model complexity adjusts to available data
4. **Realistic Variability**: Generates appropriate correlation between replicates when multiple are requested
5. **Practical Training**: Can leverage all available data, even incomplete replicate sets

## Next Steps

1. Analyze replicate availability patterns in the dataset
2. Extract drug-specific media change response profiles
3. Implement flexible masking for missing replicates
4. Build adaptive UNet prototype with variable channels

This approach handles the full complexity of real experimental data while learning the unique dynamics of each drug's response to media changes.