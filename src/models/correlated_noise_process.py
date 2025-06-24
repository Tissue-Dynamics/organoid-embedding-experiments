#!/usr/bin/env python3
"""
Correlated noise process for diffusion model with variable replicates.
Ensures realistic correlation between replicates during the diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

class CorrelatedNoiseScheduler:
    """Noise scheduler that maintains correlation between replicates."""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 correlation_start=0.9, correlation_end=0.5):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta for noise schedule
            beta_end: Ending beta for noise schedule
            correlation_start: Initial correlation between replicates
            correlation_end: Final correlation between replicates
        """
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Correlation schedule - decreases over time
        self.correlations = torch.linspace(correlation_start, correlation_end, num_timesteps)
        
    def get_correlation_matrix(self, n_replicates, correlation):
        """Create correlation matrix for replicates."""
        if n_replicates == 1:
            return torch.tensor([[1.0]])
        
        # Create correlation matrix
        corr_matrix = torch.eye(n_replicates)
        for i in range(n_replicates):
            for j in range(n_replicates):
                if i != j:
                    corr_matrix[i, j] = correlation
        
        return corr_matrix
    
    def generate_correlated_noise(self, shape, replicate_mask, timestep):
        """
        Generate correlated noise for replicates.
        
        Args:
            shape: [batch, time, max_replicates] shape of the noise
            replicate_mask: [batch, max_replicates] mask of valid replicates
            timestep: [batch] current timestep
        Returns:
            noise: [batch, time, max_replicates] correlated noise
        """
        batch_size, time_len, max_reps = shape
        device = replicate_mask.device
        
        # Initialize noise tensor
        noise = torch.zeros(shape, device=device)
        
        # Get correlation for this timestep
        correlation = self.correlations[timestep[0]]  # Assume same timestep for batch
        
        # Process each sample in batch
        for b in range(batch_size):
            n_valid_reps = int(replicate_mask[b].sum().item())
            
            if n_valid_reps == 0:
                continue
            
            # Get correlation matrix
            corr_matrix = self.get_correlation_matrix(n_valid_reps, correlation)
            
            # Generate correlated noise using Cholesky decomposition
            L = torch.linalg.cholesky(corr_matrix)
            
            # Generate independent noise
            independent_noise = torch.randn(time_len, n_valid_reps, device=device)
            
            # Apply correlation
            correlated = torch.matmul(independent_noise, L.T)
            
            # Fill in the noise tensor
            noise[b, :, :n_valid_reps] = correlated
        
        # Apply mask
        noise = noise * replicate_mask.unsqueeze(1)
        
        return noise
    
    def q_sample(self, x_start, t, noise=None, replicate_mask=None):
        """
        Forward diffusion process - add noise to data.
        
        Args:
            x_start: [batch, time, max_replicates] original data
            t: [batch] timestep
            noise: Optional pre-generated noise
            replicate_mask: [batch, max_replicates] mask of valid replicates
        Returns:
            noisy_data: [batch, time, max_replicates]
        """
        if noise is None:
            if replicate_mask is None:
                noise = torch.randn_like(x_start)
            else:
                noise = self.generate_correlated_noise(x_start.shape, replicate_mask, t)
        
        # Extract coefficients for timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        # Add noise according to schedule
        noisy_data = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Apply mask
        if replicate_mask is not None:
            noisy_data = noisy_data * replicate_mask.unsqueeze(1)
        
        return noisy_data

def visualize_noise_correlation():
    """Visualize the correlated noise process."""
    
    # Create noise scheduler
    scheduler = CorrelatedNoiseScheduler(
        num_timesteps=1000,
        correlation_start=0.95,
        correlation_end=0.3
    )
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Correlation schedule
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(scheduler.correlations.numpy(), linewidth=2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Replicate Correlation')
    ax1.set_title('Correlation Schedule During Diffusion')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Beta schedule
    ax2 = plt.subplot(4, 4, 2)
    ax2.plot(scheduler.betas.numpy(), label='β', linewidth=2)
    ax2.plot(scheduler.alphas.numpy(), label='α', linewidth=2)
    ax2.plot(scheduler.alphas_cumprod.numpy(), label='ᾱ', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Value')
    ax2.set_title('Diffusion Schedule Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Noise visualization at different timesteps
    timesteps_to_show = [0, 250, 500, 750]
    
    for idx, t in enumerate(timesteps_to_show):
        ax = plt.subplot(4, 4, 3 + idx)
        
        # Generate noise for 4 replicates
        replicate_mask = torch.ones(1, 4)
        noise = scheduler.generate_correlated_noise(
            (1, 100, 4), replicate_mask, torch.tensor([t])
        )
        
        # Plot each replicate
        for rep in range(4):
            ax.plot(noise[0, :, rep], alpha=0.7, label=f'Rep {rep+1}')
        
        ax.set_title(f'Noise at t={t} (ρ={scheduler.correlations[t]:.2f})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Noise Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 4. Correlation matrix visualization
    for idx, n_reps in enumerate([2, 3, 4]):
        ax = plt.subplot(4, 4, 7 + idx)
        
        corr_matrix = scheduler.get_correlation_matrix(n_reps, 0.7)
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add text annotations
        for i in range(n_reps):
            for j in range(n_reps):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                       ha='center', va='center')
        
        ax.set_title(f'{n_reps} Replicates (ρ=0.7)')
        ax.set_xticks(range(n_reps))
        ax.set_yticks(range(n_reps))
        plt.colorbar(im, ax=ax)
    
    # 5. Forward diffusion process
    ax = plt.subplot(4, 4, 10)
    
    # Create synthetic data
    time_points = torch.linspace(0, 100, 200)
    original_data = 30 + 5 * torch.sin(2 * np.pi * time_points / 50)
    original_data = original_data.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4)
    
    # Add slight variation between replicates
    for rep in range(4):
        original_data[0, :, rep] += torch.randn(200) * 0.5
    
    # Show diffusion at different timesteps
    timesteps_forward = [0, 100, 300, 500, 700, 900]
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_forward)))
    
    replicate_mask = torch.ones(1, 4)
    
    for t_idx, t in enumerate(timesteps_forward):
        t_tensor = torch.tensor([t])
        noisy = scheduler.q_sample(original_data, t_tensor, replicate_mask=replicate_mask)
        
        # Plot mean across replicates
        mean_signal = noisy[0].mean(dim=1)
        ax.plot(mean_signal, color=colors[t_idx], alpha=0.8, 
               label=f't={t}', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Oxygen (%)')
    ax.set_title('Forward Diffusion Process')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Replicate correlation analysis
    ax = plt.subplot(4, 4, 11)
    
    # Calculate actual correlations at different timesteps
    actual_correlations = []
    
    for t in range(0, 1000, 50):
        t_tensor = torch.tensor([t])
        noise = scheduler.generate_correlated_noise(
            (1, 1000, 4), replicate_mask, t_tensor
        )
        
        # Calculate correlation between first two replicates
        corr = np.corrcoef(noise[0, :, 0].numpy(), noise[0, :, 1].numpy())[0, 1]
        actual_correlations.append(corr)
    
    timesteps_sample = list(range(0, 1000, 50))
    ax.plot(timesteps_sample, actual_correlations, 'o-', label='Actual', linewidth=2)
    ax.plot(scheduler.correlations.numpy(), '--', label='Target', linewidth=2, alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Correlation')
    ax.set_title('Target vs Actual Replicate Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Variable replicate handling
    for idx, n_reps in enumerate([1, 2, 3, 4]):
        ax = plt.subplot(4, 4, 12 + idx)
        
        # Create mask for n_reps
        mask = torch.zeros(1, 4)
        mask[0, :n_reps] = 1.0
        
        # Generate noise
        noise = scheduler.generate_correlated_noise(
            (1, 100, 4), mask, torch.tensor([500])
        )
        
        # Plot
        for rep in range(4):
            if rep < n_reps:
                ax.plot(noise[0, :, rep], alpha=0.8, label=f'Rep {rep+1}')
            else:
                ax.plot(noise[0, :, rep], '--', color='gray', alpha=0.3, 
                       label=f'Rep {rep+1} (masked)')
        
        ax.set_title(f'{n_reps} Active Replicate{"s" if n_reps > 1 else ""}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Noise')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
    
    plt.suptitle('Correlated Noise Process for Replicate-Aware Diffusion', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    viz_path = results_dir / "correlated_noise_process.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create second figure for detailed analysis
    fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. Noise distribution analysis
    ax = axes[0, 0]
    
    # Generate large batch of noise
    noise_batch = scheduler.generate_correlated_noise(
        (100, 100, 4), torch.ones(100, 4), torch.tensor([500] * 100)
    )
    
    # Flatten and plot distribution
    noise_flat = noise_batch[noise_batch != 0].flatten().numpy()
    ax.hist(noise_flat, bins=50, density=True, alpha=0.7, label='Generated')
    
    # Overlay standard normal
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x), 'r--', linewidth=2, label='N(0,1)')
    
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.set_title('Noise Distribution Check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Correlation heatmap over time
    ax = axes[0, 1]
    
    # Calculate correlation matrix at different timesteps
    timesteps_corr = list(range(0, 1000, 100))
    corr_matrix_time = np.zeros((4, 4, len(timesteps_corr)))
    
    for t_idx, t in enumerate(timesteps_corr):
        noise_t = scheduler.generate_correlated_noise(
            (1, 10000, 4), torch.ones(1, 4), torch.tensor([t])
        )
        
        # Calculate full correlation matrix
        for i in range(4):
            for j in range(4):
                if i == j:
                    corr_matrix_time[i, j, t_idx] = 1.0
                else:
                    corr = np.corrcoef(noise_t[0, :, i].numpy(), 
                                      noise_t[0, :, j].numpy())[0, 1]
                    corr_matrix_time[i, j, t_idx] = corr
    
    # Show correlation between rep 0 and 1 over time
    im = ax.imshow(corr_matrix_time[:, :, :].mean(axis=2), cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_title('Average Correlation Matrix')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xlabel('Replicate')
    ax.set_ylabel('Replicate')
    plt.colorbar(im, ax=ax)
    
    # 3. Signal-to-noise ratio
    ax = axes[0, 2]
    
    snr_values = []
    for t in range(0, 1000, 50):
        # Signal strength (from schedule)
        signal_strength = scheduler.sqrt_alphas_cumprod[t]
        noise_strength = scheduler.sqrt_one_minus_alphas_cumprod[t]
        snr = (signal_strength / noise_strength).item()
        snr_values.append(snr)
    
    ax.semilogy(range(0, 1000, 50), snr_values, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('SNR During Diffusion')
    ax.grid(True, alpha=0.3)
    
    # 4. Replicate variance evolution
    ax = axes[1, 0]
    
    # Track variance between replicates
    var_between_reps = []
    timesteps_var = list(range(0, 1000, 100))
    
    for t in timesteps_var:
        t_tensor = torch.tensor([t])
        noisy = scheduler.q_sample(original_data, t_tensor, replicate_mask=replicate_mask)
        
        # Calculate variance across replicates at each time point
        var_t = noisy[0].var(dim=1).mean().item()
        var_between_reps.append(var_t)
    
    ax.plot(timesteps_var, var_between_reps, 'o-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variance Between Replicates')
    ax.set_title('Replicate Variance During Diffusion')
    ax.grid(True, alpha=0.3)
    
    # 5. Effective noise levels
    ax = axes[1, 1]
    
    # Show how noise affects different numbers of replicates
    for n_reps in [1, 2, 3, 4]:
        mask = torch.zeros(1, 4)
        mask[0, :n_reps] = 1.0
        
        effective_noise = []
        for t in range(0, 1000, 100):
            noise = scheduler.generate_correlated_noise(
                (1, 1000, 4), mask, torch.tensor([t])
            )
            
            # Calculate RMS noise
            rms = torch.sqrt((noise ** 2).mean()).item()
            effective_noise.append(rms)
        
        ax.plot(range(0, 1000, 100), effective_noise, 
               label=f'{n_reps} rep{"s" if n_reps > 1 else ""}', linewidth=2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('RMS Noise Level')
    ax.set_title('Effective Noise vs Replicate Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Correlation preservation test
    ax = axes[1, 2]
    
    # Test if correlation is preserved through diffusion
    original_corr = 0.8
    test_data = torch.randn(1, 1000, 4)
    
    # Make replicates correlated
    for i in range(1, 4):
        test_data[0, :, i] = original_corr * test_data[0, :, 0] + \
                            np.sqrt(1 - original_corr**2) * test_data[0, :, i]
    
    preserved_corr = []
    for t in range(0, 1000, 100):
        t_tensor = torch.tensor([t])
        noisy = scheduler.q_sample(test_data, t_tensor, replicate_mask=torch.ones(1, 4))
        
        # Measure correlation in noisy data
        corr = np.corrcoef(noisy[0, :, 0].numpy(), noisy[0, :, 1].numpy())[0, 1]
        preserved_corr.append(corr)
    
    ax.plot(range(0, 1000, 100), preserved_corr, 'o-', linewidth=2)
    ax.axhline(original_corr, color='red', linestyle='--', 
              label=f'Original correlation: {original_corr}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Correlation in Noisy Data')
    ax.set_title('Correlation Preservation Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7-9: Empty for now
    for i in range(3):
        axes[2, i].axis('off')
    
    plt.suptitle('Correlated Noise Process: Detailed Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    analysis_path = results_dir / "noise_correlation_analysis.png"
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualizations:")
    print(f"  - {viz_path}")
    print(f"  - {analysis_path}")
    
    return viz_path, analysis_path

def test_noise_scheduler():
    """Test the correlated noise scheduler."""
    
    print("=== TESTING CORRELATED NOISE SCHEDULER ===")
    
    scheduler = CorrelatedNoiseScheduler()
    
    # Test 1: Basic noise generation
    print("\nTest 1: Basic noise generation")
    
    shape = (4, 100, 4)
    mask = torch.tensor([
        [1, 1, 1, 1],  # 4 replicates
        [1, 1, 1, 0],  # 3 replicates
        [1, 1, 0, 0],  # 2 replicates
        [1, 0, 0, 0],  # 1 replicate
    ]).float()
    
    noise = scheduler.generate_correlated_noise(shape, mask, torch.tensor([500] * 4))
    
    print(f"✓ Noise shape: {noise.shape}")
    
    # Check masking
    for b in range(4):
        n_valid = int(mask[b].sum().item())
        for rep in range(4):
            if rep >= n_valid:
                assert noise[b, :, rep].abs().sum() == 0, f"Unmasked noise at batch {b}, rep {rep}"
    
    print("✓ Masking verified")
    
    # Test 2: Correlation check
    print("\nTest 2: Correlation verification")
    
    # Generate noise with high correlation
    high_corr_scheduler = CorrelatedNoiseScheduler(correlation_start=0.9, correlation_end=0.9)
    noise_high = high_corr_scheduler.generate_correlated_noise(
        (1, 10000, 4), torch.ones(1, 4), torch.tensor([0])
    )
    
    # Check correlation between replicates
    corr_01 = np.corrcoef(noise_high[0, :, 0].numpy(), noise_high[0, :, 1].numpy())[0, 1]
    corr_02 = np.corrcoef(noise_high[0, :, 0].numpy(), noise_high[0, :, 2].numpy())[0, 1]
    
    print(f"  Correlation Rep0-Rep1: {corr_01:.3f} (target: 0.9)")
    print(f"  Correlation Rep0-Rep2: {corr_02:.3f} (target: 0.9)")
    
    assert abs(corr_01 - 0.9) < 0.1, f"Correlation too far from target: {corr_01}"
    print("✓ Correlations match target")
    
    # Test 3: Forward diffusion
    print("\nTest 3: Forward diffusion process")
    
    x_start = torch.randn(2, 50, 4)
    t = torch.tensor([100, 500])
    
    x_noisy = scheduler.q_sample(x_start, t, replicate_mask=torch.ones(2, 4))
    
    assert x_noisy.shape == x_start.shape, "Shape mismatch"
    print(f"✓ Forward diffusion shape: {x_noisy.shape}")
    
    # Test 4: Different timesteps
    print("\nTest 4: Timestep-dependent correlation")
    
    correlations_measured = []
    for t in [0, 250, 500, 750, 999]:
        noise_t = scheduler.generate_correlated_noise(
            (1, 5000, 4), torch.ones(1, 4), torch.tensor([t])
        )
        corr_t = np.corrcoef(noise_t[0, :, 0].numpy(), noise_t[0, :, 1].numpy())[0, 1]
        correlations_measured.append(corr_t)
        print(f"  t={t}: correlation={corr_t:.3f}, target={scheduler.correlations[t]:.3f}")
    
    # Check that correlation decreases with timestep
    assert correlations_measured[0] > correlations_measured[-1], "Correlation should decrease"
    print("✓ Correlation decreases with timestep")
    
    print("\n✅ All tests passed!")
    
    return True

def main():
    """Test and visualize the correlated noise process."""
    print("="*80)
    print("CORRELATED NOISE PROCESS FOR DIFFUSION")
    print("="*80)
    
    # Test
    success = test_noise_scheduler()
    
    if success:
        # Visualize
        print("\nCreating visualizations...")
        viz_path, analysis_path = visualize_noise_correlation()
        
        print("\n✅ Correlated noise process complete!")
        print(f"Key features:")
        print(f"  ✓ Maintains correlation between replicates")
        print(f"  ✓ Correlation decreases during diffusion")
        print(f"  ✓ Handles variable replicate counts")
        print(f"  ✓ Preserves statistical properties")
        print(f"  ✓ Integrates with diffusion schedule")

if __name__ == "__main__":
    main()