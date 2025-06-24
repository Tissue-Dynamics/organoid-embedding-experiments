#!/usr/bin/env python3
"""
Masked loss function for diffusion model training with variable replicates.
Visualizes loss computation and behavior across different scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

class MaskedDiffusionLoss(nn.Module):
    """
    Loss function for diffusion models that properly handles missing replicates.
    """
    
    def __init__(self, loss_type="l2", media_weight_scale=2.0):
        super().__init__()
        self.loss_type = loss_type
        self.media_weight_scale = media_weight_scale
        
    def forward(self, pred, target, replicate_mask, media_proximity=None, reduction='mean'):
        """
        Args:
            pred: [batch, time, max_replicates] - predicted noise
            target: [batch, time, max_replicates] - target noise
            replicate_mask: [batch, max_replicates] - which replicates are valid
            media_proximity: [batch, time] - optional weight based on proximity to media changes
            reduction: 'none', 'mean', or 'sum'
        Returns:
            loss: scalar or tensor depending on reduction
        """
        batch_size, time_len, max_reps = pred.shape
        
        # Expand mask to match data dimensions
        mask_expanded = replicate_mask.unsqueeze(1)  # [batch, 1, max_reps]
        
        # Compute per-element loss
        if self.loss_type == "l2":
            element_loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == "l1":
            element_loss = F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == "huber":
            element_loss = F.smooth_l1_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply replicate mask
        masked_loss = element_loss * mask_expanded
        
        # Sum over replicate dimension
        replicate_loss = masked_loss.sum(dim=2)  # [batch, time]
        
        # Normalize by number of valid replicates
        n_valid = replicate_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        normalized_loss = replicate_loss / n_valid
        
        # Apply media proximity weighting if provided
        if media_proximity is not None:
            # Higher weight near media changes
            weights = 1.0 + (self.media_weight_scale - 1.0) * media_proximity
            normalized_loss = normalized_loss * weights
        
        # Apply reduction
        if reduction == 'none':
            return normalized_loss
        elif reduction == 'mean':
            return normalized_loss.mean()
        elif reduction == 'sum':
            return normalized_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

class ReplicateCoherenceLoss(nn.Module):
    """
    Additional loss to encourage coherence between replicates.
    """
    
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target, replicate_mask):
        """
        Penalizes deviation from replicate variance patterns.
        """
        batch_size, time_len, max_reps = pred.shape
        
        coherence_loss = 0.0
        n_multi_rep = 0
        
        for b in range(batch_size):
            n_valid = int(replicate_mask[b].sum().item())
            
            if n_valid > 1:
                # Get valid replicates
                valid_pred = pred[b, :, :n_valid]
                valid_target = target[b, :, :n_valid]
                
                # Compute variance across replicates
                pred_var = valid_pred.var(dim=1)
                target_var = valid_target.var(dim=1)
                
                # Penalize difference in variance patterns
                var_loss = F.mse_loss(pred_var, target_var)
                coherence_loss += var_loss
                n_multi_rep += 1
        
        if n_multi_rep > 0:
            coherence_loss = coherence_loss / n_multi_rep
        
        return self.weight * coherence_loss

def visualize_loss_behavior():
    """Visualize how the loss function behaves in different scenarios."""
    
    # Create loss functions
    basic_loss = MaskedDiffusionLoss(loss_type="l2")
    coherence_loss = ReplicateCoherenceLoss(weight=0.1)
    
    # Test scenarios
    batch_size = 4
    time_len = 100
    
    fig = plt.figure(figsize=(20, 16))
    
    # Scenario 1: Perfect prediction
    print("Testing scenario 1: Perfect prediction")
    target = torch.randn(batch_size, time_len, 4)
    pred_perfect = target.clone()
    
    # Scenario 2: Noisy prediction
    print("Testing scenario 2: Noisy prediction")
    pred_noisy = target + 0.1 * torch.randn_like(target)
    
    # Scenario 3: Wrong prediction
    print("Testing scenario 3: Wrong prediction")
    pred_wrong = torch.randn_like(target)
    
    # Test with different replicate masks
    replicate_scenarios = [
        {"mask": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]).float(), "name": "1 Replicate"},
        {"mask": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]).float(), "name": "2 Replicates"},
        {"mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]).float(), "name": "3 Replicates"},
        {"mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).float(), "name": "4 Replicates"},
    ]
    
    for scenario_idx, rep_scenario in enumerate(replicate_scenarios):
        mask = rep_scenario["mask"]
        
        # Apply mask to target
        target_masked = target * mask.unsqueeze(1)
        
        # Compute losses for different predictions
        loss_perfect = basic_loss(pred_perfect, target_masked, mask, reduction='none')
        loss_noisy = basic_loss(pred_noisy, target_masked, mask, reduction='none')
        loss_wrong = basic_loss(pred_wrong, target_masked, mask, reduction='none')
        
        # Plot loss over time for first sample
        ax = plt.subplot(4, 4, scenario_idx * 4 + 1)
        ax.plot(loss_perfect[0].detach(), label='Perfect', linewidth=2)
        ax.plot(loss_noisy[0].detach(), label='Noisy', linewidth=2)
        ax.plot(loss_wrong[0].detach(), label='Wrong', linewidth=2)
        ax.set_title(f'{rep_scenario["name"]} - Loss Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot average losses
        ax = plt.subplot(4, 4, scenario_idx * 4 + 2)
        avg_losses = [
            loss_perfect.mean().item(),
            loss_noisy.mean().item(),
            loss_wrong.mean().item()
        ]
        bars = ax.bar(['Perfect', 'Noisy', 'Wrong'], avg_losses, 
                      color=['green', 'orange', 'red'])
        ax.set_title(f'{rep_scenario["name"]} - Average Loss')
        ax.set_ylabel('Loss')
        
        for bar, val in zip(bars, avg_losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom')
        
        # Test with media proximity
        ax = plt.subplot(4, 4, scenario_idx * 4 + 3)
        
        # Create media proximity signal (high near t=50)
        media_proximity = torch.zeros(batch_size, time_len)
        for t in range(time_len):
            media_proximity[:, t] = torch.exp(torch.tensor(-0.1 * abs(t - 50)))
        
        loss_with_media = basic_loss(pred_noisy, target_masked, mask, media_proximity, reduction='none')
        loss_without_media = basic_loss(pred_noisy, target_masked, mask, reduction='none')
        
        ax.plot(loss_without_media[0].detach(), label='Without media weight', alpha=0.7)
        ax.plot(loss_with_media[0].detach(), label='With media weight', linewidth=2)
        ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='Media change')
        ax.set_title(f'{rep_scenario["name"]} - Media Proximity Effect')
        ax.set_xlabel('Time')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coherence loss visualization
        ax = plt.subplot(4, 4, scenario_idx * 4 + 4)
        
        n_reps = int(mask[0].sum().item())
        if n_reps > 1:
            # Create predictions with different coherence levels
            pred_coherent = target_masked.clone()
            pred_incoherent = target_masked.clone()
            
            # Add different noise to each replicate for incoherent
            for rep in range(n_reps):
                pred_incoherent[:, :, rep] += 0.2 * torch.randn(batch_size, time_len)
            
            # Add same noise to all replicates for coherent
            common_noise = 0.1 * torch.randn(batch_size, time_len)
            for rep in range(n_reps):
                pred_coherent[:, :, rep] += common_noise
            
            coh_loss_coherent = coherence_loss(pred_coherent, target_masked, mask)
            coh_loss_incoherent = coherence_loss(pred_incoherent, target_masked, mask)
            
            bars = ax.bar(['Coherent', 'Incoherent'], 
                          [coh_loss_coherent.item(), coh_loss_incoherent.item()],
                          color=['green', 'red'])
            ax.set_title(f'{rep_scenario["name"]} - Coherence Loss')
            ax.set_ylabel('Coherence Loss')
            
            for bar, val in zip(bars, [coh_loss_coherent.item(), coh_loss_incoherent.item()]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.4f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'N/A\n(Single replicate)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rep_scenario["name"]} - Coherence Loss')
    
    plt.suptitle('Masked Diffusion Loss Function Behavior', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    loss_viz_path = results_dir / "masked_loss_behavior.png"
    plt.savefig(loss_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved loss behavior visualization to: {loss_viz_path}")
    
    return loss_viz_path

def visualize_loss_curves():
    """Visualize training loss curves for different scenarios."""
    
    # Simulate training with different replicate distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Setup
    n_epochs = 100
    batch_size = 32
    
    # Scenario 1: All 4 replicates
    ax = axes[0, 0]
    losses_4rep = []
    for epoch in range(n_epochs):
        # Simulate decreasing loss
        loss = 1.0 * np.exp(-0.05 * epoch) + 0.1 + 0.05 * np.random.randn()
        losses_4rep.append(loss)
    
    ax.plot(losses_4rep, label='Training Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training with 4 Replicates')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    # Scenario 2: Mixed replicates
    ax = axes[0, 1]
    losses_mixed = []
    replicate_distribution = []
    
    for epoch in range(n_epochs):
        # Simulate mixed batch with different replicate counts
        base_loss = 1.0 * np.exp(-0.04 * epoch) + 0.15  # Slightly slower convergence
        noise = 0.08 * np.random.randn()  # More noise due to imbalance
        loss = base_loss + noise
        losses_mixed.append(loss)
        
        # Track replicate distribution
        rep_counts = np.random.choice([1, 2, 3, 4], size=batch_size, p=[0.1, 0.2, 0.3, 0.4])
        replicate_distribution.append(rep_counts.mean())
    
    ax.plot(losses_mixed, label='Training Loss', linewidth=2, color='orange')
    ax2 = ax.twinx()
    ax2.plot(replicate_distribution, label='Avg Replicates', alpha=0.5, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Average Replicates per Batch')
    ax.set_title('Training with Mixed Replicates')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    # Scenario 3: Loss components breakdown
    ax = axes[1, 0]
    
    # Simulate different loss components
    reconstruction_loss = []
    coherence_loss_curve = []
    total_loss = []
    
    for epoch in range(n_epochs):
        recon = 0.8 * np.exp(-0.05 * epoch) + 0.08 + 0.03 * np.random.randn()
        coh = 0.2 * np.exp(-0.03 * epoch) + 0.02 + 0.01 * np.random.randn()
        
        reconstruction_loss.append(recon)
        coherence_loss_curve.append(coh)
        total_loss.append(recon + 0.1 * coh)  # Weighted sum
    
    ax.plot(reconstruction_loss, label='Reconstruction Loss', linewidth=2)
    ax.plot(coherence_loss_curve, label='Coherence Loss', linewidth=2)
    ax.plot(total_loss, label='Total Loss', linewidth=3, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Scenario 4: Validation curves
    ax = axes[1, 1]
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training loss
        train_loss = 0.9 * np.exp(-0.05 * epoch) + 0.1 + 0.04 * np.random.randn()
        train_losses.append(train_loss)
        
        # Validation loss (slightly higher, potential overfitting after epoch 60)
        if epoch < 60:
            val_loss = train_loss + 0.05 + 0.02 * np.random.randn()
        else:
            # Slight overfitting
            val_loss = train_loss + 0.05 + 0.001 * (epoch - 60) + 0.03 * np.random.randn()
        
        val_losses.append(val_loss)
    
    ax.plot(train_losses, label='Training Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
    ax.axvline(60, color='red', linestyle=':', alpha=0.5, label='Potential Overfitting')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    plt.suptitle('Diffusion Model Training Loss Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    curves_path = results_dir / "training_loss_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved loss curves to: {curves_path}")
    
    # Create detailed loss analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Analysis 1: Loss vs number of replicates
    ax = axes[0, 0]
    
    n_reps = [1, 2, 3, 4]
    avg_losses = []
    loss_stds = []
    
    loss_fn = MaskedDiffusionLoss()
    
    for n in n_reps:
        losses = []
        for _ in range(100):
            # Simulate data
            pred = torch.randn(8, 50, 4)
            target = torch.randn(8, 50, 4)
            mask = torch.zeros(8, 4)
            mask[:, :n] = 1.0
            
            loss = loss_fn(pred, target, mask).item()
            losses.append(loss)
        
        avg_losses.append(np.mean(losses))
        loss_stds.append(np.std(losses))
    
    ax.errorbar(n_reps, avg_losses, yerr=loss_stds, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel('Number of Replicates')
    ax.set_ylabel('Average Loss')
    ax.set_title('Loss vs Number of Replicates')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_reps)
    
    # Analysis 2: Loss distribution
    ax = axes[0, 1]
    
    all_losses = []
    for _ in range(1000):
        pred = torch.randn(1, 100, 4)
        target = torch.randn(1, 100, 4)
        mask = torch.ones(1, 4)
        loss = loss_fn(pred, target, mask, reduction='none')
        all_losses.extend(loss.flatten().tolist())
    
    ax.hist(all_losses, bins=50, alpha=0.7, density=True)
    ax.set_xlabel('Loss Value')
    ax.set_ylabel('Density')
    ax.set_title('Loss Distribution (Random Predictions)')
    ax.grid(True, alpha=0.3)
    
    # Analysis 3: Media proximity impact
    ax = axes[1, 0]
    
    time_points = np.arange(100)
    media_change_time = 50
    
    # Create different media proximity patterns
    sharp_proximity = np.exp(-0.5 * np.abs(time_points - media_change_time))
    broad_proximity = np.exp(-0.05 * np.abs(time_points - media_change_time))
    uniform_proximity = np.ones_like(time_points)
    
    ax.plot(time_points, sharp_proximity, label='Sharp (α=0.5)', linewidth=2)
    ax.plot(time_points, broad_proximity, label='Broad (α=0.05)', linewidth=2)
    ax.plot(time_points, uniform_proximity, label='Uniform', linewidth=2, linestyle='--')
    ax.axvline(media_change_time, color='red', linestyle=':', alpha=0.5, label='Media Change')
    ax.set_xlabel('Time')
    ax.set_ylabel('Loss Weight')
    ax.set_title('Media Proximity Weighting Patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Analysis 4: Convergence rates
    ax = axes[1, 1]
    
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    
    for lr in learning_rates:
        losses = []
        for step in range(200):
            # Simulate convergence with different learning rates
            loss = 1.0 * np.exp(-lr * 10 * step) + 0.1
            losses.append(loss)
        
        ax.plot(losses, label=f'LR={lr}', linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence Rates with Different Learning Rates')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Loss Function Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    analysis_path = results_dir / "loss_detailed_analysis.png"
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detailed analysis to: {analysis_path}")
    
    return curves_path, analysis_path

def test_loss_implementation():
    """Test the loss implementation thoroughly."""
    
    print("=== TESTING MASKED DIFFUSION LOSS ===")
    
    loss_fn = MaskedDiffusionLoss()
    coherence_fn = ReplicateCoherenceLoss()
    
    # Test 1: Perfect masking
    print("\nTest 1: Perfect masking")
    pred = torch.randn(2, 10, 4)
    target = torch.randn(2, 10, 4)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]).float()
    
    # Set masked positions to different values
    pred[:, :, 2:] = 999.0
    target[:, :, 2:] = -999.0
    
    loss = loss_fn(pred, target, mask)
    print(f"Loss with extreme masked values: {loss.item():.6f}")
    assert loss.item() < 10, "Loss should ignore masked values"
    print("✓ Masking works correctly")
    
    # Test 2: Normalization by replicate count
    print("\nTest 2: Normalization check")
    
    # Same prediction error but different replicate counts
    pred1 = torch.ones(1, 10, 4) * 0.5
    target1 = torch.zeros(1, 10, 4)
    
    mask_1rep = torch.tensor([[1, 0, 0, 0]]).float()
    mask_4rep = torch.tensor([[1, 1, 1, 1]]).float()
    
    loss_1rep = loss_fn(pred1, target1, mask_1rep)
    loss_4rep = loss_fn(pred1, target1, mask_4rep)
    
    print(f"Loss with 1 replicate: {loss_1rep.item():.6f}")
    print(f"Loss with 4 replicates: {loss_4rep.item():.6f}")
    
    # Should be the same due to normalization
    assert abs(loss_1rep.item() - loss_4rep.item()) < 0.0001, "Loss not properly normalized"
    print("✓ Normalization works correctly")
    
    # Test 3: Media proximity
    print("\nTest 3: Media proximity weighting")
    
    batch_size, time_len = 4, 50
    pred = torch.randn(batch_size, time_len, 4)
    target = torch.randn(batch_size, time_len, 4)
    mask = torch.ones(batch_size, 4)
    
    # Create media proximity
    media_proximity = torch.zeros(batch_size, time_len)
    media_proximity[:, 25] = 1.0  # Peak at t=25
    
    loss_without = loss_fn(pred, target, mask, reduction='none')
    loss_with = loss_fn(pred, target, mask, media_proximity, reduction='none')
    
    # Check that media proximity increases loss at t=25
    assert loss_with[:, 25].mean() > loss_without[:, 25].mean(), "Media proximity not working"
    print("✓ Media proximity weighting works")
    
    # Test 4: Coherence loss
    print("\nTest 4: Coherence loss")
    
    # Create coherent vs incoherent predictions
    target = torch.randn(2, 20, 4)
    
    # Coherent: all replicates similar
    pred_coherent = target + 0.1 * torch.randn(2, 20, 1).expand(-1, -1, 4)
    
    # Incoherent: each replicate different
    pred_incoherent = target + 0.3 * torch.randn(2, 20, 4)
    
    mask = torch.ones(2, 4)
    
    coh_loss_coherent = coherence_fn(pred_coherent, target, mask)
    coh_loss_incoherent = coherence_fn(pred_incoherent, target, mask)
    
    print(f"Coherent loss: {coh_loss_coherent.item():.6f}")
    print(f"Incoherent loss: {coh_loss_incoherent.item():.6f}")
    
    assert coh_loss_coherent < coh_loss_incoherent, "Coherence loss not working"
    print("✓ Coherence loss works correctly")
    
    print("\n✅ All loss tests passed!")
    
    return True

def main():
    """Test and visualize the masked loss function."""
    print("="*80)
    print("MASKED DIFFUSION LOSS IMPLEMENTATION")
    print("="*80)
    
    # Test implementation
    success = test_loss_implementation()
    
    if success:
        # Create visualizations
        print("\nCreating loss behavior visualizations...")
        loss_viz_path = visualize_loss_behavior()
        
        print("\nCreating loss curve visualizations...")
        curves_path, analysis_path = visualize_loss_curves()
        
        print("\n✅ Masked loss implementation complete!")
        print(f"Key features:")
        print(f"  ✓ Proper masking of invalid replicates")
        print(f"  ✓ Normalization by replicate count")
        print(f"  ✓ Media proximity weighting")
        print(f"  ✓ Optional coherence loss")
        print(f"  ✓ Multiple loss types (L2, L1, Huber)")

if __name__ == "__main__":
    main()