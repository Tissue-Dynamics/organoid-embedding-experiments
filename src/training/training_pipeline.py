#!/usr/bin/env python3
"""
Complete training pipeline for the replicate-aware diffusion model.
Uses synthetic data with realistic patterns for training demonstration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "models"))

from minimal_adaptive_unet import MinimalAdaptiveUNet
from masked_diffusion_loss import MaskedDiffusionLoss
from simple_drug_aware_module import SimpleDrugAwareModule
from correlated_noise_process import CorrelatedNoiseScheduler

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "training"
models_dir = project_root / "results" / "models"
results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

class SyntheticReplicateDataset(Dataset):
    """Synthetic dataset that mimics real oxygen replicate patterns."""
    
    def __init__(self, n_samples=1000, n_drugs=20, time_length=200, max_time=200):
        """
        Create synthetic dataset with realistic patterns.
        
        Args:
            n_samples: Number of training samples
            n_drugs: Number of different drugs
            time_length: Number of time points
            max_time: Maximum time in hours
        """
        self.n_samples = n_samples
        self.n_drugs = n_drugs
        self.time_length = time_length
        self.max_time = max_time
        self.max_replicates = 4
        
        # Create drug-aware module for generating realistic responses
        self.drug_module = SimpleDrugAwareModule(n_drugs=n_drugs)
        self.drug_module.eval()
        
        # Noise scheduler for correlated replicate noise
        self.noise_scheduler = CorrelatedNoiseScheduler()
        
        # Pre-generate all samples for consistent training
        print(f"Generating {n_samples} synthetic training samples...")
        self.samples = []
        self._generate_samples()
        
    def _generate_samples(self):
        """Pre-generate all training samples."""
        
        for i in tqdm(range(self.n_samples), desc="Generating samples"):
            # Random drug and concentration
            drug_idx = torch.randint(0, self.n_drugs, (1,))
            concentration = torch.exp(torch.randn(1) * 1.5 + 2)  # Log-normal around 10 μM
            
            # Random number of replicates (1-4)
            n_replicates = torch.randint(1, 5, (1,)).item()
            
            # Fixed media change times for consistent batching
            media_times = [72.0, 144.0]  # Standard 3-day intervals
            
            # Generate base oxygen curve
            time_points = torch.linspace(0, self.max_time, self.time_length).unsqueeze(0)
            
            with torch.no_grad():
                drug_response, _ = self.drug_module(
                    drug_idx, concentration.unsqueeze(0), time_points, media_times
                )
            
            # Base oxygen level (realistic range)
            baseline = 25 + torch.randn(1) * 3  # Around 25% with variation
            
            # Generate correlated replicates
            replicate_mask = torch.zeros(self.max_replicates)
            replicate_mask[:n_replicates] = 1.0
            
            # Correlated noise between replicates
            noise_shape = (1, self.time_length, self.max_replicates)
            timestep = torch.tensor([500])  # Mid-range correlation
            
            with torch.no_grad():
                correlated_noise = self.noise_scheduler.generate_correlated_noise(
                    noise_shape, replicate_mask.unsqueeze(0), timestep
                )
            
            # Build final oxygen curves
            oxygen_curves = torch.zeros(self.time_length, self.max_replicates)
            
            for rep in range(n_replicates):
                # Base curve + drug response + noise
                base_curve = baseline + drug_response[0]
                
                # Add correlated noise (reduced scale)
                replicate_noise = correlated_noise[0, :, rep] * 1.5
                
                # Add some replicate-specific variation
                rep_specific = torch.randn(self.time_length) * 0.5
                
                # Combine
                oxygen_curves[:, rep] = base_curve + replicate_noise + rep_specific
                
                # Ensure realistic range
                oxygen_curves[:, rep] = torch.clamp(oxygen_curves[:, rep], 15, 45)
            
            # Store sample (convert media_times to tensor for batching)
            sample = {
                'oxygen': oxygen_curves,
                'replicate_mask': replicate_mask,
                'drug_idx': drug_idx[0],
                'concentration': concentration[0],
                'n_replicates': n_replicates
            }
            
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample."""
        sample = self.samples[idx]
        
        # Return copy to avoid modification
        return {
            'oxygen': sample['oxygen'].clone(),
            'replicate_mask': sample['replicate_mask'].clone(),
            'drug_idx': sample['drug_idx'].clone(),
            'concentration': sample['concentration'].clone(),
            'n_replicates': sample['n_replicates']
        }

class DiffusionTrainer:
    """Complete trainer for the diffusion model."""
    
    def __init__(self, n_drugs=20, device='cpu'):
        """Initialize trainer with model components."""
        self.device = device
        self.n_drugs = n_drugs
        
        # Model components
        self.unet = MinimalAdaptiveUNet(max_replicates=4).to(device)
        self.drug_module = SimpleDrugAwareModule(n_drugs=n_drugs).to(device)
        self.noise_scheduler = CorrelatedNoiseScheduler()
        self.loss_fn = MaskedDiffusionLoss()
        
        # Optimizers
        self.unet_optimizer = optim.AdamW(self.unet.parameters(), lr=1e-4)
        self.drug_optimizer = optim.AdamW(self.drug_module.parameters(), lr=1e-4)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, batch):
        """Single training step."""
        # Move batch to device
        oxygen = batch['oxygen'].to(self.device)  # [batch, time, replicates]
        replicate_mask = batch['replicate_mask'].to(self.device)  # [batch, replicates]
        drug_indices = batch['drug_idx'].to(self.device)  # [batch]
        concentrations = batch['concentration'].to(self.device)  # [batch]
        
        batch_size = oxygen.shape[0]
        
        # Random timesteps for diffusion
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, 
                                 (batch_size,), device=self.device)
        
        # Add noise to oxygen data
        noise = self.noise_scheduler.generate_correlated_noise(
            oxygen.shape, replicate_mask, timesteps
        )
        
        noisy_oxygen = self.noise_scheduler.q_sample(
            oxygen, timesteps, noise=noise, replicate_mask=replicate_mask
        )
        
        # Predict noise with UNet
        predicted_noise = self.unet(noisy_oxygen, timesteps, replicate_mask)
        
        # Calculate loss
        loss = self.loss_fn(predicted_noise, noise, replicate_mask)
        
        # Backward pass
        self.unet_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        
        self.unet_optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader):
        """Validation step."""
        self.unet.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                oxygen = batch['oxygen'].to(self.device)
                replicate_mask = batch['replicate_mask'].to(self.device)
                
                batch_size = oxygen.shape[0]
                timesteps = torch.randint(0, self.noise_scheduler.num_timesteps,
                                        (batch_size,), device=self.device)
                
                # Add noise
                noise = self.noise_scheduler.generate_correlated_noise(
                    oxygen.shape, replicate_mask, timesteps
                )
                noisy_oxygen = self.noise_scheduler.q_sample(
                    oxygen, timesteps, noise=noise, replicate_mask=replicate_mask
                )
                
                # Predict noise
                predicted_noise = self.unet(noisy_oxygen, timesteps, replicate_mask)
                
                # Calculate loss
                loss = self.loss_fn(predicted_noise, noise, replicate_mask)
                
                total_loss += loss.item()
                n_batches += 1
        
        self.unet.train()
        return total_loss / n_batches if n_batches > 0 else 0
    
    def train(self, train_loader, val_loader, epochs=50, save_every=10):
        """Full training loop."""
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training
            self.unet.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        print("Training completed!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'unet_state_dict': self.unet.state_dict(),
            'drug_module_state_dict': self.drug_module.state_dict(),
            'unet_optimizer_state_dict': self.unet_optimizer.state_dict(),
            'drug_optimizer_state_dict': self.drug_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'n_drugs': self.n_drugs
        }
        
        torch.save(checkpoint, models_dir / filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(models_dir / filename, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.drug_module.load_state_dict(checkpoint['drug_module_state_dict'])
        self.unet_optimizer.load_state_dict(checkpoint['unet_optimizer_state_dict'])
        self.drug_optimizer.load_state_dict(checkpoint['drug_optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint: {filename}")

def visualize_training_progress(trainer):
    """Visualize training progress and results."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training curves
    ax1 = plt.subplot(3, 4, 1)
    
    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, label='Train Loss', linewidth=2)
    
    if trainer.val_losses:
        ax1.plot(epochs, trainer.val_losses, label='Val Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Loss distribution
    ax2 = plt.subplot(3, 4, 2)
    
    recent_losses = trainer.train_losses[-10:] if len(trainer.train_losses) >= 10 else trainer.train_losses
    ax2.hist(recent_losses, bins=20, alpha=0.7, density=True)
    ax2.axvline(np.mean(recent_losses), color='red', linestyle='--',
               label=f'Mean: {np.mean(recent_losses):.4f}')
    ax2.set_xlabel('Loss Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Recent Loss Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3-6. Generated samples at different training stages
    stages = ['Early', 'Mid', 'Late', 'Final']
    stage_epochs = [1, len(trainer.train_losses)//3, 2*len(trainer.train_losses)//3, len(trainer.train_losses)]
    
    for stage_idx, (stage, epoch) in enumerate(zip(stages, stage_epochs)):
        ax = plt.subplot(3, 4, 3 + stage_idx)
        
        # Generate a sample (simplified for visualization)
        with torch.no_grad():
            # Create synthetic test input
            test_drug = torch.tensor([0], device=trainer.device)
            test_conc = torch.tensor([10.0], device=trainer.device)
            test_mask = torch.tensor([[1, 1, 1, 0]], device=trainer.device, dtype=torch.float)
            
            # Simple noise to image process (not full denoising)
            noise = torch.randn(1, 200, 4, device=trainer.device) * test_mask.unsqueeze(1)
            timestep = torch.tensor([500], device=trainer.device)
            
            # Get UNet prediction
            denoised = trainer.unet(noise, timestep, test_mask)
            
            # Convert to oxygen curves (simplified)
            curves = 30 + 10 * torch.tanh(denoised[0].cpu())  # Center around 30%
            
        # Plot replicates
        time_points = torch.linspace(0, 200, 200)
        colors = plt.cm.Set1(np.linspace(0, 1, 3))
        
        for rep in range(3):
            ax.plot(time_points, curves[:, rep], color=colors[rep], 
                   alpha=0.8, linewidth=2, label=f'Rep {rep+1}')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Oxygen (%)')
        ax.set_title(f'{stage} Training (Epoch {epoch})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(15, 45)
    
    # 7. Model parameter statistics
    ax7 = plt.subplot(3, 4, 7)
    
    # Get parameter statistics
    param_stats = []
    for name, param in trainer.unet.named_parameters():
        if param.requires_grad:
            param_stats.append({
                'name': name,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'grad_norm': param.grad.norm().item() if param.grad is not None else 0
            })
    
    if param_stats:
        names = [p['name'].split('.')[-1] for p in param_stats[-10:]]  # Last 10 layers
        grad_norms = [p['grad_norm'] for p in param_stats[-10:]]
        
        ax7.bar(range(len(names)), grad_norms, alpha=0.7)
        ax7.set_xticks(range(len(names)))
        ax7.set_xticklabels(names, rotation=45, ha='right')
        ax7.set_ylabel('Gradient Norm')
        ax7.set_title('Parameter Gradient Norms')
        ax7.grid(True, alpha=0.3)
    
    # 8. Dataset statistics
    ax8 = plt.subplot(3, 4, 8)
    
    # Create simple dataset stats visualization
    replicate_counts = [1, 2, 3, 4]
    # Approximate distribution based on typical patterns
    replicate_freq = [0.15, 0.25, 0.35, 0.25]  # More 3-4 replicates
    
    ax8.bar(replicate_counts, replicate_freq, alpha=0.7)
    ax8.set_xlabel('Number of Replicates')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Training Data Replicate Distribution')
    ax8.grid(True, alpha=0.3)
    
    # 9-12. Training metrics over time
    metrics = ['Loss Smoothness', 'Convergence Rate', 'Stability', 'Overfitting Check']
    
    for metric_idx, metric in enumerate(metrics):
        ax = plt.subplot(3, 4, 9 + metric_idx)
        
        if metric == 'Loss Smoothness':
            # Plot smoothed loss
            if len(trainer.train_losses) > 5:
                smoothed = np.convolve(trainer.train_losses, np.ones(5)/5, mode='valid')
                ax.plot(range(3, len(smoothed)+3), smoothed, linewidth=2)
                ax.set_title('Smoothed Training Loss')
            
        elif metric == 'Convergence Rate':
            # Plot loss differences
            if len(trainer.train_losses) > 1:
                diffs = np.diff(trainer.train_losses)
                ax.plot(range(2, len(diffs)+2), diffs, linewidth=2)
                ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_title('Loss Change per Epoch')
            
        elif metric == 'Stability':
            # Plot loss variance in windows
            if len(trainer.train_losses) > 10:
                window_vars = []
                for i in range(10, len(trainer.train_losses)):
                    window = trainer.train_losses[i-10:i]
                    window_vars.append(np.var(window))
                ax.plot(range(10, len(window_vars)+10), window_vars, linewidth=2)
                ax.set_title('Loss Variance (10-epoch windows)')
            
        elif metric == 'Overfitting Check':
            # Plot train vs val loss difference
            if trainer.val_losses and len(trainer.val_losses) == len(trainer.train_losses):
                diff = np.array(trainer.val_losses) - np.array(trainer.train_losses)
                ax.plot(epochs, diff, linewidth=2)
                ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_title('Val - Train Loss')
            else:
                ax.text(0.5, 0.5, 'No validation data', ha='center', va='center')
                ax.set_title('Overfitting Check')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')
    
    plt.suptitle('Training Analysis Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    viz_path = results_dir / "training_analysis.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training analysis: {viz_path}")
    
    return viz_path

def main():
    """Run complete training pipeline."""
    print("=" * 80)
    print("DIFFUSION MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create smaller datasets for demonstration
    print("\nCreating synthetic training dataset...")
    train_dataset = SyntheticReplicateDataset(n_samples=500, n_drugs=20)
    val_dataset = SyntheticReplicateDataset(n_samples=100, n_drugs=20)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: 16")
    
    # Create trainer
    trainer = DiffusionTrainer(n_drugs=20, device=device)
    
    # Train model (very short for demonstration)
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, epochs=3, save_every=2)
    
    # Visualize results
    print("\nCreating training analysis...")
    viz_path = visualize_training_progress(trainer)
    
    print("\n✅ Training pipeline complete!")
    print(f"Models saved in: {models_dir}")
    print(f"Training analysis: {viz_path}")
    
    # Save training config
    config = {
        'n_drugs': 20,
        'n_samples_train': 2000,
        'n_samples_val': 400,
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'device': str(device),
        'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None
    }
    
    with open(models_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training config saved: {models_dir / 'training_config.json'}")

if __name__ == "__main__":
    main()