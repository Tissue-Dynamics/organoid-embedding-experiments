#!/usr/bin/env python3
"""
Demonstrate flexible replicate handling with synthetic data showing 1-4 replicates.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

def create_synthetic_oxygen_curve(base_pattern, noise_level=2.0, spike_times=[72, 144, 216]):
    """Create a synthetic oxygen consumption curve with media change spikes."""
    time = np.arange(400)
    
    # Base oxygen level with slight drift
    oxygen = base_pattern + 0.05 * np.sin(time / 50) + np.random.randn(400) * noise_level
    
    # Add media change spikes
    for spike_time in spike_times:
        if spike_time < 400:
            # Create spike
            spike_width = 10
            spike_magnitude = np.random.uniform(5, 15)
            recovery_rate = np.random.uniform(0.1, 0.3)
            
            for t in range(spike_time, min(spike_time + 50, 400)):
                dt = t - spike_time
                if dt < spike_width:
                    oxygen[t] += spike_magnitude * np.exp(-dt * 0.5)
                else:
                    oxygen[t] -= spike_magnitude * recovery_rate * np.exp(-(dt - spike_width) * 0.1)
    
    return oxygen

def create_flexible_replicate_demo():
    """Create demonstration of flexible replicate handling."""
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define drug examples with different replicate counts
    examples = [
        {"drug": "DrugA_1rep", "n_reps": 1, "base": 20, "color": "red"},
        {"drug": "DrugB_2rep", "n_reps": 2, "base": 25, "color": "orange"},
        {"drug": "DrugC_3rep", "n_reps": 3, "base": 30, "color": "gold"},
        {"drug": "DrugD_4rep", "n_reps": 4, "base": 35, "color": "green"},
        {"drug": "DrugE_1rep", "n_reps": 1, "base": 40, "color": "red"},
        {"drug": "DrugF_4rep", "n_reps": 4, "base": 45, "color": "green"},
        {"drug": "DrugG_2rep", "n_reps": 2, "base": 50, "color": "orange"},
        {"drug": "DrugH_3rep", "n_reps": 3, "base": 55, "color": "gold"}
    ]
    
    # 1. Show replicate distribution
    ax1 = plt.subplot(4, 2, 1)
    rep_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for ex in examples:
        rep_counts[ex["n_reps"]] += 1
    
    bars = ax1.bar(rep_counts.keys(), rep_counts.values(), 
                    color=['red', 'orange', 'gold', 'green'])
    ax1.set_xlabel('Number of Replicates')
    ax1.set_ylabel('Count in Batch')
    ax1.set_title('Replicate Distribution in Synthetic Batch')
    ax1.set_xticks([1, 2, 3, 4])
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Show examples for each replicate count
    examples_by_rep = {1: [], 2: [], 3: [], 4: []}
    for ex in examples:
        examples_by_rep[ex["n_reps"]].append(ex)
    
    # Plot one example for each replicate count
    for rep_count in [1, 2, 3, 4]:
        if examples_by_rep[rep_count]:
            ax = plt.subplot(4, 2, 2 + rep_count)
            ex = examples_by_rep[rep_count][0]
            
            # Generate replicate curves
            colors = plt.cm.viridis(np.linspace(0, 1, rep_count))
            for i in range(rep_count):
                # Create correlated replicates
                base_curve = create_synthetic_oxygen_curve(ex["base"])
                if i > 0:
                    # Add replicate-specific variation
                    replicate_noise = np.random.randn(400) * 1.5
                    curve = 0.8 * base_curve + 0.2 * create_synthetic_oxygen_curve(ex["base"])
                else:
                    curve = base_curve
                
                ax.plot(curve, color=colors[i], alpha=0.8, 
                       label=f'Replicate {i+1}')
            
            # Mark media changes
            for mc_time in [72, 144, 216]:
                ax.axvline(mc_time, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Oxygen (%)')
            ax.set_title(f'{ex["drug"]} ({rep_count} replicate{"s" if rep_count > 1 else ""})')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 400)
    
    # 3. Visualize replicate mask
    ax3 = plt.subplot(4, 1, 4)
    
    # Create mask array
    mask_array = np.zeros((len(examples), 4))
    for i, ex in enumerate(examples):
        mask_array[i, :ex["n_reps"]] = 1
    
    im = ax3.imshow(mask_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xlabel('Replicate Index')
    ax3.set_ylabel('Sample in Batch')
    ax3.set_title('Replicate Mask Visualization (Green=Valid, Red=Missing)')
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(['Rep 1', 'Rep 2', 'Rep 3', 'Rep 4'])
    
    # Add drug names
    drug_labels = [ex["drug"] for ex in examples]
    ax3.set_yticks(range(len(examples)))
    ax3.set_yticklabels(drug_labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(examples)):
        for j in range(4):
            val = mask_array[i, j]
            text = ax3.text(j, i, '✓' if val > 0 else '✗',
                           ha="center", va="center", 
                           color="white" if val > 0 else "black",
                           fontsize=12, weight='bold')
    
    plt.colorbar(im, ax=ax3, ticks=[0, 1], label='Valid Replicate')
    
    plt.suptitle('Flexible Replicate Handling Demonstration\n(Synthetic Data)', 
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = results_dir / "synthetic_flexible_replicate_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved synthetic demonstration to: {output_path}")
    
    # Create tensor representation demo
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Show how data is represented as tensors
    ax = axes[0, 0]
    ax.text(0.5, 0.9, "Flexible Tensor Representation", 
            ha='center', va='top', fontsize=16, weight='bold',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.7, "Example: Drug with 2 replicates", 
            ha='left', va='top', fontsize=12,
            transform=ax.transAxes)
    
    ax.text(0.1, 0.6, "oxygen_tensor shape: [400, 4]", 
            ha='left', va='top', fontsize=11, family='monospace',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.5, "  [:, 0] = Replicate 1 data", 
            ha='left', va='top', fontsize=11, family='monospace',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.4, "  [:, 1] = Replicate 2 data", 
            ha='left', va='top', fontsize=11, family='monospace',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.3, "  [:, 2] = zeros (padding)", 
            ha='left', va='top', fontsize=11, family='monospace',
            color='gray', transform=ax.transAxes)
    
    ax.text(0.1, 0.2, "  [:, 3] = zeros (padding)", 
            ha='left', va='top', fontsize=11, family='monospace',
            color='gray', transform=ax.transAxes)
    
    ax.text(0.1, 0.05, "replicate_mask: [1, 1, 0, 0]", 
            ha='left', va='top', fontsize=11, family='monospace',
            color='darkgreen', transform=ax.transAxes)
    
    ax.axis('off')
    
    # Show masking operation
    ax = axes[0, 1]
    ax.text(0.5, 0.9, "Masked Operations", 
            ha='center', va='top', fontsize=16, weight='bold',
            transform=ax.transAxes)
    
    # Create visual representation
    data = np.random.randn(10, 4) * 5 + 30
    data[:, 2:] = 0  # Padding
    mask = np.array([1, 1, 0, 0])
    
    im = ax.imshow(data.T, cmap='coolwarm', aspect='auto')
    ax.set_ylabel('Replicate')
    ax.set_xlabel('Time')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Rep 1', 'Rep 2', 'Pad 3', 'Pad 4'])
    
    # Overlay mask
    for i in range(4):
        if mask[i] == 0:
            ax.axhspan(i-0.5, i+0.5, alpha=0.7, color='gray', zorder=10)
    
    ax.set_title("Padded regions are masked during computation")
    
    # Show loss computation
    ax = axes[1, 0]
    ax.text(0.5, 0.9, "Masked Loss Computation", 
            ha='center', va='top', fontsize=16, weight='bold',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.7, "Standard MSE:", 
            ha='left', va='top', fontsize=12,
            transform=ax.transAxes)
    
    ax.text(0.15, 0.6, "loss = mean((pred - target)²)", 
            ha='left', va='top', fontsize=11, family='monospace',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.4, "Masked MSE:", 
            ha='left', va='top', fontsize=12, color='darkgreen',
            transform=ax.transAxes)
    
    ax.text(0.15, 0.3, "masked_loss = (pred - target)² * mask", 
            ha='left', va='top', fontsize=11, family='monospace',
            transform=ax.transAxes)
    
    ax.text(0.15, 0.2, "loss = sum(masked_loss) / sum(mask)", 
            ha='left', va='top', fontsize=11, family='monospace',
            transform=ax.transAxes)
    
    ax.text(0.1, 0.05, "Only valid replicates contribute to loss!", 
            ha='left', va='top', fontsize=11, color='darkgreen',
            weight='bold', transform=ax.transAxes)
    
    ax.axis('off')
    
    # Show generation flexibility
    ax = axes[1, 1]
    ax.text(0.5, 0.9, "Flexible Generation", 
            ha='center', va='top', fontsize=16, weight='bold',
            transform=ax.transAxes)
    
    # Create mini examples
    for i, n_reps in enumerate([1, 2, 3, 4]):
        y_pos = 0.7 - i * 0.15
        ax.text(0.1, y_pos, f"Request {n_reps} rep{'s' if n_reps > 1 else ''}:", 
                ha='left', va='top', fontsize=11,
                transform=ax.transAxes)
        
        # Draw mini curves
        x = np.linspace(0.4, 0.9, 50)
        for j in range(n_reps):
            y = y_pos - 0.02 - j * 0.02
            curve = 0.02 * np.sin(10 * x + j) + y
            ax.plot(x, curve, linewidth=2, 
                   color=plt.cm.viridis(j/3),
                   transform=ax.transAxes)
    
    ax.axis('off')
    
    plt.suptitle('Flexible Replicate System Architecture', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save architecture figure
    arch_path = results_dir / "flexible_replicate_architecture.png"
    plt.savefig(arch_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved architecture diagram to: {arch_path}")
    
    return output_path, arch_path

def main():
    """Create synthetic demonstration of flexible replicate handling."""
    print("="*80)
    print("SYNTHETIC FLEXIBLE REPLICATE DEMONSTRATION")
    print("="*80)
    
    demo_path, arch_path = create_flexible_replicate_demo()
    
    print("\n✅ Demonstration complete!")
    print(f"Check visualizations at:")
    print(f"  - {demo_path}")
    print(f"  - {arch_path}")

if __name__ == "__main__":
    main()