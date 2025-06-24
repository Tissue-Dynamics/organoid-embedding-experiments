#!/usr/bin/env python3
"""
Simplified test of generation components.
Focus on visualizing realistic behavior without complex diffusion.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from simple_drug_aware_module import SimpleDrugAwareModule
from correlated_noise_process import CorrelatedNoiseScheduler

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

class SimpleOxygenGenerator:
    """Simple oxygen curve generator for testing."""
    
    def __init__(self, n_drugs=20):
        self.n_drugs = n_drugs
        self.drug_module = SimpleDrugAwareModule(n_drugs=n_drugs)
        self.noise_scheduler = CorrelatedNoiseScheduler()
        
    def generate_curves(self, drug_ids, concentrations, replicate_counts, 
                       media_change_times, time_length=200):
        """
        Generate realistic oxygen curves.
        
        Args:
            drug_ids: [batch] drug indices  
            concentrations: [batch, 1] concentrations
            replicate_counts: [batch] number of replicates per sample
            media_change_times: list of media change times
            time_length: number of time points
        Returns:
            oxygen_curves: [batch, time, max_replicates]
            replicate_masks: [batch, max_replicates]
        """
        batch_size = len(drug_ids)
        max_replicates = 4
        
        # Create replicate masks
        replicate_masks = torch.zeros(batch_size, max_replicates)
        for i, count in enumerate(replicate_counts):
            replicate_masks[i, :count] = 1.0
        
        # Time points
        time_points = torch.linspace(0, 200, time_length).unsqueeze(0).repeat(batch_size, 1)
        
        # Get drug-specific responses
        with torch.no_grad():
            drug_responses, params = self.drug_module(
                drug_ids, concentrations, time_points, media_change_times
            )
        
        # Create baseline oxygen levels (realistic range)
        baseline_oxygen = 30.0  # Normal oxygen around 30%
        
        # Generate correlated noise for replicates
        noise_shape = (batch_size, time_length, max_replicates)
        timestep = torch.tensor([500] * batch_size)  # Mid-range correlation
        
        with torch.no_grad():
            replicate_noise = self.noise_scheduler.generate_correlated_noise(
                noise_shape, replicate_masks, timestep
            )
        
        # Build final curves
        oxygen_curves = torch.zeros(noise_shape)
        
        for i in range(batch_size):
            n_reps = int(replicate_counts[i])
            
            # Base curve for this sample
            base_curve = baseline_oxygen + drug_responses[i]
            
            # Add realistic biological variation
            biological_noise = torch.randn(time_length) * 0.5
            base_curve += biological_noise
            
            # Create replicates with shared and independent variation
            for rep in range(n_reps):
                # Shared variation (same for all replicates)
                shared_variation = replicate_noise[i, :, rep] * 0.8
                
                # Replicate-specific variation
                rep_specific = torch.randn(time_length) * 0.3
                
                # Combine
                oxygen_curves[i, :, rep] = base_curve + shared_variation + rep_specific
                
                # Ensure realistic range
                oxygen_curves[i, :, rep] = torch.clamp(oxygen_curves[i, :, rep], 15, 45)
        
        # Apply masking
        oxygen_curves = oxygen_curves * replicate_masks.unsqueeze(1)
        
        return oxygen_curves, replicate_masks, params

def visualize_simple_generation():
    """Test and visualize the simple generation process."""
    
    print("=== TESTING SIMPLE OXYGEN GENERATION ===")
    
    generator = SimpleOxygenGenerator(n_drugs=20)
    
    # Create test conditions
    test_scenarios = [
        {
            'name': 'Variable Replicates - Same Drug',
            'drug_ids': torch.tensor([0, 0, 0, 0]),
            'concentrations': torch.tensor([[10.0], [10.0], [10.0], [10.0]]),
            'replicate_counts': [1, 2, 3, 4],
            'media_changes': [72, 144]
        },
        {
            'name': 'Different Drugs - Same Conditions',
            'drug_ids': torch.tensor([0, 5, 10, 15]),
            'concentrations': torch.tensor([[10.0], [10.0], [10.0], [10.0]]),
            'replicate_counts': [4, 4, 4, 4],
            'media_changes': [72, 144]
        },
        {
            'name': 'Concentration Response - Same Drug',
            'drug_ids': torch.tensor([5, 5, 5, 5]),
            'concentrations': torch.tensor([[0.1], [1.0], [10.0], [100.0]]),
            'replicate_counts': [3, 3, 3, 3],
            'media_changes': [72, 144]
        },
        {
            'name': 'Mixed Realistic Conditions',
            'drug_ids': torch.tensor([2, 7, 12, 18]),
            'concentrations': torch.tensor([[1.0], [5.0], [50.0], [20.0]]),
            'replicate_counts': [2, 4, 3, 1],
            'media_changes': [48, 96, 168]
        }
    ]
    
    # Create main visualization
    fig = plt.figure(figsize=(24, 20))
    
    for scenario_idx, scenario in enumerate(test_scenarios):
        print(f"\nGenerating scenario: {scenario['name']}")
        
        # Generate curves
        curves, masks, params = generator.generate_curves(
            scenario['drug_ids'],
            scenario['concentrations'], 
            scenario['replicate_counts'],
            scenario['media_changes']
        )
        
        # Plot each condition in the scenario
        for i in range(4):
            ax = plt.subplot(4, 4, scenario_idx * 4 + i + 1)
            
            time_points = torch.linspace(0, 200, 200)
            n_reps = scenario['replicate_counts'][i]
            
            # Plot each replicate
            colors = plt.cm.Set1(np.linspace(0, 1, 4))
            replicate_curves = []
            
            for rep in range(4):
                if rep < n_reps:
                    curve = curves[i, :, rep]
                    ax.plot(time_points, curve, color=colors[rep], 
                           alpha=0.8, linewidth=2, label=f'Rep {rep+1}')
                    replicate_curves.append(curve)
                else:
                    # Show that masked replicates are zero
                    ax.plot(time_points, curves[i, :, rep], 
                           color='gray', alpha=0.3, linestyle='--',
                           label=f'Rep {rep+1} (masked)')
            
            # Add mean and std if multiple replicates
            if n_reps > 1:
                replicate_stack = torch.stack(replicate_curves)
                mean_curve = replicate_stack.mean(dim=0)
                std_curve = replicate_stack.std(dim=0)
                
                ax.fill_between(time_points, 
                               mean_curve - std_curve,
                               mean_curve + std_curve,
                               alpha=0.2, color='black', label='Mean ± SD')
                ax.plot(time_points, mean_curve, 'k--', linewidth=2, label='Mean')
            
            # Mark media changes
            for mc_time in scenario['media_changes']:
                ax.axvline(mc_time, color='red', linestyle=':', alpha=0.7)
            
            # Labels and formatting
            drug_id = scenario['drug_ids'][i].item()
            conc = scenario['concentrations'][i].item()
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Oxygen (%)')
            ax.set_title(f'Drug {drug_id}, {conc:.1f} μM, {n_reps} reps')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(15, 45)
    
    plt.suptitle('Simple Oxygen Generation: Realistic Replicate Curves', 
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save main plot
    main_path = results_dir / "simple_generation_test.png"
    plt.savefig(main_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create analysis figure
    fig2 = plt.figure(figsize=(20, 16))
    
    # Analysis 1: Correlation matrix for 4 replicates
    ax1 = plt.subplot(3, 4, 1)
    
    # Generate sample with 4 replicates
    test_curves, _, _ = generator.generate_curves(
        torch.tensor([0]), torch.tensor([[10.0]]), [4], [72]
    )
    
    # Calculate correlation matrix
    curves_np = test_curves[0].numpy().T  # [4, time]
    corr_matrix = np.corrcoef(curves_np)
    
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    ax1.set_title('Replicate Correlations')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    
    # Add correlation values
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im1, ax=ax1)
    
    # Analysis 2: Variance components
    ax2 = plt.subplot(3, 4, 2)
    
    # Generate multiple samples to analyze variance
    batch_curves, _, _ = generator.generate_curves(
        torch.tensor([0] * 10), torch.ones(10, 1) * 10.0, [4] * 10, [72]
    )
    
    time_points = torch.linspace(0, 200, 200)
    
    # Variance across samples vs across replicates
    sample_var = batch_curves[:, :, :4].var(dim=0).mean(dim=1)  # Variance across samples
    rep_var = batch_curves[:, :, :4].var(dim=2).mean(dim=0)     # Variance across replicates
    
    ax2.plot(time_points, sample_var, label='Sample-to-sample', linewidth=2)
    ax2.plot(time_points, rep_var, label='Replicate-to-replicate', linewidth=2)
    ax2.axvline(72, color='red', linestyle=':', alpha=0.7, label='Media change')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Analysis 3: Drug parameter effects
    ax3 = plt.subplot(3, 4, 3)
    
    # Compare several drugs
    drug_comparison = torch.tensor([0, 5, 10, 15])
    drug_curves, _, drug_params = generator.generate_curves(
        drug_comparison, torch.ones(4, 1) * 10.0, [3] * 4, [72]
    )
    
    # Plot mean responses
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for i, drug_id in enumerate(drug_comparison):
        mean_curve = drug_curves[i, :, :3].mean(dim=1)
        recovery_tau = drug_params['recovery_tau'][i].item()
        
        ax3.plot(time_points, mean_curve, color=colors[i], 
                linewidth=2, label=f'Drug {drug_id} (τ={recovery_tau:.1f}h)')
    
    ax3.axvline(72, color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Mean Oxygen (%)')
    ax3.set_title('Drug-Specific Recovery')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Analysis 4: Concentration effects
    ax4 = plt.subplot(3, 4, 4)
    
    conc_test = torch.tensor([[0.1], [1.0], [10.0], [100.0]])
    conc_curves, _, conc_params = generator.generate_curves(
        torch.tensor([5] * 4), conc_test, [3] * 4, [72]
    )
    
    for i, conc in enumerate(conc_test):
        mean_curve = conc_curves[i, :, :3].mean(dim=1)
        spike_height = conc_params['spike_height'][i].item()
        
        ax4.plot(time_points, mean_curve, linewidth=2, 
                label=f'{conc.item():.1f} μM (spike={spike_height:.1f})')
    
    ax4.axvline(72, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Mean Oxygen (%)')
    ax4.set_title('Concentration Response')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Analysis 5-8: Individual replicate patterns by count
    for n_reps in range(1, 5):
        ax = plt.subplot(3, 4, 4 + n_reps)
        
        # Generate with specific replicate count
        rep_curves, rep_masks, _ = generator.generate_curves(
            torch.tensor([8]), torch.tensor([[25.0]]), [n_reps], [60, 120]
        )
        
        colors = plt.cm.Set1(np.linspace(0, 1, 4))
        for rep in range(4):
            if rep < n_reps:
                ax.plot(time_points, rep_curves[0, :, rep], 
                       color=colors[rep], alpha=0.8, linewidth=2,
                       label=f'Rep {rep+1}')
            else:
                # Verify masking
                max_val = rep_curves[0, :, rep].abs().max().item()
                ax.plot(time_points, rep_curves[0, :, rep], 
                       color='gray', alpha=0.3, linestyle='--',
                       label=f'Rep {rep+1} (max: {max_val:.3f})')
        
        for mc_time in [60, 120]:
            ax.axvline(mc_time, color='red', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Oxygen (%)')
        ax.set_title(f'{n_reps} Active Replicate{"s" if n_reps > 1 else ""}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(15, 45)
    
    # Analysis 9-12: Statistical validation
    ax9 = plt.subplot(3, 4, 9)
    
    # Distribution of oxygen values
    large_batch_curves, _, _ = generator.generate_curves(
        torch.randint(0, 20, (20,)), torch.rand(20, 1) * 100, [4] * 20, [72]
    )
    
    # Flatten all values
    all_values = large_batch_curves[large_batch_curves != 0].flatten().numpy()
    
    ax9.hist(all_values, bins=50, density=True, alpha=0.7, label='Generated')
    ax9.axvline(all_values.mean(), color='red', linestyle='--', 
               label=f'Mean: {all_values.mean():.1f}%')
    ax9.set_xlabel('Oxygen (%)')
    ax9.set_ylabel('Density')
    ax9.set_title('Oxygen Value Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Correlation statistics
    ax10 = plt.subplot(3, 4, 10)
    
    correlations = []
    for i in range(10):
        test_c, _, _ = generator.generate_curves(
            torch.tensor([i]), torch.tensor([[15.0]]), [4], [80]
        )
        
        # Get all pairwise correlations
        c_np = test_c[0].numpy().T
        for j in range(4):
            for k in range(j+1, 4):
                corr = np.corrcoef(c_np[j], c_np[k])[0, 1]
                correlations.append(corr)
    
    ax10.hist(correlations, bins=20, density=True, alpha=0.7)
    ax10.axvline(np.mean(correlations), color='red', linestyle='--',
                label=f'Mean: {np.mean(correlations):.3f}')
    ax10.set_xlabel('Replicate Correlation')
    ax10.set_ylabel('Density')
    ax10.set_title('Replicate Correlation Distribution')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Empty plots for now
    for i in [11, 12]:
        ax = plt.subplot(3, 4, i)
        ax.axis('off')
    
    plt.suptitle('Simple Generation: Statistical Analysis', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save analysis
    analysis_path = results_dir / "simple_generation_analysis.png"
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualizations:")
    print(f"  - Main: {main_path}")
    print(f"  - Analysis: {analysis_path}")
    
    # Print statistics
    print(f"\nGeneration Statistics:")
    print(f"  Mean replicate correlation: {np.mean(correlations):.3f}")
    print(f"  Oxygen range: [{all_values.min():.1f}, {all_values.max():.1f}]%")
    print(f"  Oxygen mean ± std: {all_values.mean():.1f} ± {all_values.std():.1f}%")
    
    return main_path, analysis_path

def test_generation_quality():
    """Test key quality metrics of the generation."""
    
    print("\n=== GENERATION QUALITY TESTS ===")
    
    generator = SimpleOxygenGenerator(n_drugs=20)
    
    # Test 1: Masking works correctly
    print("\nTest 1: Replicate masking")
    
    curves, masks, _ = generator.generate_curves(
        torch.tensor([0, 1, 2]), torch.ones(3, 1) * 10.0, [1, 2, 4], [72]
    )
    
    for i, n_reps in enumerate([1, 2, 4]):
        for rep in range(4):
            if rep >= n_reps:
                max_val = curves[i, :, rep].abs().max().item()
                assert max_val < 1e-6, f"Unmasked replicate {rep} in sample {i}: {max_val}"
    
    print("✓ Masking verified")
    
    # Test 2: Realistic oxygen range
    print("\nTest 2: Oxygen value range")
    
    large_batch, _, _ = generator.generate_curves(
        torch.randint(0, 20, (10,)), torch.rand(10, 1) * 100, [4] * 10, [72]
    )
    
    active_values = large_batch[large_batch != 0].flatten()
    min_val, max_val = active_values.min().item(), active_values.max().item()
    
    print(f"  Range: [{min_val:.1f}, {max_val:.1f}]%")
    assert 15 < min_val < 35, f"Minimum too extreme: {min_val}"
    assert 20 < max_val < 50, f"Maximum too extreme: {max_val}"
    print("✓ Realistic range")
    
    # Test 3: Replicate correlations
    print("\nTest 3: Replicate correlations")
    
    corr_test, _, _ = generator.generate_curves(
        torch.tensor([5]), torch.tensor([[10.0]]), [4], [80]
    )
    
    corr_np = corr_test[0].numpy().T
    correlations = []
    for i in range(4):
        for j in range(i+1, 4):
            corr = np.corrcoef(corr_np[i], corr_np[j])[0, 1]
            correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    print(f"  Mean correlation: {mean_corr:.3f}")
    assert 0.2 < mean_corr < 0.9, f"Correlation out of realistic range: {mean_corr}"
    print("✓ Realistic correlations")
    
    # Test 4: Drug differences
    print("\nTest 4: Drug differences")
    
    drug_test, _, _ = generator.generate_curves(
        torch.tensor([0, 5, 10, 15]), torch.ones(4, 1) * 10.0, [3] * 4, [72]
    )
    
    # Compare mean responses
    mean_responses = drug_test[:, :, :3].mean(dim=2)  # [drugs, time]
    drug_variance = mean_responses.var(dim=0).max().item()
    
    print(f"  Max drug variance: {drug_variance:.3f}")
    assert drug_variance > 1.0, f"Insufficient drug variation: {drug_variance}"
    print("✓ Drug differences present")
    
    print("\n✅ All quality tests passed!")
    
    return True

def main():
    """Run the simple generation test."""
    print("=" * 80)
    print("SIMPLE OXYGEN GENERATION TEST")
    print("=" * 80)
    
    # Test quality
    success = test_generation_quality()
    
    if success:
        # Create visualizations
        print("\nCreating visualizations...")
        main_path, analysis_path = visualize_simple_generation()
        
        print("\n✅ Simple generation complete!")
        print("Key features demonstrated:")
        print("  ✓ Realistic oxygen curves (15-45%)")
        print("  ✓ Proper replicate masking")
        print("  ✓ Correlated replicate variation")
        print("  ✓ Drug-specific responses")
        print("  ✓ Concentration dependence")
        print("  ✓ Media change integration")

if __name__ == "__main__":
    main()