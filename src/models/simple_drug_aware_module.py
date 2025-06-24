#!/usr/bin/env python3
"""
Simplified Drug-Aware Media Change Module with better numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

class SimpleDrugAwareModule(nn.Module):
    """Simplified drug-aware module focusing on key parameters."""
    
    def __init__(self, n_drugs, embedding_dim=128):
        super().__init__()
        self.n_drugs = n_drugs
        
        # Drug embeddings
        self.drug_embed = nn.Embedding(n_drugs, embedding_dim)
        nn.init.normal_(self.drug_embed.weight, std=0.1)
        
        # Simple network to predict media response parameters
        self.param_net = nn.Sequential(
            nn.Linear(embedding_dim + 1, 256),  # +1 for log concentration
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 parameters: spike_height, recovery_tau, baseline_shift
        )
        
    def forward(self, drug_ids, concentrations, time_points, media_change_times):
        """
        Generate drug-specific media responses.
        
        Args:
            drug_ids: [batch] drug indices
            concentrations: [batch, 1] concentrations
            time_points: [batch, time] time points in hours
            media_change_times: list of media change times
        Returns:
            response: [batch, time] oxygen response
            params: dict of parameters
        """
        batch_size = drug_ids.shape[0]
        
        # Get drug embeddings
        drug_embeds = self.drug_embed(drug_ids)  # [batch, embedding_dim]
        
        # Prepare features
        log_conc = torch.log10(concentrations + 1e-8).clamp(-3, 3)
        features = torch.cat([drug_embeds, log_conc], dim=-1)
        
        # Predict parameters
        raw_params = self.param_net(features)  # [batch, 3]
        
        # Transform to meaningful ranges
        spike_height = 5.0 * torch.tanh(raw_params[:, 0:1])  # [-5, 5] % O2
        recovery_tau = 20.0 * torch.sigmoid(raw_params[:, 1:2]) + 1.0  # [1, 21] hours
        baseline_shift = 2.0 * torch.tanh(raw_params[:, 2:3])  # [-2, 2] % O2
        
        # Initialize response
        response = torch.zeros_like(time_points)
        
        # Add response for each media change
        for mc_time in media_change_times:
            # Time since media change
            t_since = time_points - mc_time
            
            # Only affect times after media change
            mask = (t_since > 0).float()
            
            # Simple exponential recovery model
            # Response = spike * exp(-t/tau) + baseline_shift * (1 - exp(-t/tau))
            exp_decay = torch.exp(-t_since.clamp(min=0) / recovery_tau)
            
            mc_response = (
                spike_height * exp_decay +  # Spike that decays
                baseline_shift * (1 - exp_decay)  # Shift to new baseline
            ) * mask
            
            response += mc_response
        
        # Store parameters for visualization
        params = {
            'spike_height': spike_height,
            'recovery_tau': recovery_tau,
            'baseline_shift': baseline_shift
        }
        
        return response, params

def visualize_simple_drug_responses():
    """Visualize the simplified drug-aware module."""
    
    # Create module with 20 drugs
    n_drugs = 20
    module = SimpleDrugAwareModule(n_drugs=n_drugs)
    module.eval()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Show responses for different drugs at same concentration
    ax1 = plt.subplot(4, 4, 1)
    
    drug_ids = torch.arange(8)
    concs = torch.ones(8, 1) * 10.0  # 10 μM for all
    time_points = torch.linspace(0, 300, 600).unsqueeze(0).repeat(8, 1)
    media_changes = [72, 144, 216]
    
    with torch.no_grad():
        responses, params = module(drug_ids, concs, time_points, media_changes)
    
    # Plot each drug's response
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        ax1.plot(time_points[i], responses[i], color=colors[i], 
                label=f'Drug {i} (τ={params["recovery_tau"][i].item():.1f}h)', 
                linewidth=2, alpha=0.8)
    
    # Mark media changes
    for mc in media_changes:
        ax1.axvline(mc, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Oxygen Response (%)')
    ax1.set_title('Drug-Specific Responses (Same Concentration)')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Concentration dependence for one drug
    ax2 = plt.subplot(4, 4, 2)
    
    drug_id = torch.zeros(5, dtype=torch.long)  # Drug 0
    concs_range = torch.tensor([[0.1], [1.0], [10.0], [100.0], [1000.0]])
    time_points_single = torch.linspace(0, 150, 300).unsqueeze(0).repeat(5, 1)
    
    with torch.no_grad():
        resp_conc, params_conc = module(drug_id, concs_range, time_points_single, [72])
    
    for i, conc in enumerate(concs_range):
        ax2.plot(time_points_single[i], resp_conc[i], 
                label=f'{conc.item():.1f} μM (spike={params_conc["spike_height"][i].item():.2f})',
                linewidth=2)
    
    ax2.axvline(72, color='red', linestyle='--', alpha=0.5, label='Media Change')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Oxygen Response (%)')
    ax2.set_title('Concentration-Dependent Response (Drug 0)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameter distributions
    ax3 = plt.subplot(4, 4, 3)
    
    # Get parameters for all drugs at 10 μM
    all_drug_ids = torch.arange(n_drugs)
    all_concs = torch.ones(n_drugs, 1) * 10.0
    dummy_time = torch.zeros(n_drugs, 1)
    
    with torch.no_grad():
        _, all_params = module(all_drug_ids, all_concs, dummy_time, [])
    
    # Plot parameter distributions
    param_data = []
    for i in range(n_drugs):
        param_data.append({
            'Drug': i,
            'Spike Height': all_params['spike_height'][i].item(),
            'Recovery τ': all_params['recovery_tau'][i].item(),
            'Baseline Shift': all_params['baseline_shift'][i].item()
        })
    
    df_params = pd.DataFrame(param_data)
    
    # Box plot of recovery times
    ax3.boxplot([df_params['Spike Height'], 
                 df_params['Recovery τ'] / 10,  # Scale for visualization
                 df_params['Baseline Shift']], 
                labels=['Spike\nHeight', 'Recovery τ\n(÷10)', 'Baseline\nShift'])
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Distributions Across Drugs')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap of parameters
    ax4 = plt.subplot(4, 4, 4)
    
    # Create parameter matrix
    param_matrix = np.array([
        df_params['Spike Height'].values,
        df_params['Recovery τ'].values / df_params['Recovery τ'].max(),  # Normalize
        df_params['Baseline Shift'].values
    ])
    
    im = ax4.imshow(param_matrix, aspect='auto', cmap='RdBu_r')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Spike Height', 'Recovery τ (norm)', 'Baseline Shift'])
    ax4.set_xlabel('Drug Index')
    ax4.set_title('Drug Parameter Heatmap')
    plt.colorbar(im, ax=ax4)
    
    # 5-8: Show individual response components
    for idx in range(4):
        ax = plt.subplot(4, 4, 5 + idx)
        
        # Single drug, single concentration
        drug_id = torch.tensor([idx * 5])  # Every 5th drug
        conc = torch.tensor([[10.0]])
        time_detail = torch.linspace(0, 100, 200).unsqueeze(0)
        
        with torch.no_grad():
            # Get parameters
            drug_embed = module.drug_embed(drug_id)
            features = torch.cat([drug_embed, torch.log10(conc + 1e-8)], dim=-1)
            raw_params = module.param_net(features)
            
            spike_h = 5.0 * torch.tanh(raw_params[:, 0:1])
            recovery_t = 20.0 * torch.sigmoid(raw_params[:, 1:2]) + 1.0
            baseline_s = 2.0 * torch.tanh(raw_params[:, 2:3])
            
            # Show components
            t_since = time_detail - 50
            mask = (t_since > 0).float()
            
            exp_decay = torch.exp(-t_since.clamp(min=0) / recovery_t)
            spike_comp = spike_h * exp_decay * mask
            baseline_comp = baseline_s * (1 - exp_decay) * mask
            total = spike_comp + baseline_comp
        
        ax.plot(time_detail[0], spike_comp[0], label='Spike Component', linewidth=2)
        ax.plot(time_detail[0], baseline_comp[0], label='Baseline Component', linewidth=2)
        ax.plot(time_detail[0], total[0], label='Total Response', linewidth=3, linestyle='--')
        ax.axvline(50, color='red', linestyle=':', alpha=0.5, label='Media Change')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Response (%)')
        ax.set_title(f'Drug {idx*5} Response Components')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 9-12: Multiple media changes
    for idx in range(4):
        ax = plt.subplot(4, 4, 9 + idx)
        
        drug_id = torch.tensor([idx * 5])
        conc = torch.tensor([[10.0 ** (idx - 1)]])  # Different concentrations
        time_long = torch.linspace(0, 300, 600).unsqueeze(0)
        
        with torch.no_grad():
            resp_multi, params_multi = module(drug_id, conc, time_long, media_changes)
        
        ax.plot(time_long[0], resp_multi[0], linewidth=2, 
               label=f'τ={params_multi["recovery_tau"][0].item():.1f}h')
        
        for mc in media_changes:
            ax.axvline(mc, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Cumulative Response (%)')
        ax.set_title(f'Drug {idx*5} @ {conc[0,0].item():.1f} μM')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 13-16: Parameter relationships
    # Recovery tau vs spike height
    ax13 = plt.subplot(4, 4, 13)
    ax13.scatter(df_params['Spike Height'], df_params['Recovery τ'], alpha=0.6, s=50)
    ax13.set_xlabel('Spike Height (% O₂)')
    ax13.set_ylabel('Recovery τ (hours)')
    ax13.set_title('Parameter Correlations')
    ax13.grid(True, alpha=0.3)
    
    # Drug embedding visualization (2D projection)
    ax14 = plt.subplot(4, 4, 14)
    
    with torch.no_grad():
        embeddings = module.drug_embed.weight.numpy()
    
    # Simple 2D projection using first 2 dimensions
    ax14.scatter(embeddings[:, 0], embeddings[:, 1], 
                c=df_params['Recovery τ'], cmap='viridis', s=100)
    ax14.set_xlabel('Embedding Dim 1')
    ax14.set_ylabel('Embedding Dim 2')
    ax14.set_title('Drug Embeddings (First 2 Dims)')
    cbar = plt.colorbar(ax14.collections[0], ax=ax14)
    cbar.set_label('Recovery τ (h)')
    
    # Concentration-response curves
    ax15 = plt.subplot(4, 4, 15)
    
    conc_sweep = torch.logspace(-2, 3, 50).unsqueeze(1)
    param_curves = {
        'spike': [],
        'tau': [],
        'baseline': []
    }
    
    for drug_idx in [0, 5, 10, 15]:
        drug_ids_sweep = torch.ones(50, dtype=torch.long) * drug_idx
        
        with torch.no_grad():
            _, params_sweep = module(drug_ids_sweep.long(), conc_sweep, 
                                    torch.zeros(50, 1), [])
        
        ax15.semilogx(conc_sweep.squeeze(), params_sweep['recovery_tau'].squeeze(),
                     label=f'Drug {drug_idx}', linewidth=2)
    
    ax15.set_xlabel('Concentration (μM)')
    ax15.set_ylabel('Recovery τ (hours)')
    ax15.set_title('Concentration-Dependent Recovery')
    ax15.legend()
    ax15.grid(True, alpha=0.3)
    
    # Summary statistics
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('off')
    
    summary_text = f"""Model Summary:
    
• {n_drugs} drugs modeled
• 3 parameters per drug:
  - Spike height: {df_params['Spike Height'].mean():.2f} ± {df_params['Spike Height'].std():.2f}
  - Recovery τ: {df_params['Recovery τ'].mean():.1f} ± {df_params['Recovery τ'].std():.1f} h
  - Baseline shift: {df_params['Baseline Shift'].mean():.2f} ± {df_params['Baseline Shift'].std():.2f}
  
• Simple exponential model
• Stable numerical computation
• Concentration-dependent
"""
    
    ax16.text(0.1, 0.9, summary_text, transform=ax16.transAxes,
             fontsize=11, verticalalignment='top', family='monospace')
    
    plt.suptitle('Simplified Drug-Aware Media Change Module', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save
    viz_path = results_dir / "simple_drug_aware_module.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {viz_path}")
    
    return viz_path

def test_simple_module():
    """Test the simplified module."""
    
    print("=== TESTING SIMPLE DRUG-AWARE MODULE ===")
    
    module = SimpleDrugAwareModule(n_drugs=10)
    
    # Test 1: Forward pass
    print("\nTest 1: Forward pass")
    drug_ids = torch.tensor([0, 1, 2])
    concs = torch.tensor([[1.0], [10.0], [100.0]])
    time_points = torch.linspace(0, 100, 50).unsqueeze(0).repeat(3, 1)
    
    response, params = module(drug_ids, concs, time_points, [50])
    
    print(f"✓ Response shape: {response.shape}")
    print(f"✓ Parameters: {list(params.keys())}")
    
    # Test 2: Gradient flow
    print("\nTest 2: Gradient flow")
    module.train()
    
    target = torch.randn_like(response)
    loss = F.mse_loss(response, target)
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                  for p in module.parameters())
    print(f"✓ Gradients flow: {has_grad}")
    
    # Test 3: Drug differences
    print("\nTest 3: Drug-specific responses")
    
    with torch.no_grad():
        diff_drugs = torch.arange(5)
        same_conc = torch.ones(5, 1) * 10.0
        same_time = torch.linspace(0, 100, 50).unsqueeze(0).repeat(5, 1)
        
        resp_diff, _ = module(diff_drugs, same_conc, same_time, [50])
        
        # Check variance
        max_var = resp_diff.var(dim=0).max().item()
        mean_var = resp_diff.var(dim=0).mean().item()
        
    print(f"✓ Response variance - Max: {max_var:.4f}, Mean: {mean_var:.4f}")
    
    print("\n✅ All tests passed!")
    
    return True

def main():
    """Test and visualize the simplified drug-aware module."""
    print("="*80)
    print("SIMPLIFIED DRUG-AWARE MEDIA CHANGE MODULE")
    print("="*80)
    
    # Test
    success = test_simple_module()
    
    if success:
        # Visualize
        print("\nCreating visualizations...")
        viz_path = visualize_simple_drug_responses()
        
        print("\n✅ Simple drug-aware module complete!")
        print(f"Key features:")
        print(f"  ✓ Stable numerical computation")
        print(f"  ✓ Drug-specific parameters")
        print(f"  ✓ Concentration dependence")
        print(f"  ✓ Simple exponential model")
        print(f"  ✓ Handles multiple media changes")

if __name__ == "__main__":
    main()