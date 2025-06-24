#!/usr/bin/env python3
"""
Flexible dataset class that handles varying numbers of replicates with masking.
Demonstrates loading and batching data with 1-4 replicates per drug/concentration.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
from collections import defaultdict

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

class FlexibleReplicateDataset(Dataset):
    """
    Dataset that handles oxygen data with varying numbers of replicates.
    Returns masked tensors that can handle 1-4 replicates per sample.
    """
    
    def __init__(self, data_df, sequence_length=400, max_replicates=4):
        """
        Args:
            data_df: DataFrame with columns [drug, concentration, well_id, hours, oxygen]
            sequence_length: Length of oxygen sequences to extract
            max_replicates: Maximum number of replicates (default 4)
        """
        self.sequence_length = sequence_length
        self.max_replicates = max_replicates
        
        # Group data by drug/concentration to find replicates
        self.samples = []
        self._process_data(data_df)
        
    def _process_data(self, data_df):
        """Process data into samples with replicate information."""
        # Group by drug and concentration
        grouped = data_df.groupby(['drug', 'concentration'])
        
        for (drug, conc), group in grouped:
            # Find unique wells (replicates) for this drug/conc
            wells = group['well_id'].unique()
            n_replicates = len(wells)
            
            if n_replicates == 0:
                continue
                
            # Collect oxygen series for each replicate
            replicate_series = []
            valid_sample = True
            
            for well_id in wells[:self.max_replicates]:  # Limit to max_replicates
                well_data = group[group['well_id'] == well_id].sort_values('hours')
                
                # Check if we have enough data points
                if len(well_data) < self.sequence_length:
                    valid_sample = False
                    break
                    
                # Extract oxygen values
                oxygen = well_data['oxygen'].values[:self.sequence_length]
                hours = well_data['hours'].values[:self.sequence_length]
                
                replicate_series.append({
                    'oxygen': oxygen,
                    'hours': hours,
                    'well_id': well_id
                })
            
            if valid_sample and len(replicate_series) > 0:
                # Detect media changes from hour patterns
                media_changes = self._detect_media_changes(replicate_series[0]['hours'])
                
                self.samples.append({
                    'drug': drug,
                    'concentration': conc,
                    'replicates': replicate_series,
                    'n_replicates': len(replicate_series),
                    'media_changes': media_changes
                })
    
    def _detect_media_changes(self, hours, expected_times=[72, 144, 216]):
        """Detect media change times from hour sequence."""
        media_changes = []
        for expected_time in expected_times:
            if hours[-1] > expected_time:
                # Find closest time point
                closest_idx = np.argmin(np.abs(hours - expected_time))
                media_changes.append(closest_idx)
        return media_changes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - oxygen: [sequence_length, max_replicates] tensor (padded with 0s)
                - replicate_mask: [max_replicates] binary mask (1 for valid replicates)
                - drug: drug name
                - concentration: concentration value
                - n_replicates: actual number of replicates
                - media_schedule: [sequence_length] binary tensor indicating media changes
        """
        sample = self.samples[idx]
        
        # Create padded oxygen tensor
        oxygen_tensor = torch.zeros(self.sequence_length, self.max_replicates)
        
        # Fill in available replicates
        for i, rep_data in enumerate(sample['replicates']):
            oxygen_tensor[:, i] = torch.tensor(rep_data['oxygen'], dtype=torch.float32)
        
        # Create replicate mask
        replicate_mask = torch.zeros(self.max_replicates)
        replicate_mask[:sample['n_replicates']] = 1.0
        
        # Create media schedule
        media_schedule = torch.zeros(self.sequence_length)
        for change_idx in sample['media_changes']:
            media_schedule[change_idx] = 1.0
        
        return {
            'oxygen': oxygen_tensor,
            'replicate_mask': replicate_mask,
            'drug': sample['drug'],
            'concentration': torch.tensor(sample['concentration'], dtype=torch.float32),
            'n_replicates': sample['n_replicates'],
            'media_schedule': media_schedule
        }

def visualize_flexible_batches(dataset, batch_size=8):
    """Create visualizations showing batches with varying replicate counts."""
    
    # Try to get a diverse batch with different replicate counts
    # First, organize samples by replicate count
    samples_by_rep = {1: [], 2: [], 3: [], 4: []}
    for i in range(len(dataset)):
        sample = dataset[i]
        n_reps = sample['n_replicates']
        if n_reps in samples_by_rep:
            samples_by_rep[n_reps].append(i)
    
    # Create a balanced batch
    batch_indices = []
    for n_reps in [1, 2, 3, 4]:
        if samples_by_rep[n_reps]:
            # Take up to 2 samples of each replicate count
            batch_indices.extend(samples_by_rep[n_reps][:2])
    
    # If we don't have enough diversity, just take first batch_size samples
    if len(batch_indices) < batch_size:
        batch_indices = list(range(min(len(dataset), batch_size)))
    else:
        batch_indices = batch_indices[:batch_size]
    
    # Create batch manually
    batch = {
        'oxygen': torch.stack([dataset[i]['oxygen'] for i in batch_indices]),
        'replicate_mask': torch.stack([dataset[i]['replicate_mask'] for i in batch_indices]),
        'drug': [dataset[i]['drug'] for i in batch_indices],
        'concentration': torch.stack([dataset[i]['concentration'] for i in batch_indices]),
        'n_replicates': [dataset[i]['n_replicates'] for i in batch_indices],
        'media_schedule': torch.stack([dataset[i]['media_schedule'] for i in batch_indices])
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Visualize replicate distribution in this batch
    ax1 = plt.subplot(4, 2, 1)
    rep_counts = batch['n_replicates']  # Already a list of ints
    rep_dist = {i: rep_counts.count(i) for i in range(1, 5)}
    
    bars = ax1.bar(rep_dist.keys(), rep_dist.values(), 
                    color=['red', 'orange', 'lightgreen', 'darkgreen'])
    ax1.set_xlabel('Number of Replicates')
    ax1.set_ylabel('Count in Batch')
    ax1.set_title(f'Replicate Distribution in Batch (n={batch_size})')
    ax1.set_xticks([1, 2, 3, 4])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 2. Show example curves for each replicate count
    examples_by_rep = {1: None, 2: None, 3: None, 4: None}
    
    for i in range(len(batch['n_replicates'])):
        n_reps = batch['n_replicates'][i]  # Already an int
        if examples_by_rep[n_reps] is None:
            examples_by_rep[n_reps] = i
    
    # Plot examples
    for rep_count, idx in examples_by_rep.items():
        if idx is not None:
            ax = plt.subplot(4, 2, 2 + rep_count)
            
            oxygen_data = batch['oxygen'][idx]
            mask = batch['replicate_mask'][idx]
            drug = batch['drug'][idx]
            conc = batch['concentration'][idx].item()
            
            # Plot each replicate
            colors = ['blue', 'green', 'red', 'purple']
            for rep_idx in range(rep_count):
                if mask[rep_idx] > 0:
                    ax.plot(oxygen_data[:, rep_idx], 
                           color=colors[rep_idx], 
                           alpha=0.8,
                           label=f'Rep {rep_idx+1}')
            
            # Mark media changes
            media_changes = torch.where(batch['media_schedule'][idx] > 0)[0]
            for mc in media_changes:
                ax.axvline(mc, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Oxygen (%)')
            ax.set_title(f'{drug} @ {conc:.2f} μM\n({rep_count} replicate{"s" if rep_count > 1 else ""})')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # 3. Visualize replicate masks as heatmap
    ax3 = plt.subplot(4, 1, 4)
    
    # Stack all masks
    actual_batch_size = len(batch['n_replicates'])
    all_masks = torch.stack([batch['replicate_mask'][i] for i in range(actual_batch_size)])
    
    im = ax3.imshow(all_masks.numpy(), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xlabel('Replicate Index')
    ax3.set_ylabel('Sample in Batch')
    ax3.set_title('Replicate Mask Visualization (Green=Valid, Red=Missing)')
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(['Rep 1', 'Rep 2', 'Rep 3', 'Rep 4'])
    
    # Add drug names on y-axis
    drug_labels = [f"{batch['drug'][i][:15]}..." if len(batch['drug'][i]) > 15 else batch['drug'][i] 
                   for i in range(actual_batch_size)]
    ax3.set_yticks(range(actual_batch_size))
    ax3.set_yticklabels(drug_labels, fontsize=8)
    
    # Add text annotations
    for i in range(actual_batch_size):
        for j in range(4):
            val = all_masks[i, j].item()
            text = ax3.text(j, i, '✓' if val > 0 else '✗',
                           ha="center", va="center", 
                           color="white" if val > 0 else "black",
                           fontsize=10, weight='bold')
    
    plt.colorbar(im, ax=ax3, ticks=[0, 1], label='Valid Replicate')
    
    plt.suptitle('Flexible Replicate Dataset Batch Visualization', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = results_dir / "flexible_dataset_batch_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved batch visualization to: {output_path}")
    
    return output_path

def analyze_dataset_statistics(dataset):
    """Analyze and visualize dataset statistics."""
    
    # Collect statistics
    stats = {
        'replicate_counts': defaultdict(int),
        'drugs_by_replicate': defaultdict(set),
        'conc_by_replicate': defaultdict(list),
        'media_change_counts': []
    }
    
    for i in range(len(dataset)):
        sample = dataset[i]
        n_reps = sample['n_replicates']
        drug = sample['drug']
        conc = sample['concentration'].item()
        n_media_changes = sample['media_schedule'].sum().item()
        
        stats['replicate_counts'][n_reps] += 1
        stats['drugs_by_replicate'][n_reps].add(drug)
        stats['conc_by_replicate'][n_reps].append(conc)
        stats['media_change_counts'].append(n_media_changes)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall replicate distribution
    rep_counts = dict(stats['replicate_counts'])
    bars = ax1.bar(rep_counts.keys(), rep_counts.values(), 
                    color=['red', 'orange', 'lightgreen', 'darkgreen'])
    ax1.set_xlabel('Number of Replicates')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Distribution of Replicate Counts in Dataset')
    ax1.set_xticks([1, 2, 3, 4])
    
    total_samples = sum(rep_counts.values())
    for bar in bars:
        height = bar.get_height()
        pct = height / total_samples * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # 2. Number of unique drugs per replicate count
    drug_counts = {k: len(v) for k, v in stats['drugs_by_replicate'].items()}
    ax2.bar(drug_counts.keys(), drug_counts.values(), 
            color=['red', 'orange', 'lightgreen', 'darkgreen'])
    ax2.set_xlabel('Number of Replicates')
    ax2.set_ylabel('Number of Unique Drugs')
    ax2.set_title('Drug Coverage by Replicate Count')
    ax2.set_xticks([1, 2, 3, 4])
    
    # 3. Concentration distribution by replicate count
    for n_reps, concs in stats['conc_by_replicate'].items():
        if concs:
            ax3.hist(np.log10(np.array(concs) + 1e-6), bins=20, alpha=0.7, 
                    label=f'{n_reps} replicates', density=True)
    
    ax3.set_xlabel('Log10(Concentration)')
    ax3.set_ylabel('Density')
    ax3.set_title('Concentration Distribution by Replicate Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Media change distribution
    ax4.hist(stats['media_change_counts'], bins=np.arange(0, 5) - 0.5, 
             color='purple', edgecolor='black')
    ax4.set_xlabel('Number of Media Changes')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Distribution of Media Changes per Sample')
    ax4.set_xticks([0, 1, 2, 3])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Flexible Replicate Dataset Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    stats_path = results_dir / "flexible_dataset_statistics.png"
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dataset statistics to: {stats_path}")
    
    # Save text summary
    summary_path = results_dir / "flexible_dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("FLEXIBLE REPLICATE DATASET SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Samples by replicate count:\n")
        for n_reps in sorted(rep_counts.keys()):
            count = rep_counts[n_reps]
            pct = count / total_samples * 100
            f.write(f"  {n_reps} replicate(s): {count} ({pct:.1f}%)\n")
        f.write(f"\nUnique drugs by replicate count:\n")
        for n_reps in sorted(drug_counts.keys()):
            f.write(f"  {n_reps} replicate(s): {drug_counts[n_reps]} drugs\n")
        f.write(f"\nMedia changes per sample:\n")
        f.write(f"  Mean: {np.mean(stats['media_change_counts']):.2f}\n")
        f.write(f"  Median: {np.median(stats['media_change_counts']):.0f}\n")
        f.write(f"  Max: {max(stats['media_change_counts'])}\n")
    
    print(f"Saved dataset summary to: {summary_path}")
    
    return stats_path

def main():
    """Load data and demonstrate flexible replicate handling."""
    print("="*80)
    print("FLEXIBLE REPLICATE DATASET DEMONSTRATION")
    print("="*80)
    
    # Connect to database
    print("\nConnecting to database...")
    DATABASE_URL = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    conn.execute(f"ATTACH '{DATABASE_URL}' AS supabase (TYPE POSTGRES, READ_ONLY);")
    
    # Query a subset of data for demonstration
    # Specifically target drugs with varying replicate counts
    query = """
    WITH replicate_counts AS (
        SELECT 
            drug,
            concentration,
            COUNT(DISTINCT well_number) as n_replicates
        FROM supabase.public.well_map_data
        WHERE drug != '' 
            AND drug IS NOT NULL
            AND is_excluded = false
            AND concentration > 0
        GROUP BY drug, concentration
    ),
    diverse_samples AS (
        -- Get samples with different replicate counts
        (SELECT drug, concentration FROM replicate_counts WHERE n_replicates = 1 LIMIT 5)
        UNION ALL
        (SELECT drug, concentration FROM replicate_counts WHERE n_replicates = 2 LIMIT 5)
        UNION ALL
        (SELECT drug, concentration FROM replicate_counts WHERE n_replicates = 3 LIMIT 5)
        UNION ALL
        (SELECT drug, concentration FROM replicate_counts WHERE n_replicates = 4 LIMIT 10)
    ),
    filtered_data AS (
        SELECT 
            w.plate_id || '_' || w.well_number as well_id,
            w.drug,
            w.concentration,
            p.timestamp,
            p.median_o2 as oxygen,
            EXTRACT(EPOCH FROM (p.timestamp - MIN(p.timestamp) OVER (PARTITION BY w.plate_id))) / 3600.0 as hours
        FROM supabase.public.well_map_data w
        JOIN supabase.public.processed_data p
            ON w.plate_id = p.plate_id AND w.well_number = p.well_number
        WHERE (w.drug, w.concentration) IN (SELECT drug, concentration FROM diverse_samples)
            AND w.drug != '' 
            AND w.drug IS NOT NULL
            AND w.is_excluded = false
            AND p.is_excluded = false
            AND w.concentration > 0
        ORDER BY w.drug, w.concentration, w.well_number, p.timestamp
    )
    SELECT * FROM filtered_data
    """
    
    print("Fetching data...")
    df = conn.execute(query).df()
    conn.close()
    
    print(f"Loaded {len(df):,} measurements")
    print(f"Covering {df['drug'].nunique()} drugs")
    
    # Create dataset
    print("\nCreating flexible replicate dataset...")
    dataset = FlexibleReplicateDataset(df, sequence_length=400)
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Visualize batch
    print("\nVisualizing example batch...")
    batch_viz_path = visualize_flexible_batches(dataset, batch_size=8)
    
    # Analyze dataset statistics
    print("\nAnalyzing dataset statistics...")
    stats_path = analyze_dataset_statistics(dataset)
    
    print("\n✅ Flexible dataset demonstration complete!")
    print(f"Visualizations saved to:")
    print(f"  - {batch_viz_path}")
    print(f"  - {stats_path}")

if __name__ == "__main__":
    main()