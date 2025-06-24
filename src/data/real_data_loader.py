#!/usr/bin/env python3
"""
Load real oxygen time series data from database for training.
Groups data by drug/concentration and handles replicate structure.
"""

import psycopg2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

class RealOxygenDataLoader:
    """Load and preprocess real oxygen data from database."""
    
    def __init__(self, database_url=None):
        """Initialize with database connection."""
        if database_url is None:
            database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.database_url = database_url
        self.connection = None
        
    def connect(self):
        """Connect to database."""
        try:
            self.connection = psycopg2.connect(self.database_url)
            print("✓ Database connection established")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")
    
    def explore_schema(self):
        """Explore database schema to understand table structure."""
        if not self.connection:
            self.connect()
        
        # Get all table names
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        tables_df = pd.read_sql_query(query, self.connection)
        print("Available tables:")
        for table in tables_df['table_name']:
            print(f"  - {table}")
        
        # Check key tables for oxygen data
        key_tables = ['processed_data', 'well_metabolic_data', 'raw_data', 'well_map_data', 'event_table']
        
        for table in key_tables:
            if table in tables_df['table_name'].values:
                try:
                    col_query = f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                    """
                    cols_df = pd.read_sql_query(col_query, self.connection)
                    print(f"\nColumns in {table}:")
                    for _, row in cols_df.iterrows():
                        print(f"  - {row['column_name']}: {row['data_type']}")
                        
                    # Sample a few rows to understand data structure
                    sample_query = f"SELECT * FROM {table} LIMIT 3"
                    sample_df = pd.read_sql_query(sample_query, self.connection)
                    print(f"\nSample data from {table}:")
                    print(sample_df.head())
                    
                except Exception as e:
                    print(f"  Error reading {table}: {e}")
        
        return tables_df['table_name'].tolist()

    def load_oxygen_data(self, min_timepoints=100, max_drugs=50, exclude_flagged=True):
        """
        Load oxygen time series data grouped by drug/concentration.
        
        Args:
            min_timepoints: Minimum number of time points required
            max_drugs: Maximum number of drugs to include
            exclude_flagged: Whether to exclude flagged data points
        
        Returns:
            Dict with loaded data grouped by (drug_name, concentration)
        """
        if not self.connection:
            self.connect()
        
        # Skip schema exploration for speed
        print("Loading data directly...")
        
        # Simplified query for testing - get a small sample first
        query = """
        SELECT 
            wm.drug,
            wm.concentration,
            wm.well_number,
            wm.plate_id
        FROM well_map_data wm
        WHERE wm.concentration > 0 
          AND wm.drug IS NOT NULL 
          AND wm.drug != ''
        LIMIT 100
        """
        
        try:
            print(f"\nLoading oxygen data with correct schema...")
            df = pd.read_sql_query(query, self.connection)
            print(f"✓ Query successful, got {len(df)} rows")
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return {}
        
        if exclude_flagged:
            initial_len = len(df)
            df = df[~df['is_excluded']]
            print(f"Excluded {initial_len - len(df)} flagged data points")
        
        print(f"Loaded {len(df)} data points")
        print(f"Unique drugs: {df['drug'].nunique()}")
        print(f"Unique plates: {df['plate_id'].nunique()}")
        print(f"Concentration range: {df['concentration'].min():.3f} - {df['concentration'].max():.1f}")
        print(f"Time range: {df['time_hours'].min():.1f} - {df['time_hours'].max():.1f} hours")
        
        # Group by drug and concentration 
        # Use well_number as replicate identifier
        grouped_data = {}
        
        for (drug_name, concentration), group in df.groupby(['drug', 'concentration']):
            # Get all wells (replicates) for this drug/concentration
            replicates = {}
            
            for well_number, well_group in group.groupby('well_number'):
                # Sort by time and extract time series
                well_group = well_group.sort_values('time_hours')
                
                if len(well_group) >= min_timepoints:
                    replicates[well_number] = {
                        'time_hours': well_group['time_hours'].values,
                        'oxygen_percent': well_group['oxygen_percent'].values,
                        'plate_id': well_group['plate_id'].iloc[0],
                        'well_number': well_number
                    }
            
            # Only include if we have at least one valid replicate
            if replicates:
                grouped_data[(drug_name, concentration)] = {
                    'replicates': replicates,
                    'n_replicates': len(replicates),
                    'plate_ids': group['plate_id'].unique().tolist()
                }
        
        print(f"Grouped into {len(grouped_data)} drug/concentration combinations")
        
        # Limit to max_drugs if specified
        if max_drugs and len(grouped_data) > max_drugs:
            # Sort by number of replicates and take top ones
            sorted_combinations = sorted(
                grouped_data.items(), 
                key=lambda x: x[1]['n_replicates'], 
                reverse=True
            )
            grouped_data = dict(sorted_combinations[:max_drugs])
            print(f"Limited to top {max_drugs} combinations by replicate count")
        
        return grouped_data
    
    def load_media_change_events(self, plate_ids=None):
        """Load media change events for plates."""
        if not self.connection:
            self.connect()
        
        # Look for media change events in event_table
        # Events might have titles like "Media Change" or descriptions containing "media"
        query = """
        SELECT 
            plate_id,
            occurred_at,
            title,
            description,
            EXTRACT(EPOCH FROM (occurred_at - 
                (SELECT MIN(occurred_at) FROM event_table et2 
                 WHERE et2.plate_id = event_table.plate_id))) / 3600.0 as event_time_hours
        FROM event_table
        WHERE (LOWER(title) LIKE '%media%' OR LOWER(description) LIKE '%media%')
          AND is_excluded = false
        """
        
        try:
            df = pd.read_sql_query(query, self.connection)
            print(f"Found {len(df)} potential media change events")
            
            if len(df) > 0:
                print("Sample events:")
                print(df[['title', 'description', 'event_time_hours']].head())
            
            # Group by plate_id
            events_by_plate = {}
            for plate_id, group in df.groupby('plate_id'):
                events_by_plate[str(plate_id)] = sorted(group['event_time_hours'].tolist())
            
            return events_by_plate
            
        except Exception as e:
            print(f"Error loading events: {e}")
            return {}

def interpolate_time_series(time_hours, oxygen_percent, target_length=200, 
                          max_time=200, method='linear'):
    """
    Interpolate time series to fixed length and time range.
    
    Args:
        time_hours: Array of time points
        oxygen_percent: Array of oxygen values
        target_length: Target number of time points
        max_time: Maximum time in hours
        method: Interpolation method
    
    Returns:
        Tuple of (interpolated_time, interpolated_oxygen)
    """
    # Create target time grid
    target_time = np.linspace(0, max_time, target_length)
    
    # Only interpolate within the range of available data
    min_time = time_hours.min()
    max_available_time = time_hours.max()
    
    # Interpolate
    interpolated_oxygen = np.interp(target_time, time_hours, oxygen_percent)
    
    # Mask extrapolated regions (set to NaN)
    mask = (target_time < min_time) | (target_time > max_available_time)
    interpolated_oxygen[mask] = np.nan
    
    return target_time, interpolated_oxygen

class RealReplicateDataset(Dataset):
    """Dataset for real oxygen replicate data."""
    
    def __init__(self, grouped_data, media_events=None, target_length=200, 
                 max_time=200, min_replicates=1, max_replicates=4):
        """
        Initialize with grouped data.
        
        Args:
            grouped_data: Dict from RealOxygenDataLoader
            media_events: Dict of media change events by experiment_id
            target_length: Number of time points to interpolate to
            max_time: Maximum time in hours
            min_replicates: Minimum replicates required
            max_replicates: Maximum replicates to use
        """
        self.grouped_data = grouped_data
        self.media_events = media_events or {}
        self.target_length = target_length
        self.max_time = max_time
        self.min_replicates = min_replicates
        self.max_replicates = max_replicates
        
        # Filter and prepare samples
        self.samples = []
        self.drug_to_idx = {}
        drug_idx = 0
        
        for (drug_name, concentration), data in grouped_data.items():
            n_reps = data['n_replicates']
            
            # Only include if we have enough replicates
            if n_reps >= min_replicates:
                # Assign drug index
                if drug_name not in self.drug_to_idx:
                    self.drug_to_idx[drug_name] = drug_idx
                    drug_idx += 1
                
                self.samples.append({
                    'drug_name': drug_name,
                    'drug_idx': self.drug_to_idx[drug_name],
                    'concentration': concentration,
                    'data': data
                })
        
        print(f"Dataset created with {len(self.samples)} samples")
        print(f"Drugs: {len(self.drug_to_idx)}")
        
        # Create target time grid
        self.time_grid = np.linspace(0, max_time, target_length)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample with variable replicates."""
        sample = self.samples[idx]
        data = sample['data']
        replicates = data['replicates']
        
        # Determine how many replicates to use
        available_reps = len(replicates)
        n_use = min(available_reps, self.max_replicates)
        
        # Initialize tensors
        oxygen_tensor = torch.zeros(self.target_length, self.max_replicates)
        replicate_mask = torch.zeros(self.max_replicates)
        
        # Process each replicate
        rep_keys = list(replicates.keys())[:n_use]
        
        for i, rep_key in enumerate(rep_keys):
            rep_data = replicates[rep_key]
            
            # Interpolate to target grid
            _, interpolated_oxygen = interpolate_time_series(
                rep_data['time_hours'],
                rep_data['oxygen_percent'],
                self.target_length,
                self.max_time
            )
            
            # Handle NaN values (use forward/backward fill)
            mask = ~np.isnan(interpolated_oxygen)
            if mask.any():
                # Forward fill
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    first_valid = valid_indices[0]
                    last_valid = valid_indices[-1]
                    
                    # Backward fill before first valid
                    interpolated_oxygen[:first_valid] = interpolated_oxygen[first_valid]
                    # Forward fill after last valid
                    interpolated_oxygen[last_valid:] = interpolated_oxygen[last_valid]
                
                oxygen_tensor[:, i] = torch.from_numpy(interpolated_oxygen).float()
                replicate_mask[i] = 1.0
        
        # Get media change times for this sample
        plate_ids = [str(rep_data['plate_id']) for rep_data in replicates.values()]
        media_times = []
        for plate_id in plate_ids:
            if plate_id in self.media_events:
                media_times.extend(self.media_events[plate_id])
        
        # Remove duplicates and sort
        media_times = sorted(list(set(media_times)))
        media_times = [t for t in media_times if 0 <= t <= self.max_time]
        
        return {
            'oxygen': oxygen_tensor,  # [time, max_replicates]
            'replicate_mask': replicate_mask,  # [max_replicates]
            'drug_idx': torch.tensor(sample['drug_idx'], dtype=torch.long),
            'concentration': torch.tensor([sample['concentration']], dtype=torch.float),
            'media_change_times': media_times,
            'time_grid': torch.from_numpy(self.time_grid).float(),
            'n_replicates': torch.tensor(n_use, dtype=torch.long),
            'drug_name': sample['drug_name']
        }

def visualize_real_data_loading():
    """Visualize the loaded real data."""
    
    print("=== LOADING REAL OXYGEN DATA ===")
    
    # Load data
    loader = RealOxygenDataLoader()
    try:
        loader.connect()
        
        # Load oxygen data
        grouped_data = loader.load_oxygen_data(
            min_timepoints=50, 
            max_drugs=20,
            exclude_flagged=True
        )
        
        # Get plate IDs for media events
        all_plate_ids = []
        for data in grouped_data.values():
            all_plate_ids.extend(data['plate_ids'])
        
        # Load media events
        media_events = loader.load_media_change_events(all_plate_ids)
        
        loader.disconnect()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # Create dataset
    dataset = RealReplicateDataset(
        grouped_data, 
        media_events,
        target_length=200,
        max_time=200
    )
    
    # Create visualizations
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Dataset overview
    ax1 = plt.subplot(4, 4, 1)
    
    # Count replicates per sample
    replicate_counts = []
    concentrations = []
    drug_names = []
    
    for i in range(min(len(dataset), 50)):  # Sample first 50
        sample = dataset[i]
        replicate_counts.append(sample['n_replicates'].item())
        concentrations.append(sample['concentration'].item())
        drug_names.append(sample['drug_name'])
    
    unique, counts = np.unique(replicate_counts, return_counts=True)
    ax1.bar(unique, counts, alpha=0.7)
    ax1.set_xlabel('Number of Replicates')
    ax1.set_ylabel('Count')
    ax1.set_title('Replicate Distribution in Real Data')
    ax1.grid(True, alpha=0.3)
    
    # 2. Concentration distribution
    ax2 = plt.subplot(4, 4, 2)
    ax2.hist(concentrations, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Concentration (μM)')
    ax2.set_ylabel('Count')
    ax2.set_title('Concentration Distribution')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3-6. Example time series with different replicate counts
    example_indices = []
    for target_reps in [1, 2, 3, 4]:
        # Find sample with target number of replicates
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['n_replicates'].item() == target_reps:
                example_indices.append(i)
                break
        else:
            example_indices.append(0)  # Fallback
    
    for plot_idx, sample_idx in enumerate(example_indices):
        ax = plt.subplot(4, 4, 3 + plot_idx)
        
        sample = dataset[sample_idx]
        time_grid = sample['time_grid']
        oxygen = sample['oxygen']
        mask = sample['replicate_mask']
        n_reps = sample['n_replicates'].item()
        
        # Plot each replicate
        colors = plt.cm.Set1(np.linspace(0, 1, 4))
        for rep in range(4):
            if mask[rep] > 0:
                ax.plot(time_grid, oxygen[:, rep], color=colors[rep], 
                       alpha=0.8, linewidth=2, label=f'Rep {rep+1}')
            else:
                # Show masked (should be zero)
                ax.plot(time_grid, oxygen[:, rep], color='gray', 
                       alpha=0.3, linestyle='--', label=f'Rep {rep+1} (masked)')
        
        # Mark media changes
        for mc_time in sample['media_change_times']:
            ax.axvline(mc_time, color='red', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Oxygen (%)')
        ax.set_title(f'{sample["drug_name"]}, {sample["concentration"].item():.1f} μM, {n_reps} reps')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)
    
    # 7. Oxygen range analysis
    ax7 = plt.subplot(4, 4, 7)
    
    all_oxygen_values = []
    for i in range(min(len(dataset), 100)):
        sample = dataset[i]
        oxygen = sample['oxygen']
        mask = sample['replicate_mask']
        
        # Only include active replicates
        for rep in range(4):
            if mask[rep] > 0:
                values = oxygen[:, rep].numpy()
                all_oxygen_values.extend(values[values > 0])  # Exclude zeros
    
    ax7.hist(all_oxygen_values, bins=50, alpha=0.7, density=True)
    ax7.axvline(np.mean(all_oxygen_values), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_oxygen_values):.1f}%')
    ax7.set_xlabel('Oxygen (%)')
    ax7.set_ylabel('Density')
    ax7.set_title('Real Data Oxygen Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Media change timing
    ax8 = plt.subplot(4, 4, 8)
    
    all_media_times = []
    for i in range(min(len(dataset), 100)):
        sample = dataset[i]
        all_media_times.extend(sample['media_change_times'])
    
    if all_media_times:
        ax8.hist(all_media_times, bins=20, alpha=0.7)
        ax8.set_xlabel('Media Change Time (hours)')
        ax8.set_ylabel('Count')
        ax8.set_title('Media Change Timing Distribution')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No media change data', ha='center', va='center')
        ax8.set_title('Media Change Events')
    
    # 9-12. Drug-specific examples
    unique_drugs = list(set(drug_names))[:4]
    
    for plot_idx, drug_name in enumerate(unique_drugs):
        ax = plt.subplot(4, 4, 9 + plot_idx)
        
        # Find samples for this drug
        drug_samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['drug_name'] == drug_name:
                drug_samples.append(i)
                if len(drug_samples) >= 3:  # Max 3 samples per drug
                    break
        
        # Plot samples
        colors = plt.cm.tab10(np.linspace(0, 1, len(drug_samples)))
        for i, sample_idx in enumerate(drug_samples):
            sample = dataset[sample_idx]
            time_grid = sample['time_grid']
            oxygen = sample['oxygen']
            mask = sample['replicate_mask']
            
            # Plot mean across replicates
            active_oxygen = oxygen[:, mask > 0]
            if active_oxygen.shape[1] > 0:
                mean_oxygen = active_oxygen.mean(dim=1)
                conc = sample['concentration'].item()
                
                ax.plot(time_grid, mean_oxygen, color=colors[i], 
                       linewidth=2, label=f'{conc:.1f} μM')
                
                # Mark media changes
                for mc_time in sample['media_change_times']:
                    ax.axvline(mc_time, color=colors[i], linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Mean Oxygen (%)')
        ax.set_title(f'{drug_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)
    
    # 13-16. Statistical analysis
    ax13 = plt.subplot(4, 4, 13)
    
    # Replicate correlations in real data
    real_correlations = []
    for i in range(min(len(dataset), 50)):
        sample = dataset[i]
        oxygen = sample['oxygen']
        mask = sample['replicate_mask']
        n_reps = int(mask.sum().item())
        
        if n_reps >= 2:
            for rep1 in range(n_reps):
                for rep2 in range(rep1 + 1, n_reps):
                    corr = np.corrcoef(oxygen[:, rep1].numpy(), 
                                     oxygen[:, rep2].numpy())[0, 1]
                    if not np.isnan(corr):
                        real_correlations.append(corr)
    
    if real_correlations:
        ax13.hist(real_correlations, bins=20, alpha=0.7, density=True)
        ax13.axvline(np.mean(real_correlations), color='red', linestyle='--',
                    label=f'Mean: {np.mean(real_correlations):.3f}')
        ax13.set_xlabel('Replicate Correlation')
        ax13.set_ylabel('Density')
        ax13.set_title('Real Data Replicate Correlations')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
    
    # 14. Time series length analysis
    ax14 = plt.subplot(4, 4, 14)
    
    # Check data coverage across time
    time_coverage = np.zeros(200)
    for i in range(min(len(dataset), 100)):
        sample = dataset[i]
        oxygen = sample['oxygen']
        mask = sample['replicate_mask']
        
        for rep in range(4):
            if mask[rep] > 0:
                # Count non-zero timepoints
                valid_mask = oxygen[:, rep] > 0
                time_coverage += valid_mask.numpy()
    
    time_grid = np.linspace(0, 200, 200)
    ax14.plot(time_grid, time_coverage, linewidth=2)
    ax14.set_xlabel('Time (hours)')
    ax14.set_ylabel('Number of Valid Series')
    ax14.set_title('Data Coverage Over Time')
    ax14.grid(True, alpha=0.3)
    
    # 15-16. Summary statistics
    for i in [15, 16]:
        ax = plt.subplot(4, 4, i)
        ax.axis('off')
    
    # Add summary text
    ax15 = plt.subplot(4, 4, 15)
    ax15.axis('off')
    
    summary_text = f"""Real Data Summary:
    
• Total samples: {len(dataset)}
• Unique drugs: {len(dataset.drug_to_idx)}
• Time points: {dataset.target_length}
• Max time: {dataset.max_time}h

• Oxygen range: {np.min(all_oxygen_values):.1f} - {np.max(all_oxygen_values):.1f}%
• Mean oxygen: {np.mean(all_oxygen_values):.1f} ± {np.std(all_oxygen_values):.1f}%

• Mean replicate correlation: {np.mean(real_correlations):.3f}
• Media changes per sample: {len(all_media_times)/len(dataset):.1f}
"""
    
    ax15.text(0.1, 0.9, summary_text, transform=ax15.transAxes,
             fontsize=11, verticalalignment='top', family='monospace')
    
    plt.suptitle('Real Oxygen Data Loading and Analysis', 
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save
    viz_path = results_dir / "real_data_loading.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization: {viz_path}")
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Unique drugs: {len(dataset.drug_to_idx)}")
    print(f"  Mean oxygen: {np.mean(all_oxygen_values):.1f} ± {np.std(all_oxygen_values):.1f}%")
    if real_correlations:
        print(f"  Mean replicate correlation: {np.mean(real_correlations):.3f}")
    
    return dataset, grouped_data

def main():
    """Load and analyze real data."""
    print("=" * 80)
    print("REAL OXYGEN DATA LOADING")
    print("=" * 80)
    
    try:
        dataset, grouped_data = visualize_real_data_loading()
        
        if dataset is not None:
            print("\n✅ Real data loading complete!")
            print("Ready for training pipeline implementation")
        else:
            print("\n❌ Failed to load real data")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check database connection and environment variables")

if __name__ == "__main__":
    main()