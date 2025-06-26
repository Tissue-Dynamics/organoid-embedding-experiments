#!/usr/bin/env python3
"""
Create oxygen vs time plots for drugs with all concentrations.
Shows events as dotted vertical lines and respects data exclusions.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output directory
OUTPUT_DIR = Path("results/figures/drug_timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_drug_oxygen_timeseries(drug_name: str, plate_id: str = None, max_wells_per_conc: int = 5):
    """
    Plot oxygen vs time for all concentrations of a drug.
    
    Args:
        drug_name: Name of drug to plot
        plate_id: Optional specific plate ID, if None uses all plates
        max_wells_per_conc: Maximum wells to plot per concentration (to avoid clutter)
    """
    
    print(f"Creating oxygen timeseries plot for {drug_name}")
    print(f"Plate filter: {plate_id or 'All plates'}")
    
    with DataLoader() as loader:
        # 1. Load oxygen data
        print("Loading oxygen data...")
        if plate_id:
            oxygen_data = loader.load_oxygen_data(plate_ids=[plate_id])
        else:
            # Load limited plates to avoid timeout
            oxygen_data = loader.load_oxygen_data(limit=1)
        
        # Filter for specific drug
        drug_data = oxygen_data[oxygen_data['drug'] == drug_name].copy()
        
        if drug_data.empty:
            print(f"No data found for drug: {drug_name}")
            return None
        
        print(f"Found {len(drug_data)} oxygen measurements for {drug_name}")
        
        # 2. Load events for the plates
        print("Loading events...")
        plate_ids = drug_data['plate_id'].unique()
        
        # Get events for these plates
        all_events = []
        for pid in plate_ids:
            try:
                plate_events = loader.load_plate_events(pid)
                if not plate_events.empty:
                    # Add plate_id column if not present
                    if 'plate_id' not in plate_events.columns:
                        plate_events['plate_id'] = pid
                    all_events.append(plate_events)
            except Exception as e:
                print(f"    Warning: Could not load events for plate {pid}: {e}")
                continue
        
        if all_events:
            events_data = pd.concat(all_events, ignore_index=True)
            events_data['occurred_at'] = pd.to_datetime(events_data['occurred_at'])
        else:
            events_data = pd.DataFrame()
        
        # 3. Process data for plotting
        drug_data['timestamp'] = pd.to_datetime(drug_data['timestamp'])
        
        # Get unique concentrations
        concentrations = sorted(drug_data['concentration'].unique())
        print(f"Concentrations found: {concentrations}")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Color map for concentrations
        colors = plt.cm.viridis(np.linspace(0, 1, len(concentrations)))
        
        # 4. Plot each concentration
        for i, conc in enumerate(concentrations):
            conc_data = drug_data[drug_data['concentration'] == conc]
            
            # Group by well and limit number of wells per concentration
            wells = conc_data['well_id'].unique()
            if len(wells) > max_wells_per_conc:
                wells = wells[:max_wells_per_conc]
                print(f"  Limiting to {max_wells_per_conc} wells for concentration {conc}")
            
            # Plot each well for this concentration
            for j, well_id in enumerate(wells):
                well_data = conc_data[conc_data['well_id'] == well_id].sort_values('timestamp')
                
                if len(well_data) > 0:
                    # Use alpha to distinguish between wells of same concentration
                    alpha = 0.7 if j == 0 else 0.4
                    linewidth = 2 if j == 0 else 1
                    
                    # Label only the first well of each concentration
                    label = f"{conc} μM" if j == 0 else None
                    
                    ax.plot(well_data['timestamp'], well_data['o2'], 
                           color=colors[i], alpha=alpha, linewidth=linewidth,
                           label=label)
        
        # 5. Add event lines
        if not events_data.empty:
            # Filter events to plates we're showing
            plot_events = events_data[events_data['plate_id'].isin(plate_ids)]
            
            # Get time range of the plot
            time_min = drug_data['timestamp'].min()
            time_max = drug_data['timestamp'].max()
            
            # Filter events to time range
            plot_events = plot_events[
                (plot_events['occurred_at'] >= time_min) & 
                (plot_events['occurred_at'] <= time_max)
            ]
            
            # Add vertical lines for events
            event_types = plot_events['event_type'].unique()
            event_colors = {'Medium Change': 'red', 'Drugs Start': 'green', 
                          'Communication Failure': 'orange', 'Data Upload': 'blue',
                          'Experiment End': 'purple', 'Data Exclusion': 'black'}
            
            for event_type in event_types:
                event_times = plot_events[plot_events['event_type'] == event_type]['occurred_at']
                color = event_colors.get(event_type, 'gray')
                
                for event_time in event_times:
                    ax.axvline(x=event_time, color=color, linestyle='--', alpha=0.7, linewidth=1)
            
            # Create event legend
            event_legend_elements = []
            for event_type in event_types:
                color = event_colors.get(event_type, 'gray')
                event_legend_elements.append(
                    plt.Line2D([0], [0], color=color, linestyle='--', alpha=0.7, label=event_type)
                )
        
        # 6. Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Oxygen Consumption (%)', fontsize=12)
        ax.set_title(f'Oxygen vs Time: {drug_name}\n'
                    f'All Concentrations (Plates: {len(plate_ids)})', fontsize=14, fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add concentration legend
        conc_legend = ax.legend(title='Concentration', loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add event legend if events exist
        if not events_data.empty and len(event_legend_elements) > 0:
            event_legend = ax.legend(handles=event_legend_elements, title='Events', 
                                   loc='upper left', bbox_to_anchor=(1.02, 0.7))
            ax.add_artist(conc_legend)  # Add back concentration legend
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # 7. Save plot
        safe_drug_name = "".join(c for c in drug_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plate_suffix = f"_plate_{plate_id}" if plate_id else "_all_plates"
        filename = f"{safe_drug_name.replace(' ', '_')}_oxygen_timeseries{plate_suffix}.png"
        filepath = OUTPUT_DIR / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        
        # Show plot
        plt.show()
        
        # 8. Print summary statistics
        print(f"\nSummary for {drug_name}:")
        print(f"  Total measurements: {len(drug_data)}")
        print(f"  Plates: {len(plate_ids)}")
        print(f"  Wells: {drug_data['well_id'].nunique()}")
        print(f"  Concentrations: {len(concentrations)}")
        print(f"  Time range: {drug_data['timestamp'].min()} to {drug_data['timestamp'].max()}")
        
        if not events_data.empty:
            print(f"  Events: {len(plot_events)} events of {len(event_types)} types")
        
        return filepath

def list_available_drugs(loader):
    """List drugs available in the dataset."""
    print("Loading available drugs...")
    
    # Get sample of oxygen data to see what drugs are available
    oxygen_sample = loader.load_oxygen_data(limit=3)
    drug_counts = oxygen_sample['drug'].value_counts()
    
    print(f"\nTop drugs by measurement count (from sample):")
    for drug, count in drug_counts.head(10).items():
        if drug != 'Unknown':
            print(f"  {drug}: {count} measurements")
    
    return drug_counts.index.tolist()

def main():
    """Main function to create drug timeseries plots."""
    
    print("=" * 70)
    print("Drug Oxygen Timeseries Visualization")
    print("=" * 70)
    
    with DataLoader() as loader:
        # List available drugs
        available_drugs = list_available_drugs(loader)
        
        # Example plots for top drugs
        top_drugs = [drug for drug in available_drugs[:3] if drug != 'Unknown']
        
        print(f"\nCreating plots for top {len(top_drugs)} drugs...")
        
        created_plots = []
        for drug in top_drugs:
            try:
                plot_path = plot_drug_oxygen_timeseries(drug, max_wells_per_conc=3)
                if plot_path:
                    created_plots.append(plot_path)
                print()  # Add spacing between drugs
            except Exception as e:
                print(f"Error plotting {drug}: {e}")
                continue
        
        print(f"Created {len(created_plots)} plots in {OUTPUT_DIR}")
        
        # Suggest how to plot specific drugs
        print(f"\nTo plot a specific drug, use:")
        print(f"plot_drug_oxygen_timeseries('DrugName')")
        print(f"plot_drug_oxygen_timeseries('DrugName', plate_id='specific-plate-id')")

if __name__ == "__main__":
    main()