#!/usr/bin/env python3
"""
Create overlay visualizations of events and oxygen spikes.

This script creates detailed plots showing oxygen consumption data with
event markers and detected spikes to visually verify alignment.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.signal import find_peaks

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output directory
OUTPUT_DIR = Path("results/figures/event_spike_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_event_spike_overlay(plate_id, drug_name=None, max_concentration=1.0, hours_window=48):
    """
    Create detailed overlay plot of oxygen data with events and spikes.
    
    Args:
        plate_id: Plate ID to visualize
        drug_name: Specific drug to focus on (None for all)
        max_concentration: Maximum concentration to include
        hours_window: Time window to display (hours from start)
    """
    
    print(f"\nCreating overlay for plate: {plate_id}")
    
    with DataLoader() as loader:
        # Load oxygen data
        oxygen_data = loader.load_oxygen_data(plate_ids=[plate_id])
        
        if oxygen_data.empty:
            print(f"No oxygen data found for plate {plate_id}")
            return
        
        # Filter by drug if specified
        if drug_name:
            oxygen_data = oxygen_data[oxygen_data['drug'] == drug_name]
            if oxygen_data.empty:
                print(f"No data found for drug {drug_name}")
                return
        
        # Filter by concentration
        oxygen_data = oxygen_data[oxygen_data['concentration'] <= max_concentration].copy()
        oxygen_data['timestamp'] = pd.to_datetime(oxygen_data['timestamp'])
        
        # Get time window
        start_time = oxygen_data['timestamp'].min()
        end_time = start_time + pd.Timedelta(hours=hours_window)
        oxygen_data = oxygen_data[oxygen_data['timestamp'] <= end_time]
        
        print(f"  Data range: {start_time} to {oxygen_data['timestamp'].max()}")
        print(f"  Wells: {oxygen_data['well_id'].nunique()}")
        print(f"  Concentrations: {sorted(oxygen_data['concentration'].unique())}")
        
        # Load events
        events = loader.load_plate_events(plate_id)
        events['occurred_at'] = pd.to_datetime(events['occurred_at'])
        events = events[(events['occurred_at'] >= start_time) & (events['occurred_at'] <= end_time)]
        
        # Group by concentration
        conc_groups = oxygen_data.groupby('concentration')
        n_concs = len(conc_groups)
        
        # Create subplots
        fig, axes = plt.subplots(n_concs, 1, figsize=(16, 4*n_concs), sharex=True)
        if n_concs == 1:
            axes = [axes]
        
        # Color map for events
        event_colors = {
            'Medium Change': 'red',
            'Drugs Start': 'green',
            'Data Exclusion': 'black',
            'Communication Failure': 'orange',
            'Experiment End': 'purple'
        }
        
        # Plot each concentration
        for idx, (conc, conc_data) in enumerate(conc_groups):
            ax = axes[idx]
            
            # Plot each well
            wells = conc_data['well_id'].unique()[:3]  # Limit to 3 wells per concentration
            
            for well_idx, well_id in enumerate(wells):
                well_data = conc_data[conc_data['well_id'] == well_id].sort_values('timestamp')
                
                if len(well_data) < 10:
                    continue
                
                # Plot oxygen data
                alpha = 0.8 if well_idx == 0 else 0.4
                line = ax.plot(well_data['timestamp'], well_data['o2'], 
                              alpha=alpha, linewidth=1.5 if well_idx == 0 else 1,
                              label=f'Well {well_idx+1}' if well_idx < 3 else None)[0]
                
                # Detect spikes using scipy
                if well_idx == 0:  # Only detect spikes for first well
                    # Parameters for spike detection
                    prominence = well_data['o2'].std() * 1.5
                    distance = 10  # Minimum distance between spikes
                    
                    peaks, properties = find_peaks(well_data['o2'].values, 
                                                 prominence=prominence,
                                                 distance=distance)
                    
                    if len(peaks) > 0:
                        spike_times = well_data.iloc[peaks]['timestamp']
                        spike_values = well_data.iloc[peaks]['o2']
                        
                        ax.scatter(spike_times, spike_values, 
                                 color='red', s=100, marker='^', 
                                 zorder=5, label='Detected Spikes')
                        
                        # Annotate spike times
                        for st, sv in zip(spike_times, spike_values):
                            ax.annotate(st.strftime('%H:%M'), 
                                      xy=(st, sv), 
                                      xytext=(0, 10), 
                                      textcoords='offset points',
                                      fontsize=8, ha='center')
            
            # Add event lines
            for event_type, event_group in events.groupby('event_type'):
                color = event_colors.get(event_type, 'gray')
                for _, event in event_group.iterrows():
                    ax.axvline(x=event['occurred_at'], 
                             color=color, linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add event label at top
                    ax.text(event['occurred_at'], ax.get_ylim()[1]*0.95, 
                           event_type[:3], rotation=90, 
                           verticalalignment='top', 
                           color=color, fontsize=8, fontweight='bold')
            
            # Formatting
            ax.set_ylabel(f'O₂ (%) - {conc} µM', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Add spike detection zones
            if 'Medium Change' in events['event_type'].values:
                media_times = events[events['event_type'] == 'Medium Change']['occurred_at']
                for mt in media_times:
                    # Shade ±1 hour window around media change
                    ax.axvspan(mt - pd.Timedelta(hours=1), 
                             mt + pd.Timedelta(hours=1),
                             alpha=0.1, color='red', zorder=0)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Time')
        
        # Title
        title = f'Event-Spike Overlay: Plate {plate_id[:8]}...'
        if drug_name:
            title += f' | Drug: {drug_name}'
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend for events
        event_handles = []
        for event_type, color in event_colors.items():
            if event_type in events['event_type'].values:
                handle = plt.Line2D([0], [0], color=color, linestyle='--', 
                                  alpha=0.7, linewidth=2, label=event_type)
                event_handles.append(handle)
        
        if event_handles:
            axes[0].legend(handles=event_handles, loc='upper left', 
                         bbox_to_anchor=(1.01, 1), title='Events')
        
        plt.tight_layout()
        
        # Save figure
        drug_suffix = f"_{drug_name.replace(' ', '_')}" if drug_name else ""
        filename = f"overlay_{plate_id[:8]}{drug_suffix}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        
        plt.show()
        
        return fig

def create_spike_timing_analysis(plate_id):
    """
    Analyze spike timing relative to media change events.
    """
    
    print(f"\nAnalyzing spike timing for plate: {plate_id}")
    
    with DataLoader() as loader:
        # Load data
        oxygen_data = loader.load_oxygen_data(plate_ids=[plate_id])
        events = loader.load_plate_events(plate_id)
        
        if oxygen_data.empty or events.empty:
            print("  Insufficient data")
            return
        
        oxygen_data['timestamp'] = pd.to_datetime(oxygen_data['timestamp'])
        events['occurred_at'] = pd.to_datetime(events['occurred_at'])
        
        # Focus on media changes
        media_events = events[events['event_type'] == 'Medium Change']
        
        if media_events.empty:
            print("  No media change events found")
            return
        
        # Analyze control wells only
        control_data = oxygen_data[oxygen_data['concentration'] == 0.0]
        
        spike_timings = []
        
        # For each media change event
        for _, event in media_events.iterrows():
            event_time = event['occurred_at']
            
            # Look at ±4 hour window
            window_start = event_time - pd.Timedelta(hours=4)
            window_end = event_time + pd.Timedelta(hours=4)
            
            # Get data in window for each well
            window_data = control_data[
                (control_data['timestamp'] >= window_start) & 
                (control_data['timestamp'] <= window_end)
            ]
            
            for well_id in window_data['well_id'].unique():
                well_data = window_data[window_data['well_id'] == well_id].sort_values('timestamp')
                
                if len(well_data) < 5:
                    continue
                
                # Detect spikes
                peaks, _ = find_peaks(well_data['o2'].values, 
                                    prominence=well_data['o2'].std() * 1.5)
                
                if len(peaks) > 0:
                    spike_times = well_data.iloc[peaks]['timestamp']
                    
                    # Calculate time relative to event
                    for spike_time in spike_times:
                        time_diff = (spike_time - event_time).total_seconds() / 3600
                        spike_timings.append({
                            'event_time': event_time,
                            'spike_time': spike_time,
                            'time_diff_hours': time_diff,
                            'well_id': well_id
                        })
        
        if not spike_timings:
            print("  No spikes detected near events")
            return
        
        # Create timing distribution plot
        spike_df = pd.DataFrame(spike_timings)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Histogram of spike timings
        ax1.hist(spike_df['time_diff_hours'], bins=40, range=(-4, 4), 
                alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Media Change')
        ax1.axvspan(-0.5, 0.5, alpha=0.2, color='red', label='±30 min window')
        ax1.set_xlabel('Time Relative to Media Change (hours)')
        ax1.set_ylabel('Number of Spikes')
        ax1.set_title(f'Spike Timing Distribution - Plate {plate_id[:8]}...')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_times = np.sort(spike_df['time_diff_hours'])
        cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        
        ax2.plot(sorted_times, cumulative, 'b-', linewidth=2)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time Relative to Media Change (hours)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution of Spike Timing')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-4, 4)
        
        # Add statistics
        within_30min = sum(abs(t) < 0.5 for t in spike_df['time_diff_hours'])
        within_1hr = sum(abs(t) < 1.0 for t in spike_df['time_diff_hours'])
        
        stats_text = f"Statistics:\n"
        stats_text += f"Total spikes: {len(spike_df)}\n"
        stats_text += f"Within ±30 min: {within_30min} ({within_30min/len(spike_df):.1%})\n"
        stats_text += f"Within ±1 hour: {within_1hr} ({within_1hr/len(spike_df):.1%})\n"
        stats_text += f"Mean offset: {spike_df['time_diff_hours'].mean():.2f} hours\n"
        stats_text += f"Median offset: {spike_df['time_diff_hours'].median():.2f} hours"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"spike_timing_{plate_id[:8]}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        
        plt.show()

def main():
    """Main function to create event-spike visualizations."""
    
    print("=" * 70)
    print("EVENT-SPIKE OVERLAY VISUALIZATION")
    print("=" * 70)
    
    with DataLoader() as loader:
        # Find plates with media change events
        events = loader.load_all_events()
        media_events = events[events['title'] == 'Medium Change']
        
        # Get top plates by event count
        top_plates = media_events['plate_id'].value_counts().head(3).index.tolist()
        
        print(f"\nAnalyzing {len(top_plates)} plates with most media change events")
        
        for plate_id in top_plates:
            # Create overlay visualization
            create_event_spike_overlay(plate_id, max_concentration=1.0, hours_window=72)
            
            # Create timing analysis
            create_spike_timing_analysis(plate_id)
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()