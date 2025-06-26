#!/usr/bin/env python3
"""
Analyze alignment between events and oxygen consumption spikes.

This script checks if recorded events (especially media changes) correspond
to spikes in oxygen consumption data, focusing on control and low concentrations
where drug effects should be minimal.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import zscore

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output directory
OUTPUT_DIR = Path("results/figures/event_alignment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_spikes(oxygen_series, window_size=5, z_threshold=2.5):
    """
    Detect spikes in oxygen consumption using z-score method.
    
    Args:
        oxygen_series: Time series of oxygen values
        window_size: Window for local baseline calculation
        z_threshold: Z-score threshold for spike detection
        
    Returns:
        spike_indices: Indices where spikes occur
        spike_magnitudes: Magnitude of each spike
    """
    if len(oxygen_series) < window_size * 2:
        return np.array([]), np.array([])
    
    # Calculate rolling baseline and std
    rolling_mean = oxygen_series.rolling(window=window_size, center=True).mean()
    rolling_std = oxygen_series.rolling(window=window_size, center=True).std()
    
    # Calculate z-scores relative to local baseline
    z_scores = (oxygen_series - rolling_mean) / (rolling_std + 1e-6)
    
    # Detect spikes (positive z-scores above threshold)
    spike_mask = z_scores > z_threshold
    spike_indices = np.where(spike_mask)[0]
    spike_magnitudes = z_scores[spike_mask].values
    
    return spike_indices, spike_magnitudes

def analyze_event_spike_alignment(plate_id, max_concentration=1.0):
    """
    Analyze alignment between events and oxygen spikes for a single plate.
    
    Args:
        plate_id: Plate ID to analyze
        max_concentration: Maximum drug concentration to include (µM)
        
    Returns:
        results: Dictionary with analysis results
    """
    print(f"\nAnalyzing plate: {plate_id}")
    
    with DataLoader() as loader:
        # Load oxygen data
        oxygen_data = loader.load_oxygen_data(plate_ids=[plate_id])
        
        if oxygen_data.empty:
            print(f"  No oxygen data found for plate {plate_id}")
            return None
        
        # Filter for control and low concentrations
        oxygen_data = oxygen_data[oxygen_data['concentration'] <= max_concentration].copy()
        oxygen_data['timestamp'] = pd.to_datetime(oxygen_data['timestamp'])
        oxygen_data = oxygen_data.sort_values(['well_id', 'timestamp'])
        
        print(f"  Found {len(oxygen_data)} measurements for {oxygen_data['well_id'].nunique()} wells")
        print(f"  Concentrations: {sorted(oxygen_data['concentration'].unique())}")
        
        # Load events
        events = loader.load_plate_events(plate_id)
        events['occurred_at'] = pd.to_datetime(events['occurred_at'])
        
        # Focus on media change events
        media_events = events[events['event_type'] == 'Medium Change'].copy()
        print(f"  Found {len(media_events)} media change events")
        
        if media_events.empty:
            print(f"  No media change events found")
            return None
        
        # Analyze each well
        results = {
            'plate_id': plate_id,
            'wells_analyzed': 0,
            'total_spikes': 0,
            'total_events': len(media_events),
            'aligned_events': 0,
            'alignment_scores': [],
            'well_results': []
        }
        
        # Process each well
        for well_id in oxygen_data['well_id'].unique():
            well_data = oxygen_data[oxygen_data['well_id'] == well_id].copy()
            well_data = well_data.sort_values('timestamp')
            
            if len(well_data) < 20:  # Need minimum data points
                continue
            
            # Detect spikes
            spike_indices, spike_magnitudes = detect_spikes(well_data['o2'])
            
            if len(spike_indices) == 0:
                continue
            
            spike_times = well_data.iloc[spike_indices]['timestamp']
            
            # Check alignment with events
            well_result = {
                'well_id': well_id,
                'concentration': well_data['concentration'].iloc[0],
                'num_spikes': len(spike_indices),
                'spike_times': spike_times.tolist(),
                'event_alignments': []
            }
            
            # For each event, find nearest spike
            for event_time in media_events['occurred_at']:
                # Find spikes within ±2 hours of event
                time_diffs = np.abs((spike_times - event_time).total_seconds() / 3600)
                
                if len(time_diffs) > 0 and time_diffs.min() < 2.0:  # Within 2 hours
                    nearest_idx = time_diffs.argmin()
                    alignment = {
                        'event_time': event_time,
                        'nearest_spike_time': spike_times.iloc[nearest_idx],
                        'time_diff_hours': time_diffs.min(),
                        'spike_magnitude': spike_magnitudes[nearest_idx]
                    }
                    well_result['event_alignments'].append(alignment)
                    results['alignment_scores'].append(time_diffs.min())
            
            results['well_results'].append(well_result)
            results['wells_analyzed'] += 1
            results['total_spikes'] += len(spike_indices)
        
        # Calculate summary statistics
        if results['alignment_scores']:
            results['aligned_events'] = len([s for s in results['alignment_scores'] if s < 1.0])
            results['mean_alignment_hours'] = np.mean(results['alignment_scores'])
            results['median_alignment_hours'] = np.median(results['alignment_scores'])
            results['alignment_rate'] = results['aligned_events'] / results['total_events']
        
        return results

def create_alignment_visualization(plate_results):
    """Create visualization of event-spike alignment."""
    
    if not plate_results or not plate_results['well_results']:
        print("No results to visualize")
        return
    
    # Select a representative well with good alignments
    best_well = None
    max_alignments = 0
    
    for well_result in plate_results['well_results']:
        if len(well_result['event_alignments']) > max_alignments:
            max_alignments = len(well_result['event_alignments'])
            best_well = well_result
    
    if not best_well:
        print("No wells with event alignments found")
        return
    
    # Load full data for visualization
    with DataLoader() as loader:
        well_data = loader.load_oxygen_data(plate_ids=[plate_results['plate_id']])
        well_data = well_data[well_data['well_id'] == best_well['well_id']].copy()
        well_data['timestamp'] = pd.to_datetime(well_data['timestamp'])
        well_data = well_data.sort_values('timestamp')
        
        events = loader.load_plate_events(plate_results['plate_id'])
        events['occurred_at'] = pd.to_datetime(events['occurred_at'])
        media_events = events[events['event_type'] == 'Medium Change']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot oxygen data
    ax1.plot(well_data['timestamp'], well_data['o2'], 'b-', alpha=0.7, label='O₂ Consumption')
    
    # Detect and mark spikes
    spike_indices, spike_magnitudes = detect_spikes(well_data['o2'])
    if len(spike_indices) > 0:
        spike_times = well_data.iloc[spike_indices]['timestamp']
        spike_values = well_data.iloc[spike_indices]['o2']
        ax1.scatter(spike_times, spike_values, color='red', s=100, 
                   marker='^', label='Detected Spikes', zorder=5)
    
    # Mark events
    for _, event in media_events.iterrows():
        ax1.axvline(x=event['occurred_at'], color='green', linestyle='--', 
                   alpha=0.7, linewidth=2, label='Media Change' if _ == 0 else '')
    
    # Add alignment annotations
    for alignment in best_well['event_alignments']:
        if alignment['time_diff_hours'] < 1.0:  # Good alignment
            ax1.annotate(f"{alignment['time_diff_hours']:.1f}h", 
                        xy=(alignment['nearest_spike_time'], 
                            well_data[well_data['timestamp'] == alignment['nearest_spike_time']]['o2'].iloc[0]),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Oxygen Consumption (%)')
    ax1.set_title(f'Event-Spike Alignment Analysis\n'
                  f'Well: {best_well["well_id"]} | Concentration: {best_well["concentration"]} µM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot alignment histogram
    all_alignments = []
    for well in plate_results['well_results']:
        for align in well['event_alignments']:
            all_alignments.append(align['time_diff_hours'])
    
    if all_alignments:
        ax2.hist(all_alignments, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=1.0, color='red', linestyle='--', label='1 hour threshold')
        ax2.set_xlabel('Time Difference (hours)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Event-Spike Time Differences')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"event_spike_alignment_{plate_results['plate_id']}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {filepath}")
    
    plt.show()

def analyze_multiple_plates(num_plates=3):
    """Analyze event-spike alignment across multiple plates."""
    
    print("=" * 70)
    print("EVENT-SPIKE ALIGNMENT ANALYSIS")
    print("=" * 70)
    
    with DataLoader() as loader:
        # Get plates with both oxygen data and events
        plates_query = """
            SELECT DISTINCT p.plate_id, COUNT(DISTINCT e.id) as event_count
            FROM processed_data p
            JOIN event_table e ON p.plate_id = e.plate_id
            WHERE e.title = 'Medium Change'
              AND e.is_excluded = false
            GROUP BY p.plate_id
            HAVING COUNT(DISTINCT e.id) > 2
            LIMIT 10
        """
        
        # Use a simpler query that works with local database
        try:
            plates_df = loader._execute_and_convert(plates_query)
            plate_ids = plates_df['plate_id'].tolist()[:num_plates]
        except:
            # Fallback: just get some plates
            all_events = loader.load_all_events()
            media_events = all_events[all_events['title'] == 'Medium Change']
            plate_ids = media_events['plate_id'].value_counts().head(num_plates).index.tolist()
    
    all_results = []
    
    for plate_id in plate_ids:
        results = analyze_event_spike_alignment(plate_id)
        if results:
            all_results.append(results)
            
            # Create visualization for this plate
            create_alignment_visualization(results)
            
            # Print summary
            print(f"\nSummary for plate {plate_id}:")
            print(f"  Wells analyzed: {results['wells_analyzed']}")
            print(f"  Total spikes detected: {results['total_spikes']}")
            print(f"  Media change events: {results['total_events']}")
            
            if 'alignment_rate' in results:
                print(f"  Events aligned (< 1 hour): {results['aligned_events']} ({results['alignment_rate']:.1%})")
                print(f"  Mean alignment: {results['mean_alignment_hours']:.2f} hours")
                print(f"  Median alignment: {results['median_alignment_hours']:.2f} hours")
    
    # Create summary plot across all plates
    if all_results:
        create_summary_plot(all_results)
    
    return all_results

def create_summary_plot(all_results):
    """Create summary visualization across all analyzed plates."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Alignment rates by plate
    ax = axes[0, 0]
    plate_ids = [r['plate_id'][:8] for r in all_results if 'alignment_rate' in r]
    alignment_rates = [r['alignment_rate'] for r in all_results if 'alignment_rate' in r]
    
    if plate_ids:
        ax.bar(plate_ids, alignment_rates, alpha=0.7, color='green')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax.set_xlabel('Plate ID')
        ax.set_ylabel('Alignment Rate')
        ax.set_title('Event-Spike Alignment Rate by Plate')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Distribution of all alignment times
    ax = axes[0, 1]
    all_alignments = []
    for result in all_results:
        all_alignments.extend(result.get('alignment_scores', []))
    
    if all_alignments:
        ax.hist(all_alignments, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', label='1 hour threshold')
        ax.set_xlabel('Time Difference (hours)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of All Event-Spike Alignments')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Spikes per well by concentration
    ax = axes[1, 0]
    conc_spikes = {}
    for result in all_results:
        for well in result['well_results']:
            conc = well['concentration']
            if conc not in conc_spikes:
                conc_spikes[conc] = []
            conc_spikes[conc].append(well['num_spikes'])
    
    if conc_spikes:
        concentrations = sorted(conc_spikes.keys())
        spike_means = [np.mean(conc_spikes[c]) for c in concentrations]
        spike_stds = [np.std(conc_spikes[c]) for c in concentrations]
        
        ax.bar(range(len(concentrations)), spike_means, yerr=spike_stds, 
               alpha=0.7, color='orange', capsize=5)
        ax.set_xticks(range(len(concentrations)))
        ax.set_xticklabels([f"{c:.1f}" for c in concentrations])
        ax.set_xlabel('Concentration (µM)')
        ax.set_ylabel('Spikes per Well')
        ax.set_title('Average Spike Count by Concentration')
        ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"
    
    total_wells = sum(r['wells_analyzed'] for r in all_results)
    total_spikes = sum(r['total_spikes'] for r in all_results)
    total_events = sum(r['total_events'] for r in all_results)
    aligned_events = sum(r.get('aligned_events', 0) for r in all_results)
    
    summary_text += f"Plates analyzed: {len(all_results)}\n"
    summary_text += f"Wells analyzed: {total_wells}\n"
    summary_text += f"Total spikes detected: {total_spikes}\n"
    summary_text += f"Total media events: {total_events}\n"
    summary_text += f"Aligned events (<1h): {aligned_events} ({aligned_events/total_events:.1%})\n"
    
    if all_alignments:
        summary_text += f"\nAlignment times:\n"
        summary_text += f"  Mean: {np.mean(all_alignments):.2f} hours\n"
        summary_text += f"  Median: {np.median(all_alignments):.2f} hours\n"
        summary_text += f"  < 30 min: {sum(1 for a in all_alignments if a < 0.5)} events\n"
        summary_text += f"  < 1 hour: {sum(1 for a in all_alignments if a < 1.0)} events\n"
        summary_text += f"  < 2 hours: {sum(1 for a in all_alignments if a < 2.0)} events\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Event-Spike Alignment Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filepath = OUTPUT_DIR / "event_spike_alignment_summary.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot: {filepath}")
    
    plt.show()

def main():
    """Main analysis function."""
    
    # Analyze multiple plates
    results = analyze_multiple_plates(num_plates=3)
    
    if results:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Analyzed {len(results)} plates")
        print(f"Results saved to: {OUTPUT_DIR}")
    else:
        print("\nNo results to analyze")

if __name__ == "__main__":
    main()