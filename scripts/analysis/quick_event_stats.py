#!/usr/bin/env python3
"""
Quick analysis of event detection statistics using existing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "step3_validation"

def analyze_existing_results():
    """Analyze existing spike detection results."""
    
    # Load spike features from Step 3
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    
    # Load event data
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    if event_path.exists():
        events_df = pd.read_parquet(event_path)
        media_changes = events_df[events_df['title'] == 'Medium Change']
    else:
        media_changes = pd.DataFrame()
    
    # Load Step 1 data for total plate count
    step1_df = pd.read_parquet(data_dir / "step1_quality_assessment_all_plates.parquet")
    
    print("=== EVENT DETECTION STATISTICS ===\n")
    
    # Basic counts
    print(f"Total plates in dataset: {step1_df['plate_id'].nunique()}")
    print(f"Plates processed in Step 3: {spike_df['plate_id'].nunique()}")
    print(f"Wells analyzed: {spike_df['well_id'].nunique()}")
    print(f"Total spike characterizations: {len(spike_df)}")
    
    if len(media_changes) > 0:
        print(f"\nRecorded media change events: {len(media_changes)}")
        print(f"Plates with recorded events: {media_changes['plate_id'].nunique()}")
    
    # Events per plate/well
    events_per_well = spike_df.groupby('well_id')['event_number'].max()
    events_per_plate = spike_df.groupby('plate_id')['event_number'].max()
    
    print(f"\nEvents per well: {events_per_well.mean():.1f} ± {events_per_well.std():.1f}")
    print(f"Events per plate (max): {events_per_plate.mean():.1f} ± {events_per_plate.std():.1f}")
    print(f"Max events in a single well: {events_per_well.max()}")
    
    # Event timing
    print(f"\nEvent timing distribution:")
    print(f"  First event: {spike_df[spike_df['event_number']==1]['event_time_hours'].mean():.1f} ± {spike_df[spike_df['event_number']==1]['event_time_hours'].std():.1f} hours")
    
    # Create simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Event Detection Statistics from Step 3 Results', fontsize=16)
    
    # 1. Events per well distribution
    ax = axes[0, 0]
    events_per_well.hist(bins=range(1, events_per_well.max()+2), ax=ax, edgecolor='black')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Number of Wells')
    ax.set_title('Distribution of Events per Well')
    ax.axvline(events_per_well.mean(), color='red', linestyle='--', 
               label=f'Mean: {events_per_well.mean():.1f}')
    ax.legend()
    
    # 2. Event timing by number
    ax = axes[0, 1]
    for event_num in range(1, min(6, spike_df['event_number'].max()+1)):
        event_times = spike_df[spike_df['event_number']==event_num]['event_time_hours']
        if len(event_times) > 0:
            ax.scatter([event_num]*len(event_times), event_times, alpha=0.5, s=30)
    
    # Add means
    mean_times = spike_df.groupby('event_number')['event_time_hours'].mean()
    ax.plot(mean_times.index, mean_times.values, 'r-', linewidth=2, label='Mean')
    
    ax.set_xlabel('Event Number')
    ax.set_ylabel('Time Since Start (hours)')
    ax.set_title('Event Timing Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Summary statistics
    ax = axes[1, 0]
    ax.axis('off')
    
    # Estimate untagged events
    if len(media_changes) > 0:
        # For the plates we processed, compare detected vs recorded
        processed_plates = spike_df['plate_id'].unique()
        recorded_in_processed = media_changes[media_changes['plate_id'].isin(processed_plates)]
        
        detected_events = len(spike_df)
        recorded_events_in_processed = len(recorded_in_processed)
        
        extrapolation_text = f"""
        Based on {len(processed_plates)} processed plates:
        
        • Detected spike events: {detected_events}
        • Recorded events in same plates: {recorded_events_in_processed}
        • Detection rate: {detected_events/max(1, recorded_events_in_processed)*100:.0f}%
        
        If extrapolated to all {step1_df['plate_id'].nunique()} plates:
        • Expected total events: ~{detected_events * step1_df['plate_id'].nunique() / len(processed_plates):.0f}
        • Recorded total events: {len(media_changes)}
        • Potential untagged events: ~{max(0, detected_events * step1_df['plate_id'].nunique() / len(processed_plates) - len(media_changes)):.0f}
        """
    else:
        extrapolation_text = "No recorded event data available for comparison"
    
    ax.text(0.05, 0.95, extrapolation_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Peak characteristics by event
    ax = axes[1, 1]
    event_stats = spike_df.groupby('event_number').agg({
        'peak_height': 'mean',
        'recovery_time': lambda x: x.notna().mean() * 100  # Recovery rate
    })
    
    ax2 = ax.twinx()
    
    bars = ax.bar(event_stats.index, event_stats['peak_height'], 
                   alpha=0.7, color='blue', label='Peak Height')
    line = ax2.plot(event_stats.index, event_stats['recovery_time'], 
                    'ro-', linewidth=2, markersize=8, label='Recovery Rate %')
    
    ax.set_xlabel('Event Number')
    ax.set_ylabel('Mean Peak Height (O₂%)', color='blue')
    ax2.set_ylabel('Recovery Detection Rate (%)', color='red')
    ax.set_title('Event Characteristics by Number')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'event_detection_quick_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {fig_dir}/event_detection_quick_stats.png")

if __name__ == "__main__":
    analyze_existing_results()