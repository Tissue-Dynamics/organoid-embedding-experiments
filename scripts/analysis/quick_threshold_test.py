#!/usr/bin/env python3
"""
Quick test of different thresholds on recorded events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "step3_validation"

def analyze_existing_spikes():
    """Analyze characteristics of spikes at recorded event times."""
    # Load spike features from Step 3
    spike_df = pd.read_parquet(data_dir / "step3_spike_features.parquet")
    
    # Load recorded events
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    events_df = pd.read_parquet(event_path)
    media_changes = events_df[events_df['title'] == 'Medium Change']
    
    print("=== Analyzing Spike Characteristics at Recorded Events ===\n")
    
    # For each plate with recorded events
    plates_with_events = media_changes['plate_id'].unique()
    
    matched_spikes = []
    
    for plate_id in plates_with_events:
        plate_spikes = spike_df[spike_df['plate_id'] == plate_id]
        plate_events = media_changes[media_changes['plate_id'] == plate_id]
        
        if len(plate_spikes) > 0 and len(plate_events) > 0:
            # Get spike characteristics
            print(f"\nPlate {plate_id}:")
            print(f"  Recorded events: {len(plate_events)}")
            print(f"  Detected spikes: {len(plate_spikes)}")
            
            # Analyze spike heights and characteristics
            print(f"  Spike heights: min={plate_spikes['peak_height'].min():.1f}, "
                  f"max={plate_spikes['peak_height'].max():.1f}, "
                  f"mean={plate_spikes['peak_height'].mean():.1f}")
            
            # Store for analysis
            matched_spikes.extend(plate_spikes.to_dict('records'))
    
    if matched_spikes:
        matched_df = pd.DataFrame(matched_spikes)
        
        # Calculate statistics
        print("\n=== OVERALL SPIKE STATISTICS (at recorded events) ===")
        print(f"Total spikes analyzed: {len(matched_df)}")
        print(f"\nPeak Heights:")
        print(f"  Minimum: {matched_df['peak_height'].min():.1f}% O₂")
        print(f"  5th percentile: {matched_df['peak_height'].quantile(0.05):.1f}% O₂")
        print(f"  25th percentile: {matched_df['peak_height'].quantile(0.25):.1f}% O₂")
        print(f"  Median: {matched_df['peak_height'].median():.1f}% O₂")
        print(f"  Mean: {matched_df['peak_height'].mean():.1f}% O₂")
        print(f"  75th percentile: {matched_df['peak_height'].quantile(0.75):.1f}% O₂")
        print(f"  95th percentile: {matched_df['peak_height'].quantile(0.95):.1f}% O₂")
        print(f"  Maximum: {matched_df['peak_height'].max():.1f}% O₂")
        
        print(f"\nTime to Peak:")
        print(f"  Mean: {matched_df['peak_time_relative'].mean():.1f} hours")
        print(f"  Std: {matched_df['peak_time_relative'].std():.1f} hours")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Spike Characteristics at Recorded Media Change Events', fontsize=16)
        
        # 1. Peak height distribution
        ax = axes[0, 0]
        ax.hist(matched_df['peak_height'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(matched_df['peak_height'].quantile(0.05), color='red', linestyle='--', 
                   label=f'5th percentile: {matched_df["peak_height"].quantile(0.05):.1f}')
        ax.axvline(matched_df['peak_height'].median(), color='green', linestyle='-', 
                   label=f'Median: {matched_df["peak_height"].median():.1f}')
        ax.set_xlabel('Peak Height (% O₂)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Peak Heights')
        ax.legend()
        
        # 2. Peak height by event number
        ax = axes[0, 1]
        event_heights = matched_df.groupby('event_number')['peak_height'].agg(['mean', 'std', 'count'])
        ax.errorbar(event_heights.index, event_heights['mean'], 
                   yerr=event_heights['std'], fmt='o-', capsize=5)
        ax.set_xlabel('Event Number')
        ax.set_ylabel('Peak Height (% O₂)')
        ax.set_title('Peak Height by Event Number')
        ax.grid(True, alpha=0.3)
        
        # 3. Recommended thresholds
        ax = axes[1, 0]
        ax.axis('off')
        
        min_height = matched_df['peak_height'].quantile(0.05)
        
        threshold_text = f"""
        RECOMMENDED THRESHOLDS
        
        Based on recorded event analysis:
        
        Conservative (captures 95% of recorded events):
        • Min spike height: {min_height:.1f}% O₂
        • Min sharpness: ~{min_height/2:.1f}% O₂/hour
        • Min well fraction: 50%
        
        Strict (captures ~80% of recorded events):
        • Min spike height: {matched_df['peak_height'].quantile(0.20):.1f}% O₂
        • Min sharpness: ~{matched_df['peak_height'].quantile(0.20)/2:.1f}% O₂/hour
        • Min well fraction: 70%
        
        Very Strict (captures ~50% of recorded events):
        • Min spike height: {matched_df['peak_height'].median():.1f}% O₂
        • Min sharpness: ~{matched_df['peak_height'].median()/2:.1f}% O₂/hour
        • Min well fraction: 80%
        """
        
        ax.text(0.05, 0.95, threshold_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 4. Spike height vs well count
        ax = axes[1, 1]
        # Count wells per event
        wells_per_event = matched_df.groupby(['plate_id', 'event_number']).size()
        heights_per_event = matched_df.groupby(['plate_id', 'event_number'])['peak_height'].mean()
        
        ax.scatter(wells_per_event, heights_per_event, alpha=0.6)
        ax.set_xlabel('Number of Wells with Spike')
        ax.set_ylabel('Mean Peak Height (% O₂)')
        ax.set_title('Spike Height vs Number of Affected Wells')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'recorded_event_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {fig_dir}/recorded_event_characteristics.png")
        
        # Test thresholds
        print("\n=== TESTING THRESHOLD COMBINATIONS ===")
        
        test_configs = [
            ("Very Lenient", 5, 2.5, 0.3),
            ("Lenient", 10, 5, 0.5),
            ("Moderate", 15, 7.5, 0.6),
            ("Strict", 20, 10, 0.7),
            ("Very Strict", 25, 12.5, 0.8),
            ("Ultra Strict", 30, 15, 0.9)
        ]
        
        for name, height, sharpness, well_frac in test_configs:
            # Count how many recorded spikes would be detected
            detected = matched_df[matched_df['peak_height'] >= height]
            detection_rate = len(detected) / len(matched_df) * 100
            
            print(f"\n{name} (height≥{height}%, sharpness≥{sharpness}%/hr, wells≥{well_frac*100:.0f}%):")
            print(f"  Would detect {len(detected)}/{len(matched_df)} ({detection_rate:.0f}%) of recorded event spikes")
            
            if detection_rate >= 95:
                print(f"  ✅ Captures nearly all recorded events")
            elif detection_rate >= 80:
                print(f"  ⚠️  Misses some recorded events")
            else:
                print(f"  ❌ Too strict - misses many recorded events")

def main():
    """Run quick threshold analysis."""
    analyze_existing_spikes()

if __name__ == "__main__":
    main()