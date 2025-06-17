#!/usr/bin/env python3
"""
Quick event integration for testing - creates a simplified dataset
with event information for feature engineering pipeline development.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def create_quick_integration():
    """Create a quick integration of oxygen and event data for testing."""
    print("=== Quick Event Integration ===\n")
    
    data_dir = project_root / "data" / "raw"
    
    # Load event data
    print("Loading event data...")
    events = pd.read_parquet(data_dir / "event_data.parquet")
    events['occurred_at'] = pd.to_datetime(events['occurred_at']).dt.tz_localize(None)
    
    # Get key event types
    dosing_events = events[events['title'].isin(['Drugs Start', 'Drugs Added'])]
    media_events = events[events['title'] == 'Medium Change']
    
    print(f"  Found {len(dosing_events)} dosing events across {dosing_events['plate_id'].nunique()} plates")
    print(f"  Found {len(media_events)} media change events across {media_events['plate_id'].nunique()} plates")
    
    # Create plate-level event summary
    plate_events = []
    
    for plate_id in events['plate_id'].unique():
        plate_dosing = dosing_events[dosing_events['plate_id'] == plate_id]
        plate_media = media_events[media_events['plate_id'] == plate_id]
        
        plate_events.append({
            'plate_id': plate_id,
            'has_dosing': len(plate_dosing) > 0,
            'dosing_time': plate_dosing['occurred_at'].min() if len(plate_dosing) > 0 else None,
            'n_media_changes': len(plate_media),
            'media_change_times': plate_media['occurred_at'].tolist() if len(plate_media) > 0 else [],
            'first_media_change': plate_media['occurred_at'].min() if len(plate_media) > 0 else None,
            'last_media_change': plate_media['occurred_at'].max() if len(plate_media) > 0 else None
        })
    
    plate_summary = pd.DataFrame(plate_events)
    
    # Save plate-level event summary
    summary_path = data_dir / "plate_event_summary.parquet"
    plate_summary.to_parquet(summary_path, index=False)
    print(f"Saved plate event summary to: {summary_path}")
    
    # Show statistics
    print(f"\n=== Event Statistics ===")
    print(f"Total plates: {len(plate_summary)}")
    print(f"Plates with dosing: {plate_summary['has_dosing'].sum()}")
    print(f"Plates with media changes: {(plate_summary['n_media_changes'] > 0).sum()}")
    print(f"Mean media changes per plate: {plate_summary['n_media_changes'].mean():.1f}")
    
    # Calculate typical dosing-to-media timing
    plates_with_both = plate_summary[(plate_summary['has_dosing']) & (plate_summary['n_media_changes'] > 0)]
    
    if len(plates_with_both) > 0:
        timing_diffs = []
        for _, row in plates_with_both.iterrows():
            if row['dosing_time'] and row['first_media_change']:
                diff_hours = (row['first_media_change'] - row['dosing_time']).total_seconds() / 3600
                timing_diffs.append(diff_hours)
        
        if timing_diffs:
            print(f"Time from dosing to first media change: {np.mean(timing_diffs):.1f} ± {np.std(timing_diffs):.1f} hours")
    
    return plate_summary


def load_sample_oxygen_data(n_plates=5):
    """Load a sample of oxygen data for testing."""
    print(f"\nLoading sample oxygen data ({n_plates} plates)...")
    
    data_dir = project_root / "data" / "raw"
    
    # Load small sample
    oxygen = pd.read_parquet(data_dir / "processed_data_updated.parquet")
    well_map = pd.read_parquet(data_dir / "well_map_data_updated.parquet")
    
    # Sample plates
    sample_plates = oxygen['plate_id'].unique()[:n_plates]
    oxygen_sample = oxygen[oxygen['plate_id'].isin(sample_plates)]
    well_map_sample = well_map[well_map['plate_id'].isin(sample_plates)]
    
    # Merge
    data = oxygen_sample.merge(well_map_sample, on=['plate_id', 'well_number'], how='left')
    
    # Basic processing
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
    data['well_id'] = data['plate_id'].astype(str) + '_' + data['well_number'].astype(str)
    data['value'] = data['median_o2']
    data['is_control'] = data['drug'].isna() | (data['drug'] == 'DMSO') | (data['concentration'] == 0)
    
    print(f"  Sample data: {len(data):,} measurements from {data['well_id'].nunique():,} wells")
    
    return data


def demonstrate_event_aware_features():
    """Demonstrate how to extract event-aware features."""
    print(f"\n=== Event-Aware Feature Demo ===")
    
    # Load plate event summary
    data_dir = project_root / "data" / "raw"
    plate_summary = pd.read_parquet(data_dir / "plate_event_summary.parquet")
    
    # Load sample oxygen data
    sample_data = load_sample_oxygen_data(n_plates=3)
    
    # Add event information to sample
    enhanced_data = []
    
    for plate_id in sample_data['plate_id'].unique():
        plate_data = sample_data[sample_data['plate_id'] == plate_id]
        plate_info = plate_summary[plate_summary['plate_id'] == plate_id]
        
        if len(plate_info) == 0:
            print(f"  No event data for plate {plate_id}")
            continue
            
        plate_info = plate_info.iloc[0]
        
        # Add event features
        plate_data = plate_data.copy()
        
        # Basic event flags
        plate_data['has_dosing_events'] = plate_info['has_dosing']
        plate_data['n_media_changes'] = plate_info['n_media_changes']
        
        # Time-based features
        if plate_info['dosing_time']:
            dosing_time = pd.to_datetime(plate_info['dosing_time'])
            plate_data['hours_since_dosing'] = (plate_data['timestamp'] - dosing_time).dt.total_seconds() / 3600
            plate_data['is_pre_dosing'] = plate_data['hours_since_dosing'] < 0
        else:
            plate_data['hours_since_dosing'] = np.nan
            plate_data['is_pre_dosing'] = True  # Assume all pre-dosing if no dosing event
        
        # Media change features
        if plate_info['n_media_changes'] > 0:
            media_times = [pd.to_datetime(t) for t in plate_info['media_change_times']]
            
            # Time since last media change
            plate_data['hours_since_last_media_change'] = np.nan
            
            for i, row in plate_data.iterrows():
                timestamp = row['timestamp']
                past_media = [t for t in media_times if t <= timestamp]
                
                if past_media:
                    last_media = max(past_media)
                    hours_since = (timestamp - last_media).total_seconds() / 3600
                    plate_data.loc[i, 'hours_since_last_media_change'] = hours_since
        
        enhanced_data.append(plate_data)
    
    if enhanced_data:
        final_data = pd.concat(enhanced_data, ignore_index=True)
        
        # Show feature statistics
        print(f"Enhanced sample data: {len(final_data):,} measurements")
        print(f"Wells with dosing info: {final_data['hours_since_dosing'].notna().sum():,}")
        print(f"Pre-dosing measurements: {final_data['is_pre_dosing'].sum():,}")
        print(f"Measurements with media change timing: {final_data['hours_since_last_media_change'].notna().sum():,}")
        
        # Save enhanced sample
        sample_path = data_dir / "event_enhanced_sample.parquet"
        final_data.to_parquet(sample_path, index=False)
        print(f"Enhanced sample saved to: {sample_path}")
        
        return final_data
    
    return None


def main():
    """Main pipeline."""
    print("Creating quick event integration for feature engineering development...\n")
    
    # Create plate-level event summary
    plate_summary = create_quick_integration()
    
    # Demonstrate event-aware features
    enhanced_sample = demonstrate_event_aware_features()
    
    print(f"\n✅ Quick integration complete!")
    print(f"🎯 Ready for event-aware feature engineering development")
    print(f"📋 Next steps:")
    print(f"   1. Implement multi-timescale catch22 with event windows")
    print(f"   2. Add dose-response Hill curve fitting")
    print(f"   3. Extract baseline period features (pre-dosing)")
    print(f"   4. Build media-change-aware temporal features")
    
    return plate_summary, enhanced_sample


if __name__ == "__main__":
    main()