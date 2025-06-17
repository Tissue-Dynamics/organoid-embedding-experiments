#!/usr/bin/env python3
"""
Integrated data loader that combines oxygen time series data with event data.
This creates the foundation for event-aware feature engineering.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()


class IntegratedDataLoader:
    """
    Loads and integrates oxygen time series data with experimental events.
    Provides the foundation for event-aware feature engineering.
    """
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or (project_root / "data" / "raw")
        self.oxygen_data = None
        self.well_map = None
        self.event_data = None
        self.integrated_data = None
        
    def load_oxygen_data(self):
        """Load oxygen time series data."""
        print("Loading oxygen time series data...")
        
        # Load from parquet files
        processed_data = pd.read_parquet(self.data_dir / "processed_data_updated.parquet")
        well_map = pd.read_parquet(self.data_dir / "well_map_data_updated.parquet")
        
        # Merge on plate_id and well_number
        result = processed_data.merge(well_map, on=['plate_id', 'well_number'], how='left', suffixes=('', '_map'))
        
        # Standardize column names
        result['value'] = result['median_o2']
        result['drug_name'] = result['drug']
        result['well_id'] = result['plate_id'].astype(str) + '_' + result['well_number'].astype(str)
        result['experiment_id'] = result['plate_id']
        
        # Convert timestamps and make timezone naive
        result['timestamp'] = pd.to_datetime(result['timestamp']).dt.tz_localize(None)
        result = result.sort_values(['plate_id', 'well_number', 'timestamp'])
        
        # Calculate elapsed time per plate
        result['elapsed_hours'] = 0.0
        result['elapsed_days'] = 0.0
        
        for plate_id in result['plate_id'].unique():
            plate_mask = result['plate_id'] == plate_id
            min_time = result.loc[plate_mask, 'timestamp'].min()
            elapsed = (result.loc[plate_mask, 'timestamp'] - min_time).dt.total_seconds() / 3600
            result.loc[plate_mask, 'elapsed_hours'] = elapsed
            result.loc[plate_mask, 'elapsed_days'] = elapsed / 24
        
        # Identify controls
        result['is_control'] = result['drug'].isna() | (result['drug'] == 'DMSO') | (result['concentration'] == 0)
        
        # Handle exclusions
        if 'is_excluded_map' in result.columns:
            result['is_excluded'] = result['is_excluded_map'].fillna(False).astype(bool)
        else:
            result['is_excluded'] = result['is_excluded'].fillna(False).astype(bool)
        
        self.oxygen_data = result
        print(f"  Loaded {len(result):,} oxygen measurements")
        print(f"  Unique wells: {result['well_id'].nunique():,}")
        print(f"  Unique plates: {result['plate_id'].nunique():,}")
        
        return result
    
    def load_event_data(self):
        """Load experimental event data."""
        print("Loading experimental event data...")
        
        event_path = self.data_dir / "event_data.parquet"
        if not event_path.exists():
            print(f"  Warning: Event data not found at {event_path}")
            print("  Run scripts/database/download_event_data.py first")
            return None
        
        events = pd.read_parquet(event_path)
        
        # Convert timestamps and make timezone naive for comparison
        events['created_at'] = pd.to_datetime(events['created_at']).dt.tz_localize(None)
        events['occurred_at'] = pd.to_datetime(events['occurred_at']).dt.tz_localize(None)
        
        self.event_data = events
        print(f"  Loaded {len(events):,} events")
        print(f"  Event types: {', '.join(events['title'].value_counts().head(5).index.tolist())}")
        
        return events
    
    def integrate_events_with_oxygen(self):
        """Integrate event data with oxygen time series."""
        print("Integrating events with oxygen data...")
        
        if self.oxygen_data is None:
            self.load_oxygen_data()
        if self.event_data is None:
            self.load_event_data()
        
        if self.event_data is None:
            print("  No event data available, returning oxygen data only")
            self.integrated_data = self.oxygen_data.copy()
            return self.integrated_data
        
        # Add event flags to oxygen data
        integrated = self.oxygen_data.copy()
        
        # Initialize event columns
        integrated['has_media_change'] = False
        integrated['has_dosing_event'] = False
        integrated['hours_since_last_media_change'] = np.nan
        integrated['hours_since_dosing'] = np.nan
        integrated['media_change_number'] = 0
        integrated['time_to_next_media_change'] = np.nan
        
        # Process each plate (sample first 5 for testing)
        for plate_id in integrated['plate_id'].unique()[:5]:
            plate_mask = integrated['plate_id'] == plate_id
            plate_events = self.event_data[self.event_data['plate_id'] == plate_id]
            
            if len(plate_events) == 0:
                continue
                
            # Get media change events for this plate
            media_events = plate_events[plate_events['title'] == 'Medium Change'].sort_values('occurred_at')
            dosing_events = plate_events[plate_events['title'].isin(['Drugs Start', 'Drugs Added'])].sort_values('occurred_at')
            
            # Mark timepoints near events
            plate_data = integrated[plate_mask].copy()
            
            # Mark media change events
            for _, event in media_events.iterrows():
                event_time = event['occurred_at']
                
                # Find oxygen measurements within ±30 minutes of event
                time_diff = abs((plate_data['timestamp'] - event_time).dt.total_seconds())
                event_mask = time_diff <= 1800  # 30 minutes
                
                integrated.loc[plate_mask & event_mask, 'has_media_change'] = True
            
            # Mark dosing events
            for _, event in dosing_events.iterrows():
                event_time = event['occurred_at']
                
                # Find oxygen measurements within ±30 minutes of event
                time_diff = abs((plate_data['timestamp'] - event_time).dt.total_seconds())
                event_mask = time_diff <= 1800  # 30 minutes
                
                integrated.loc[plate_mask & event_mask, 'has_dosing_event'] = True
            
            # Calculate time-based features
            if len(media_events) > 0:
                media_times = media_events['occurred_at'].values
                
                for i, (idx, row) in enumerate(plate_data.iterrows()):
                    timestamp = row['timestamp']
                    
                    # Hours since last media change
                    past_media = media_times[media_times <= timestamp]
                    if len(past_media) > 0:
                        last_media = pd.to_datetime(past_media[-1])
                        hours_since = (timestamp - last_media).total_seconds() / 3600
                        integrated.loc[idx, 'hours_since_last_media_change'] = hours_since
                        integrated.loc[idx, 'media_change_number'] = len(past_media)
                    
                    # Time to next media change
                    future_media = media_times[media_times > timestamp]
                    if len(future_media) > 0:
                        next_media = pd.to_datetime(future_media[0])
                        hours_until = (next_media - timestamp).total_seconds() / 3600
                        integrated.loc[idx, 'time_to_next_media_change'] = hours_until
            
            # Hours since dosing
            if len(dosing_events) > 0:
                dosing_start = dosing_events['occurred_at'].min()
                
                for idx, row in plate_data.iterrows():
                    timestamp = row['timestamp']
                    if timestamp >= dosing_start:
                        hours_since = (timestamp - dosing_start).total_seconds() / 3600
                        integrated.loc[idx, 'hours_since_dosing'] = hours_since
        
        self.integrated_data = integrated
        
        # Summary statistics
        n_with_media = integrated['has_media_change'].sum()
        n_with_dosing = integrated['has_dosing_event'].sum()
        
        print(f"  Integration complete:")
        print(f"    Timepoints with media change: {n_with_media:,}")
        print(f"    Timepoints with dosing event: {n_with_dosing:,}")
        print(f"    Plates with event data: {integrated['hours_since_dosing'].notna().groupby(integrated['plate_id']).any().sum()}")
        
        return integrated
    
    def get_event_summary(self):
        """Get summary of events by plate."""
        if self.event_data is None:
            self.load_event_data()
            
        if self.event_data is None:
            return None
        
        summary = []
        for plate_id in self.event_data['plate_id'].unique():
            plate_events = self.event_data[self.event_data['plate_id'] == plate_id]
            
            # Count different event types
            event_counts = plate_events['title'].value_counts().to_dict()
            
            # Get timing
            dosing_events = plate_events[plate_events['title'].isin(['Drugs Start', 'Drugs Added'])]
            media_events = plate_events[plate_events['title'] == 'Medium Change']
            
            summary.append({
                'plate_id': plate_id,
                'total_events': len(plate_events),
                'media_changes': event_counts.get('Medium Change', 0),
                'dosing_events': len(dosing_events),
                'data_exclusions': event_counts.get('Data Exclusion', 0),
                'first_dosing': dosing_events['occurred_at'].min() if len(dosing_events) > 0 else None,
                'first_media_change': media_events['occurred_at'].min() if len(media_events) > 0 else None,
                'last_event': plate_events['occurred_at'].max()
            })
        
        return pd.DataFrame(summary)
    
    def get_baseline_periods(self, baseline_hours=48):
        """Identify baseline periods before dosing for each well."""
        if self.integrated_data is None:
            self.integrate_events_with_oxygen()
        
        baselines = []
        
        for well_id in self.integrated_data['well_id'].unique():
            well_data = self.integrated_data[self.integrated_data['well_id'] == well_id].sort_values('timestamp')
            
            # Find dosing time for this plate
            plate_id = well_data['plate_id'].iloc[0]
            dosing_time = well_data[well_data['hours_since_dosing'].notna()]['timestamp'].min()
            
            if pd.isna(dosing_time):
                # No dosing event, use first N hours
                baseline_data = well_data[well_data['elapsed_hours'] <= baseline_hours]
            else:
                # Use data before dosing
                baseline_data = well_data[well_data['timestamp'] < dosing_time]
                
                # Limit to last N hours before dosing
                if len(baseline_data) > 0:
                    cutoff_time = dosing_time - pd.Timedelta(hours=baseline_hours)
                    baseline_data = baseline_data[baseline_data['timestamp'] >= cutoff_time]
            
            if len(baseline_data) >= 10:  # Minimum 10 timepoints
                baselines.append({
                    'well_id': well_id,
                    'plate_id': plate_id,
                    'drug_name': well_data['drug_name'].iloc[0],
                    'concentration': well_data['concentration'].iloc[0],
                    'is_control': well_data['is_control'].iloc[0],
                    'baseline_start': baseline_data['timestamp'].min(),
                    'baseline_end': baseline_data['timestamp'].max(),
                    'baseline_duration_hours': (baseline_data['timestamp'].max() - baseline_data['timestamp'].min()).total_seconds() / 3600,
                    'baseline_n_points': len(baseline_data),
                    'baseline_mean_o2': baseline_data['value'].mean(),
                    'baseline_std_o2': baseline_data['value'].std(),
                    'baseline_cv': baseline_data['value'].std() / baseline_data['value'].mean() if baseline_data['value'].mean() > 0 else np.nan
                })
        
        return pd.DataFrame(baselines)
    
    def save_integrated_data(self, output_path=None):
        """Save integrated data for future use."""
        if self.integrated_data is None:
            self.integrate_events_with_oxygen()
        
        if output_path is None:
            output_path = self.data_dir / "integrated_oxygen_events.parquet"
        
        self.integrated_data.to_parquet(output_path, index=False)
        print(f"Integrated data saved to: {output_path}")
        
        return output_path


def main():
    """Demonstration of integrated data loading."""
    print("=== Integrated Data Loader Demo ===\n")
    
    # Initialize loader
    loader = IntegratedDataLoader()
    
    # Load and integrate data
    integrated_data = loader.integrate_events_with_oxygen()
    
    # Get event summary
    event_summary = loader.get_event_summary()
    if event_summary is not None:
        print(f"\n=== Event Summary by Plate ===")
        print(f"Plates with events: {len(event_summary)}")
        print(f"Mean media changes per plate: {event_summary['media_changes'].mean():.1f}")
        print(f"Plates with dosing events: {(event_summary['dosing_events'] > 0).sum()}")
    
    # Get baseline periods
    baseline_summary = loader.get_baseline_periods()
    print(f"\n=== Baseline Period Analysis ===")
    print(f"Wells with valid baselines: {len(baseline_summary)}")
    print(f"Mean baseline duration: {baseline_summary['baseline_duration_hours'].mean():.1f} hours")
    print(f"Mean baseline CV: {baseline_summary['baseline_cv'].mean():.3f}")
    
    # Save integrated data
    output_path = loader.save_integrated_data()
    
    print(f"\n✅ Integration complete!")
    print(f"📊 Integrated data: {len(integrated_data):,} measurements")
    print(f"🔄 Event-aware features ready for extraction")
    print(f"💾 Data saved to: {output_path}")
    
    return loader


if __name__ == "__main__":
    main()