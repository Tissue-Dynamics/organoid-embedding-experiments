#!/usr/bin/env python3
"""
Step 3: Media Change Event Detection and Event-Aware Features
Detect media change events and extract event-indexed features.
"""

import os
import sys
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import from Step 2
from step2_multiscale_features import MultiScaleFeatureExtractor

# Load environment variables
load_dotenv()


class MediaChangeDetector:
    """Detect and characterize media change events in time series data."""
    
    def __init__(self, variance_threshold=2.0, peak_prominence=0.5):
        """
        Initialize media change detector.
        
        Args:
            variance_threshold: Multiplier for baseline variance to detect events
            peak_prominence: Minimum prominence for peak detection
        """
        self.variance_threshold = variance_threshold
        self.peak_prominence = peak_prominence
        
    def detect_media_changes_from_events(self, events_df, plate_id):
        """Get media change times from event data."""
        plate_events = events_df[
            (events_df['plate_id'] == plate_id) & 
            (events_df['title'] == 'Medium Change')
        ].copy()
        
        if len(plate_events) == 0:
            return []
        
        # Sort by occurred_at
        plate_events = plate_events.sort_values('occurred_at')
        
        media_changes = []
        for _, event in plate_events.iterrows():
            media_changes.append({
                'event_time': event['occurred_at'],
                'event_id': event['id'],
                'description': event.get('description', '')
            })
        
        return media_changes
    
    def detect_media_changes_from_variance(self, time_series_data):
        """Detect media changes using variance-based method."""
        if len(time_series_data) < 20:
            return []
        
        # Calculate rolling variance
        window_size = 6  # 6 hour window
        o2_values = time_series_data['o2_percent'].values
        elapsed_hours = time_series_data['elapsed_hours'].values
        
        # Simple rolling variance calculation
        variances = []
        variance_times = []
        
        for i in range(len(time_series_data)):
            # Find points within window
            current_time = elapsed_hours[i]
            window_mask = (elapsed_hours >= current_time - window_size/2) & \
                         (elapsed_hours <= current_time + window_size/2)
            
            if np.sum(window_mask) >= 3:
                window_variance = np.var(o2_values[window_mask])
                variances.append(window_variance)
                variance_times.append(current_time)
        
        if len(variances) == 0:
            return []
        
        variances = np.array(variances)
        variance_times = np.array(variance_times)
        
        # Calculate baseline variance (first 48h)
        baseline_mask = variance_times <= 48
        if np.sum(baseline_mask) < 5:
            baseline_variance = np.median(variances)
        else:
            baseline_variance = np.median(variances[baseline_mask])
        
        # Find peaks in variance
        threshold = baseline_variance * self.variance_threshold
        peaks, properties = find_peaks(
            variances, 
            height=threshold,
            prominence=self.peak_prominence * baseline_variance,
            distance=10  # Minimum 10 hours between events
        )
        
        media_changes = []
        for peak_idx in peaks:
            media_changes.append({
                'detected_time_hours': variance_times[peak_idx],
                'variance_ratio': variances[peak_idx] / baseline_variance,
                'detection_method': 'variance'
            })
        
        return media_changes
    
    def characterize_spike(self, time_series_data, event_time_hours, window_before=6, window_after=12):
        """Characterize a media change spike."""
        o2_values = time_series_data['o2_percent'].values
        elapsed_hours = time_series_data['elapsed_hours'].values
        
        # Define analysis windows
        pre_event_mask = (elapsed_hours >= event_time_hours - window_before) & \
                        (elapsed_hours < event_time_hours)
        event_window_mask = (elapsed_hours >= event_time_hours - 1) & \
                           (elapsed_hours <= event_time_hours + window_after)
        
        if np.sum(pre_event_mask) < 3 or np.sum(event_window_mask) < 5:
            return None
        
        # Pre-spike baseline
        pre_spike_o2 = o2_values[pre_event_mask]
        pre_spike_mean = np.mean(pre_spike_o2)
        pre_spike_std = np.std(pre_spike_o2)
        
        # Event window analysis
        event_o2 = o2_values[event_window_mask]
        event_times = elapsed_hours[event_window_mask]
        
        # Find peak
        peak_idx = np.argmax(np.abs(event_o2 - pre_spike_mean))
        peak_value = event_o2[peak_idx]
        peak_time = event_times[peak_idx]
        peak_height = peak_value - pre_spike_mean
        
        # Recovery analysis
        recovery_threshold = pre_spike_mean + 0.1 * peak_height  # 90% recovery
        
        # Find recovery time
        recovery_time = None
        for i in range(peak_idx + 1, len(event_o2)):
            if abs(event_o2[i] - pre_spike_mean) <= abs(recovery_threshold - pre_spike_mean):
                recovery_time = event_times[i] - event_time_hours
                break
        
        # Post-spike baseline (if recovery found)
        post_spike_mean = None
        post_spike_std = None
        
        if recovery_time is not None:
            post_event_mask = (elapsed_hours >= event_time_hours + recovery_time + 1) & \
                             (elapsed_hours <= event_time_hours + recovery_time + window_before)
            if np.sum(post_event_mask) >= 3:
                post_spike_o2 = o2_values[post_event_mask]
                post_spike_mean = np.mean(post_spike_o2)
                post_spike_std = np.std(post_spike_o2)
        
        return {
            'pre_spike_mean': pre_spike_mean,
            'pre_spike_std': pre_spike_std,
            'peak_height': peak_height,
            'peak_time_relative': peak_time - event_time_hours,
            'peak_absolute_value': peak_value,
            'recovery_time': recovery_time,
            'post_spike_mean': post_spike_mean,
            'post_spike_std': post_spike_std,
            'baseline_shift': post_spike_mean - pre_spike_mean if post_spike_mean else None
        }


class EventAwareFeatureExtractor:
    """Extract features from inter-event periods."""
    
    def __init__(self, feature_extractor=None):
        """Initialize with optional feature extractor from Step 2."""
        self.feature_extractor = feature_extractor or MultiScaleFeatureExtractor(
            window_sizes=[24, 48],  # Smaller windows for inter-event periods
            overlap_ratio=0.5
        )
    
    def extract_inter_event_features(self, time_series_data, media_changes, well_id):
        """Extract features from periods between media changes."""
        if len(media_changes) == 0:
            # No media changes - use full time series
            return self._extract_full_series_features(time_series_data, well_id)
        
        elapsed_hours = time_series_data['elapsed_hours'].values
        
        # Sort media changes by time
        sorted_changes = sorted(media_changes, key=lambda x: x.get('event_time_hours', x.get('detected_time_hours', 0)))
        
        all_features = []
        
        # Pre-first event period
        first_event_time = sorted_changes[0].get('event_time_hours', sorted_changes[0].get('detected_time_hours'))
        pre_first_mask = elapsed_hours < first_event_time - 3  # 3h buffer before event
        
        if np.sum(pre_first_mask) > 10:
            period_data = time_series_data[pre_first_mask].copy()
            period_features = self._extract_period_features(
                period_data, well_id, 
                period_type='pre_first_media_change',
                period_number=0
            )
            all_features.extend(period_features)
        
        # Inter-event periods
        for i in range(len(sorted_changes) - 1):
            current_event = sorted_changes[i]
            next_event = sorted_changes[i + 1]
            
            current_time = current_event.get('event_time_hours', current_event.get('detected_time_hours'))
            next_time = next_event.get('event_time_hours', next_event.get('detected_time_hours'))
            
            # Extract period between events (with buffers)
            period_mask = (elapsed_hours >= current_time + 6) & \
                         (elapsed_hours < next_time - 3)
            
            if np.sum(period_mask) > 10:
                period_data = time_series_data[period_mask].copy()
                period_features = self._extract_period_features(
                    period_data, well_id,
                    period_type='inter_media_change',
                    period_number=i + 1
                )
                all_features.extend(period_features)
        
        # Post-last event period
        if len(sorted_changes) > 0:
            last_event_time = sorted_changes[-1].get('event_time_hours', sorted_changes[-1].get('detected_time_hours'))
            post_last_mask = elapsed_hours >= last_event_time + 6  # 6h buffer after event
            
            if np.sum(post_last_mask) > 10:
                period_data = time_series_data[post_last_mask].copy()
                period_features = self._extract_period_features(
                    period_data, well_id,
                    period_type='post_last_media_change',
                    period_number=len(sorted_changes)
                )
                all_features.extend(period_features)
        
        return all_features
    
    def _extract_period_features(self, period_data, well_id, period_type, period_number):
        """Extract features from a specific period."""
        # Reset elapsed hours for period
        min_time = period_data['elapsed_hours'].min()
        period_data['elapsed_hours'] = period_data['elapsed_hours'] - min_time
        
        # Extract features using Step 2 extractor
        features = self.feature_extractor.process_well(period_data, well_id)
        
        # Add period metadata
        for feature in features:
            feature['period_type'] = period_type
            feature['period_number'] = period_number
            feature['period_duration'] = period_data['elapsed_hours'].max()
            feature['period_start_time'] = min_time
        
        return features
    
    def _extract_full_series_features(self, time_series_data, well_id):
        """Extract features from full series when no media changes detected."""
        features = self.feature_extractor.process_well(time_series_data, well_id)
        
        for feature in features:
            feature['period_type'] = 'full_series'
            feature['period_number'] = 0
            feature['period_duration'] = time_series_data['elapsed_hours'].max()
            feature['period_start_time'] = 0
        
        return features


def load_event_data():
    """Load event data from parquet file."""
    event_path = project_root / "data" / "raw" / "event_data.parquet"
    
    if not event_path.exists():
        print("Warning: Event data not found. Using variance-based detection only.")
        return None
    
    events_df = pd.read_parquet(event_path)
    events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
    
    print(f"Loaded {len(events_df)} events")
    media_changes = events_df[events_df['title'] == 'Medium Change']
    print(f"  Media change events: {len(media_changes)}")
    print(f"  Plates with media changes: {media_changes['plate_id'].nunique()}")
    
    return events_df


def load_time_series_for_plate(plate_id):
    """Load all time series data for a plate."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    query = f"""
    SELECT 
        (plate_id::text || '_' || well_number::text) as well_id,
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND is_excluded = false
    ORDER BY well_id, timestamp
    """
    
    data = conn.execute(query).fetchdf()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate elapsed hours
    min_timestamp = data.groupby('well_id')['timestamp'].transform('min')
    data['elapsed_hours'] = (data['timestamp'] - min_timestamp).dt.total_seconds() / 3600
    
    conn.close()
    return data


def process_plate_events(plate_id, events_df=None):
    """Process all media change events for a plate."""
    print(f"\nProcessing plate: {plate_id}")
    
    # Load time series data
    plate_data = load_time_series_for_plate(plate_id)
    
    if len(plate_data) == 0:
        print(f"  No data found for plate {plate_id}")
        return []
    
    wells = plate_data['well_id'].unique()
    print(f"  Found {len(wells)} wells")
    
    # Initialize detectors
    detector = MediaChangeDetector()
    feature_extractor = EventAwareFeatureExtractor()
    
    # Get media changes from event data if available
    event_media_changes = []
    if events_df is not None:
        event_media_changes = detector.detect_media_changes_from_events(events_df, plate_id)
        if event_media_changes:
            # Convert to elapsed hours
            plate_start = plate_data['timestamp'].min()
            for mc in event_media_changes:
                mc['event_time_hours'] = (mc['event_time'] - plate_start).total_seconds() / 3600
    
    print(f"  Found {len(event_media_changes)} media changes from event data")
    
    all_results = []
    
    # Process each well
    for well_id in wells[:5]:  # Sample 5 wells for testing
        well_data = plate_data[plate_data['well_id'] == well_id].copy()
        
        # Detect media changes (use event data if available, else variance method)
        if event_media_changes:
            media_changes = event_media_changes
        else:
            media_changes = detector.detect_media_changes_from_variance(well_data)
        
        # Characterize each spike
        spike_features = []
        for i, mc in enumerate(media_changes):
            event_time = mc.get('event_time_hours', mc.get('detected_time_hours'))
            spike_char = detector.characterize_spike(well_data, event_time)
            
            if spike_char:
                spike_char['event_number'] = i + 1
                spike_char['event_time_hours'] = event_time
                spike_features.append(spike_char)
        
        # Extract inter-event features
        inter_event_features = feature_extractor.extract_inter_event_features(
            well_data, media_changes, well_id
        )
        
        # Combine results
        result = {
            'plate_id': plate_id,
            'well_id': well_id,
            'n_media_changes': len(media_changes),
            'n_spike_features': len(spike_features),
            'n_inter_event_features': len(inter_event_features),
            'spike_features': spike_features,
            'inter_event_features': inter_event_features
        }
        
        all_results.append(result)
    
    return all_results


def main():
    """Main Step 3 pipeline for event detection and characterization."""
    print("=== Step 3: Media Change Event Detection ===\n")
    
    # Load event data
    events_df = load_event_data()
    
    # Load Step 1 results to get plates
    step1_path = project_root / "results" / "data" / "step1_quality_assessment_all_plates.parquet"
    
    if not step1_path.exists():
        print("Step 1 data not found. Please run Step 1 first.")
        return
    
    step1_data = pd.read_parquet(step1_path)
    
    # Get sample plates
    sample_plates = step1_data['plate_id'].unique()[:3]  # Process 3 plates for testing
    
    print(f"Processing {len(sample_plates)} sample plates...")
    
    all_plate_results = []
    
    for plate_id in sample_plates:
        plate_results = process_plate_events(plate_id, events_df)
        all_plate_results.extend(plate_results)
    
    # Summarize results
    print(f"\n=== Event Detection Results ===")
    print(f"Total wells processed: {len(all_plate_results)}")
    
    total_media_changes = sum(r['n_media_changes'] for r in all_plate_results)
    total_spike_features = sum(r['n_spike_features'] for r in all_plate_results)
    total_inter_features = sum(r['n_inter_event_features'] for r in all_plate_results)
    
    print(f"Total media changes detected: {total_media_changes}")
    print(f"Total spike characterizations: {total_spike_features}")
    print(f"Total inter-event feature records: {total_inter_features}")
    
    # Extract spike features for saving
    spike_data = []
    for result in all_plate_results:
        for spike in result['spike_features']:
            spike_record = {
                'plate_id': result['plate_id'],
                'well_id': result['well_id'],
                **spike
            }
            spike_data.append(spike_record)
    
    if spike_data:
        spike_df = pd.DataFrame(spike_data)
        
        print(f"\n=== Spike Characterization Summary ===")
        print(f"Mean peak height: {spike_df['peak_height'].mean():.2f} ± {spike_df['peak_height'].std():.2f}")
        print(f"Mean recovery time: {spike_df['recovery_time'].dropna().mean():.2f} ± {spike_df['recovery_time'].dropna().std():.2f} hours")
        
        # Save results
        output_dir = project_root / "results" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        spike_path = output_dir / "step3_spike_features.parquet"
        spike_df.to_parquet(spike_path, index=False)
        
        csv_path = output_dir / "step3_spike_features.csv"
        spike_df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Step 3 Complete!")
        print(f"📊 Spike features saved to: {spike_path}")
        print(f"📄 CSV saved to: {csv_path}")
    
    # Extract inter-event features for saving
    inter_event_data = []
    for result in all_plate_results:
        inter_event_data.extend(result['inter_event_features'])
    
    if inter_event_data:
        inter_event_df = pd.DataFrame(inter_event_data)
        
        inter_path = output_dir / "step3_inter_event_features.parquet"
        inter_event_df.to_parquet(inter_path, index=False)
        
        print(f"📊 Inter-event features saved to: {inter_path}")
        print(f"🔍 Feature records by period type:")
        print(inter_event_df['period_type'].value_counts())
    
    print(f"\n🎯 Next: Step 4 - Dose-Response Normalization")
    
    return spike_df if spike_data else None, inter_event_df if inter_event_data else None


if __name__ == "__main__":
    main()