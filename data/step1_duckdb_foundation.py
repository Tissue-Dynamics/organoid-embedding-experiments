#!/usr/bin/env python3
"""
Step 1: Data Pipeline Foundation with DuckDB
Event-relative baseline detection and comprehensive quality flags.
"""

import os
import sys
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()


class DuckDBDataLoader:
    """Enhanced data loader using DuckDB for direct database queries."""
    
    def __init__(self, sample_plates=10):
        self.sample_plates = sample_plates
        self.conn = None
        self.postgres_string = None
        self._setup_connection()
        
    def _setup_connection(self):
        """Setup DuckDB connection with postgres extension."""
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL postgres;")
        self.conn.execute("LOAD postgres;")
        
        # Parse connection string
        parsed = urlparse(database_url)
        self.postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    def load_event_data(self):
        """Load event data directly from database."""
        print("Loading event data...")
        
        events_query = f"""
        SELECT 
            plate_id::text as plate_id,
            title,
            occurred_at,
            description
        FROM postgres_scan('{self.postgres_string}', 'public', 'event_table')
        WHERE is_excluded = false
        ORDER BY plate_id, occurred_at
        """
        
        events_df = self.conn.execute(events_query).fetchdf()
        events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
        
        print(f"  Loaded {len(events_df):,} events from {events_df['plate_id'].nunique()} plates")
        return events_df
    
    def get_dosing_times(self, events_df):
        """Extract dosing times for event-relative baseline detection."""
        dosing_events = events_df[events_df['title'].isin(['Drugs Start', 'Drugs Added'])]
        dosing_times = dosing_events.groupby('plate_id')['occurred_at'].min().to_dict()
        
        print(f"  Found dosing times for {len(dosing_times)} plates")
        return dosing_times
    
    def load_sample_plates(self):
        """Load list of sample plates for processing."""
        print(f"Getting sample of {self.sample_plates} plates...")
        
        query = f"""
        SELECT DISTINCT plate_id::text as plate_id
        FROM postgres_scan('{self.postgres_string}', 'public', 'processed_data')
        WHERE is_excluded = false
        LIMIT {self.sample_plates}
        """
        
        plates_df = self.conn.execute(query).fetchdf()
        plate_ids = plates_df['plate_id'].tolist()
        
        print(f"  Selected {len(plate_ids)} plates for processing")
        return plate_ids
    
    def load_plate_data(self, plate_id):
        """Load time series data for a specific plate."""
        query = f"""
        SELECT 
            (plate_id::text || '_' || well_number::text) as well_id,
            timestamp,
            median_o2 as o2_percent,
            is_excluded
        FROM postgres_scan('{self.postgres_string}', 'public', 'processed_data')
        WHERE plate_id::text = '{plate_id}' 
        AND is_excluded = false
        ORDER BY well_number, timestamp
        """
        
        data = self.conn.execute(query).fetchdf()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Calculate elapsed hours in pandas (more efficient)
        min_timestamp = data['timestamp'].min()
        data['elapsed_hours'] = (data['timestamp'] - min_timestamp).dt.total_seconds() / 3600
        
        return data
    
    def load_well_metadata(self, plate_id):
        """Load well metadata for mapping control wells."""
        query = f"""
        SELECT 
            (plate_id::text || '_' || well_number::text) as well_id,
            drug as drug_name,
            concentration as concentration_um,
            (drug = 'DMSO' OR concentration = 0) as is_control
        FROM postgres_scan('{self.postgres_string}', 'public', 'well_map_data')
        WHERE plate_id::text = '{plate_id}'
        """
        
        metadata = self.conn.execute(query).fetchdf()
        return metadata


def detect_quality_flags(well_data, baseline_data, baseline_type='unknown'):
    """
    Detect quality flags for a single well with relaxed CV thresholds.
    Relaxed thresholds for organoid data: 0.5 for noise, 0.2 for baseline.
    """
    flags = {
        'low_points': len(well_data) < 100,  # Minimum data points
        'high_noise': False,
        'sensor_drift': False,
        'baseline_unstable': False,
        'replicate_discord': False,  # Will be calculated at plate level
        'media_change_outlier': False,  # Will be calculated with event data
        'baseline_type': baseline_type,
        'baseline_duration_hours': 0 if len(baseline_data) == 0 else baseline_data['elapsed_hours'].max() - baseline_data['elapsed_hours'].min()
    }
    
    if len(well_data) == 0:
        return flags
    
    # High noise detection (relaxed threshold)
    cv = well_data['o2_percent'].std() / well_data['o2_percent'].mean()
    flags['high_noise'] = cv > 0.5  # Relaxed from 0.3
    
    # Sensor drift detection (trend in residuals)
    if len(well_data) > 50:
        # Simple linear trend detection
        x = np.arange(len(well_data))
        y = well_data['o2_percent'].values
        correlation = np.corrcoef(x, y)[0, 1]
        flags['sensor_drift'] = abs(correlation) > 0.7  # Strong trend
    
    # Baseline stability (relaxed threshold)
    if len(baseline_data) > 10:
        baseline_cv = baseline_data['o2_percent'].std() / baseline_data['o2_percent'].mean()
        flags['baseline_unstable'] = baseline_cv > 0.2  # Relaxed from 0.1
    
    return flags


def process_plate(loader, plate_id, dosing_times):
    """Process a single plate with event-relative baseline detection."""
    print(f"\nProcessing plate: {plate_id}")
    
    # Load plate data
    plate_data = loader.load_plate_data(plate_id)
    well_metadata = loader.load_well_metadata(plate_id)
    
    if len(plate_data) == 0:
        print(f"  No data found for plate {plate_id}")
        return []
    
    print(f"  Loaded {len(plate_data):,} data points from {plate_data['well_id'].nunique()} wells")
    
    # Process each well
    well_results = []
    
    for well_id in plate_data['well_id'].unique():
        well_data = plate_data[plate_data['well_id'] == well_id].copy()
        well_meta = well_metadata[well_metadata['well_id'] == well_id]
        
        if len(well_data) == 0:
            continue
        
        # Determine baseline period based on actual dosing time
        if plate_id in dosing_times:
            # Use pre-dosing period as baseline
            dosing_time = dosing_times[plate_id]
            baseline_data = well_data[well_data['timestamp'] < dosing_time]
            baseline_type = 'pre_dosing'
        else:
            # Fallback: use first 48h if no dosing event
            baseline_data = well_data[well_data['elapsed_hours'] <= 48]
            baseline_type = 'first_48h'
        
        # Quality flag detection
        flags = detect_quality_flags(well_data, baseline_data, baseline_type)
        
        # Add well metadata
        drug_name = well_meta['drug_name'].iloc[0] if len(well_meta) > 0 else 'Unknown'
        concentration = well_meta['concentration_um'].iloc[0] if len(well_meta) > 0 else 0
        is_control = well_meta['is_control'].iloc[0] if len(well_meta) > 0 else False
        
        well_result = {
            'plate_id': plate_id,
            'well_id': well_id,
            'drug_name': drug_name,
            'concentration_um': concentration,
            'is_control': is_control,
            'total_points': len(well_data),
            'baseline_points': len(baseline_data),
            'baseline_duration_hours': flags['baseline_duration_hours'],
            'baseline_type': flags['baseline_type'],
            'low_points': flags['low_points'],
            'high_noise': flags['high_noise'],
            'sensor_drift': flags['sensor_drift'],
            'baseline_unstable': flags['baseline_unstable'],
            'mean_o2': well_data['o2_percent'].mean(),
            'std_o2': well_data['o2_percent'].std(),
            'cv_o2': well_data['o2_percent'].std() / well_data['o2_percent'].mean(),
            'baseline_mean': baseline_data['o2_percent'].mean() if len(baseline_data) > 0 else np.nan,
            'baseline_std': baseline_data['o2_percent'].std() if len(baseline_data) > 0 else np.nan
        }
        
        well_results.append(well_result)
    
    print(f"  Processed {len(well_results)} wells")
    return well_results


def analyze_results(results_df):
    """Analyze the quality assessment results."""
    print("\n=== Quality Assessment Results ===")
    print(f"Total wells processed: {len(results_df):,}")
    print(f"Total plates: {results_df['plate_id'].nunique()}")
    
    # Quality flag statistics
    print("\nQuality Flags:")
    flag_cols = ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable']
    for flag in flag_cols:
        count = results_df[flag].sum()
        pct = 100 * count / len(results_df)
        print(f"  {flag}: {count:,} wells ({pct:.1f}%)")
    
    # Baseline type distribution
    print("\nBaseline Type Distribution:")
    baseline_dist = results_df['baseline_type'].value_counts()
    for baseline_type, count in baseline_dist.items():
        pct = 100 * count / len(results_df)
        print(f"  {baseline_type}: {count:,} wells ({pct:.1f}%)")
    
    # Baseline duration statistics
    baseline_duration = results_df['baseline_duration_hours'].dropna()
    if len(baseline_duration) > 0:
        print(f"\nBaseline Duration:")
        print(f"  Mean: {baseline_duration.mean():.1f} ± {baseline_duration.std():.1f} hours")
        print(f"  Range: {baseline_duration.min():.1f} - {baseline_duration.max():.1f} hours")
    
    # Control vs treatment wells
    control_wells = results_df[results_df['is_control'] == True]
    treatment_wells = results_df[results_df['is_control'] == False]
    
    print(f"\nWell Types:")
    print(f"  Control wells: {len(control_wells):,}")
    print(f"  Treatment wells: {len(treatment_wells):,}")
    
    if len(control_wells) > 0 and len(treatment_wells) > 0:
        print(f"\nControl vs Treatment Quality:")
        for flag in flag_cols:
            control_pct = 100 * control_wells[flag].sum() / len(control_wells)
            treatment_pct = 100 * treatment_wells[flag].sum() / len(treatment_wells)
            print(f"  {flag} - Control: {control_pct:.1f}%, Treatment: {treatment_pct:.1f}%")


def main():
    """Main Step 1 pipeline with DuckDB and event-relative baselines."""
    print("=== Step 1: Data Pipeline Foundation (DuckDB Version) ===\n")
    
    # Initialize data loader
    loader = DuckDBDataLoader(sample_plates=10)
    
    # Load event data for baseline detection
    events_df = loader.load_event_data()
    dosing_times = loader.get_dosing_times(events_df)
    
    # Load sample plates
    plate_ids = loader.load_sample_plates()
    
    # Process each plate
    all_results = []
    for plate_id in plate_ids:
        try:
            plate_results = process_plate(loader, plate_id, dosing_times)
            all_results.extend(plate_results)
        except Exception as e:
            print(f"  Error processing plate {plate_id}: {e}")
            continue
    
    if len(all_results) == 0:
        print("❌ No wells processed successfully")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Analyze results
    analyze_results(results_df)
    
    # Save results
    output_dir = project_root / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "step1_quality_assessment_duckdb.parquet"
    results_df.to_parquet(output_path, index=False)
    
    csv_path = output_dir / "step1_quality_assessment_duckdb.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Step 1 Complete!")
    print(f"📊 Results saved to: {output_path}")
    print(f"📄 CSV saved to: {csv_path}")
    print(f"🔍 Key improvements:")
    print(f"  - Event-relative baseline detection")
    print(f"  - Relaxed CV thresholds for organoid data")
    print(f"  - Direct DuckDB database queries")
    
    return results_df


if __name__ == "__main__":
    main()