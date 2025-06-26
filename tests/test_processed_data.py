#!/usr/bin/env python3
"""
Comprehensive tests for processed_data table via DataLoader.

Tests include:
- Data quality validation
- Temporal consistency
- Exclusion handling
- Cycle data integrity
- O2 value ranges
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader


# Data quality thresholds based on actual database content
MIN_TOTAL_RECORDS = 3_000_000      # Actually have ~3.15M
MIN_UNIQUE_PLATES = 30             # Actually have 33
MIN_WELLS_PER_PLATE = 350          # 384-well plates, some excluded
MIN_CYCLES_PER_PLATE = 300         # Experiments run for days
MAX_EXCLUSION_RATE = 5.0           # Less than 5% excluded data
MIN_EXPERIMENT_DURATION_HOURS = 300 # At least 12.5 days
MAX_EXPERIMENT_DURATION_HOURS = 1200 # At most 50 days

# O2 value thresholds
MIN_REASONABLE_O2 = -50            # Some negative values are ok
MAX_REASONABLE_O2 = 150            # Above 150% is suspicious
TYPICAL_O2_RANGE = (0, 100)        # Most values should be here

# Expected columns
PROCESSED_DATA_COLUMNS = {'plate_id', 'well_id', 'well_number', 'drug', 
                         'concentration', 'elapsed_hours', 'o2', 'timestamp'}
SUMMARY_COLUMNS = {'plate_id', 'record_count', 'unique_wells', 'start_time', 
                  'end_time', 'duration_hours', 'total_cycles', 'min_o2', 
                  'max_o2', 'avg_o2', 'excluded_records', 'exclusion_rate'}


class TestProcessedDataValidation:
    """Test processed_data quality and completeness."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    def test_validate_processed_data(self, loader):
        """Test overall processed_data validation."""
        validation = loader.validate_processed_data()
        
        # Check total records
        assert validation['total_records'] >= MIN_TOTAL_RECORDS, \
            f"Expected at least {MIN_TOTAL_RECORDS:,} records, got {validation['total_records']:,}"
        
        # Check unique plates
        assert validation['unique_plates'] >= MIN_UNIQUE_PLATES, \
            f"Expected at least {MIN_UNIQUE_PLATES} plates, got {validation['unique_plates']}"
        
        # Check O2 ranges
        assert validation['min_o2'] >= MIN_REASONABLE_O2, \
            f"Minimum O2 {validation['min_o2']} is too low"
        assert validation['max_o2'] <= MAX_REASONABLE_O2, \
            f"Maximum O2 {validation['max_o2']} is too high"
        
        # Check exclusions
        exclusion_rate = (validation['excluded_count'] / validation['total_records']) * 100
        assert exclusion_rate <= MAX_EXCLUSION_RATE, \
            f"Exclusion rate {exclusion_rate:.2f}% exceeds maximum {MAX_EXCLUSION_RATE}%"
        
        # Check for null O2 values
        assert validation['null_o2_count'] == 0, \
            f"Found {validation['null_o2_count']} null O2 values"
        
        # Check extreme values
        extreme_low_rate = (validation['extreme_low_o2'] / validation['total_records']) * 100
        extreme_high_rate = (validation['extreme_high_o2'] / validation['total_records']) * 100
        assert extreme_low_rate < 0.1, f"Too many extreme low O2 values: {extreme_low_rate:.3f}%"
        assert extreme_high_rate < 0.1, f"Too many extreme high O2 values: {extreme_high_rate:.3f}%"
        
        # Print summary for debugging
        print(f"\nProcessed Data Validation Summary:")
        print(f"  Total records: {validation['total_records']:,}")
        print(f"  Unique plates: {validation['unique_plates']}")
        print(f"  Excluded records: {validation['excluded_count']:,} ({exclusion_rate:.2f}%)")
        print(f"  O2 range: {validation['min_o2']:.2f} to {validation['max_o2']:.2f}")
        print(f"  Date range: {validation['earliest_timestamp']} to {validation['latest_timestamp']}")
    
    def test_load_processed_data_summary(self, loader):
        """Test plate-level summary statistics."""
        summary = loader.load_processed_data_summary()
        
        # Check DataFrame structure
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty
        assert set(summary.columns) == SUMMARY_COLUMNS
        
        # Check plate count
        assert len(summary) >= MIN_UNIQUE_PLATES, \
            f"Expected at least {MIN_UNIQUE_PLATES} plates, got {len(summary)}"
        
        # Validate each plate
        for idx, row in summary.iterrows():
            # Check well count
            assert row['unique_wells'] >= MIN_WELLS_PER_PLATE, \
                f"Plate {row['plate_id']} has only {row['unique_wells']} wells"
            
            # Check experiment duration
            assert row['duration_hours'] >= MIN_EXPERIMENT_DURATION_HOURS, \
                f"Plate {row['plate_id']} duration {row['duration_hours']}h is too short"
            assert row['duration_hours'] <= MAX_EXPERIMENT_DURATION_HOURS, \
                f"Plate {row['plate_id']} duration {row['duration_hours']}h is too long"
            
            # Check cycles
            assert row['total_cycles'] >= MIN_CYCLES_PER_PLATE, \
                f"Plate {row['plate_id']} has only {row['total_cycles']} cycles"
            
            # Check O2 ranges
            assert row['min_o2'] >= MIN_REASONABLE_O2, \
                f"Plate {row['plate_id']} min O2 {row['min_o2']} is too low"
            assert row['max_o2'] <= MAX_REASONABLE_O2, \
                f"Plate {row['plate_id']} max O2 {row['max_o2']} is too high"
            
            # Check exclusion rate
            assert row['exclusion_rate'] <= MAX_EXCLUSION_RATE, \
                f"Plate {row['plate_id']} exclusion rate {row['exclusion_rate']}% is too high"
    
    def test_temporal_consistency(self, loader):
        """Test that timestamps are consistent and reasonable."""
        summary = loader.load_processed_data_summary()
        
        # Check that experiments don't overlap too much
        sorted_summary = summary.sort_values('start_time')
        
        for i in range(len(sorted_summary) - 1):
            current = sorted_summary.iloc[i]
            next_plate = sorted_summary.iloc[i + 1]
            
            # Some overlap is OK (different instruments)
            overlap = (current['end_time'] - next_plate['start_time']).total_seconds() / 3600
            if overlap > 0:
                assert overlap < 1000, \
                    f"Excessive overlap between plates: {overlap:.1f} hours"
        
        # Check all experiments are within reasonable date range
        earliest = pd.to_datetime(summary['start_time'].min())
        latest = pd.to_datetime(summary['end_time'].max())
        
        assert earliest >= pd.Timestamp('2024-01-01'), \
            f"Data starts too early: {earliest}"
        assert latest <= pd.Timestamp.now() + pd.Timedelta(days=1), \
            f"Data extends into future: {latest}"


class TestExcludedData:
    """Test excluded data handling."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    def test_load_excluded_data(self, loader):
        """Test loading excluded data records."""
        # Get a plate with exclusions
        summary = loader.load_processed_data_summary()
        plates_with_exclusions = summary[summary['excluded_records'] > 0]['plate_id'].tolist()
        
        if not plates_with_exclusions:
            pytest.skip("No plates with excluded data found")
        
        # Load excluded data for first plate
        excluded = loader.load_excluded_data(plate_ids=[plates_with_exclusions[0]])
        
        assert not excluded.empty, "Should have excluded records"
        assert (excluded['exclusion_reason'].notna()).all(), \
            "All excluded records should have a reason"
        
        # Check exclusion reasons
        unique_reasons = excluded['exclusion_reason'].unique()
        print(f"\nExclusion reasons found: {unique_reasons}")
        
        # Validate excluded O2 values are reasonable
        assert excluded['median_o2'].notna().all(), \
            "Excluded records should still have O2 values"
    
    def test_exclusion_patterns(self, loader):
        """Test that exclusions follow expected patterns."""
        # Load all excluded data (limited)
        excluded = loader.load_excluded_data()
        
        if excluded.empty:
            pytest.skip("No excluded data found")
        
        # Group by reason
        reason_counts = excluded['exclusion_reason'].value_counts()
        print(f"\nExclusion reason distribution:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count:,} records")
        
        # Check that exclusions are not concentrated in specific wells
        well_exclusions = excluded.groupby(['plate_id', 'well_number']).size()
        max_exclusions_per_well = well_exclusions.max()
        
        assert max_exclusions_per_well < 1000, \
            f"Well has too many exclusions: {max_exclusions_per_well}"


class TestCycleData:
    """Test cycle-level data integrity."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    @pytest.fixture
    def sample_plate(self, loader):
        """Get a sample plate ID for testing."""
        summary = loader.load_processed_data_summary()
        # Get a plate with good data (low exclusion rate)
        good_plates = summary[summary['exclusion_rate'] < 1.0]
        return good_plates.iloc[0]['plate_id']
    
    def test_load_cycle_statistics(self, loader, sample_plate):
        """Test cycle-level statistics."""
        cycles = loader.load_cycle_statistics(sample_plate)
        
        assert not cycles.empty
        assert len(cycles) >= MIN_CYCLES_PER_PLATE
        
        # Check cycle numbering
        assert cycles['cycle_num'].min() >= 0
        assert (cycles['cycle_num'].diff().dropna() == 1).all(), \
            "Cycle numbers should be sequential"
        
        # Check wells measured per cycle
        assert (cycles['wells_measured'] >= MIN_WELLS_PER_PLATE).all(), \
            "Each cycle should measure most wells"
        
        # Check O2 statistics
        assert (cycles['avg_o2'] >= 0).all(), "Average O2 should be positive"
        assert (cycles['std_o2'] >= 0).all(), "Std deviation should be non-negative"
        
        # Check temporal progression
        time_diffs = cycles['cycle_start'].diff().dropna()
        median_cycle_time = time_diffs.median()
        
        # Cycles should be roughly regular (within 2x median)
        assert (time_diffs <= median_cycle_time * 2).all(), \
            "Some cycles have unusually long gaps"
    
    def test_cycle_completeness(self, loader, sample_plate):
        """Test that cycles have complete data."""
        cycles = loader.load_cycle_statistics(sample_plate)
        
        # Most cycles should have no exclusions
        cycles_with_exclusions = (cycles['excluded_count'] > 0).sum()
        exclusion_rate = cycles_with_exclusions / len(cycles) * 100
        
        assert exclusion_rate < 10, \
            f"Too many cycles have exclusions: {exclusion_rate:.1f}%"
        
        # Check measurement consistency
        expected_measurements = cycles['wells_measured'].median()
        inconsistent_cycles = (
            (cycles['measurements'] < expected_measurements * 0.9) |
            (cycles['measurements'] > expected_measurements * 1.1)
        ).sum()
        
        assert inconsistent_cycles < len(cycles) * 0.05, \
            "Too many cycles have inconsistent measurement counts"


class TestOxygenDataQuality:
    """Test oxygen data quality and patterns."""
    
    @pytest.fixture
    def sample_data(self):
        """Load a sample of oxygen data for testing."""
        with DataLoader() as loader:
            return loader.load_oxygen_data(limit=1)
    
    def test_oxygen_data_structure(self, sample_data):
        """Test oxygen data structure and types."""
        assert set(sample_data.columns) == PROCESSED_DATA_COLUMNS
        
        # Check data types
        assert sample_data['o2'].dtype == 'float64'
        assert sample_data['elapsed_hours'].dtype == 'float64'
        assert pd.api.types.is_datetime64_any_dtype(sample_data['timestamp'])
        
        # Check no nulls in critical columns
        critical_columns = ['plate_id', 'well_id', 'o2', 'timestamp', 'elapsed_hours']
        for col in critical_columns:
            assert sample_data[col].notna().all(), f"Found nulls in {col}"
    
    def test_oxygen_value_distribution(self, sample_data):
        """Test that O2 values follow expected distribution."""
        o2_values = sample_data['o2']
        
        # Most values should be in typical range
        typical_count = o2_values.between(*TYPICAL_O2_RANGE).sum()
        typical_rate = typical_count / len(o2_values) * 100
        
        assert typical_rate >= 90, \
            f"Only {typical_rate:.1f}% of O2 values in typical range {TYPICAL_O2_RANGE}"
        
        # Check for reasonable statistics
        assert 0 <= o2_values.mean() <= 100, \
            f"Mean O2 {o2_values.mean():.2f} outside expected range"
        assert 0 <= o2_values.median() <= 100, \
            f"Median O2 {o2_values.median():.2f} outside expected range"
        assert o2_values.std() < 50, \
            f"O2 standard deviation {o2_values.std():.2f} is too high"
    
    def test_temporal_progression(self, sample_data):
        """Test that time progresses correctly."""
        # Group by well and check time progression
        for well_id, well_data in sample_data.groupby('well_id'):
            well_data = well_data.sort_values('timestamp')
            
            # Elapsed hours should increase
            assert (well_data['elapsed_hours'].diff().dropna() >= 0).all(), \
                f"Elapsed hours go backwards for well {well_id}"
            
            # Timestamps should increase
            assert well_data['timestamp'].is_monotonic_increasing, \
                f"Timestamps not monotonic for well {well_id}"
            
            # Check measurement frequency (should be roughly regular)
            time_diffs = well_data['elapsed_hours'].diff().dropna()
            if len(time_diffs) > 10:
                median_diff = time_diffs.median()
                # Allow 3x variation in measurement frequency
                assert (time_diffs <= median_diff * 3).all(), \
                    f"Irregular measurement frequency for well {well_id}"


class TestDataConsistency:
    """Test consistency between processed_data and other tables."""
    
    @pytest.fixture
    def data_samples(self):
        """Load samples from different tables."""
        with DataLoader() as loader:
            return {
                'oxygen': loader.load_oxygen_data(limit=2),
                'wells': loader.load_well_metadata(),
                'summary': loader.load_processed_data_summary(),
                'events': loader.load_media_events()
            }
    
    def test_well_consistency(self, data_samples):
        """Test that wells in processed_data match well_metadata."""
        oxygen_wells = set(data_samples['oxygen']['well_id'].unique())
        metadata_wells = set(data_samples['wells']['well_id'].unique())
        
        # Oxygen wells should be subset of metadata wells
        missing_wells = oxygen_wells - metadata_wells
        assert len(missing_wells) == 0 or len(missing_wells) / len(oxygen_wells) < 0.01, \
            f"Too many wells in oxygen data missing from metadata: {len(missing_wells)}"
    
    def test_plate_consistency(self, data_samples):
        """Test plate IDs are consistent."""
        oxygen_plates = set(data_samples['oxygen']['plate_id'].unique())
        summary_plates = set(data_samples['summary']['plate_id'].unique())
        
        # All oxygen plates should be in summary
        assert oxygen_plates.issubset(summary_plates), \
            "Some plates in oxygen data missing from summary"
    
    def test_drug_assignment_consistency(self, data_samples):
        """Test that drug assignments are consistent."""
        # Check that all non-control wells have drug assignments
        drug_counts = data_samples['oxygen']['drug'].value_counts()
        
        unknown_rate = drug_counts.get('Unknown', 0) / len(data_samples['oxygen']) * 100
        assert unknown_rate < 5, \
            f"Too many unknown drug assignments: {unknown_rate:.1f}%"
        
        # Check concentration values
        assert (data_samples['oxygen']['concentration'] >= 0).all(), \
            "Found negative concentrations"


class TestPerformance:
    """Test query performance for processed_data."""
    
    def test_summary_query_performance(self):
        """Test that summary queries complete quickly."""
        import time
        
        with DataLoader() as loader:
            start_time = time.time()
            summary = loader.load_processed_data_summary()
            elapsed = time.time() - start_time
        
        assert elapsed < 10, f"Summary query took too long: {elapsed:.2f} seconds"
        assert not summary.empty
    
    def test_large_data_query_performance(self):
        """Test performance of larger queries."""
        import time
        
        with DataLoader() as loader:
            start_time = time.time()
            validation = loader.validate_processed_data()
            elapsed = time.time() - start_time
        
        assert elapsed < 30, f"Validation query took too long: {elapsed:.2f} seconds"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])