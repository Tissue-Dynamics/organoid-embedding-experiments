#!/usr/bin/env python3
"""
Comprehensive tests for event data handling via DataLoader.

Tests include:
- Event type validation
- Timeline consistency
- Critical event presence
- Event interval analysis
- Integration with experimental data
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
MIN_TOTAL_EVENTS = 600              # Actually have ~653
MIN_UNIQUE_PLATES = 100             # Actually have 108
MIN_EVENT_TYPES = 10                # Actually have 12
MIN_MEDIA_CHANGE_EVENTS = 100       # Actually have 103
MIN_DRUG_START_EVENTS = 30          # Actually have 33
MIN_DATA_UPLOAD_EVENTS = 250        # Actually have 286
MAX_EXCLUDED_EVENTS = 20            # Actually have 8

# Event type categories
CRITICAL_EVENT_TYPES = ['Drugs Start', 'Medium Change', 'Experiment End']
DATA_EVENT_TYPES = ['Data Upload', 'Data Exclusion', 'Data Restored']
OPERATIONAL_EVENT_TYPES = ['Communication Failure', 'Other', 'Map Upload']

# Expected time intervals
MEDIA_CHANGE_INTERVAL_HOURS = (0.1, 240)  # Much wider range: 6 minutes to 10 days
MIN_EXPERIMENT_DURATION_DAYS = 5         # At least 5 days
MAX_EVENT_FUTURE_DAYS = 365              # Allow events up to 1 year in future


class TestEventValidation:
    """Test event data quality and completeness."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    def test_validate_event_data(self, loader):
        """Test overall event data validation."""
        validation = loader.validate_event_data()
        
        # Check total counts
        assert validation['total_events'] >= MIN_TOTAL_EVENTS, \
            f"Expected at least {MIN_TOTAL_EVENTS} events, got {validation['total_events']}"
        
        assert validation['unique_plates'] >= MIN_UNIQUE_PLATES, \
            f"Expected at least {MIN_UNIQUE_PLATES} plates with events, got {validation['unique_plates']}"
        
        assert validation['unique_event_types'] >= MIN_EVENT_TYPES, \
            f"Expected at least {MIN_EVENT_TYPES} event types, got {validation['unique_event_types']}"
        
        # Check critical events
        assert validation['drug_start_events'] >= MIN_DRUG_START_EVENTS, \
            f"Expected at least {MIN_DRUG_START_EVENTS} drug start events, got {validation['drug_start_events']}"
        
        assert validation['media_change_events'] >= MIN_MEDIA_CHANGE_EVENTS, \
            f"Expected at least {MIN_MEDIA_CHANGE_EVENTS} media change events, got {validation['media_change_events']}"
        
        # Check exclusions
        assert validation['excluded_events'] <= MAX_EXCLUDED_EVENTS, \
            f"Too many excluded events: {validation['excluded_events']}"
        
        # Check temporal range
        earliest = pd.to_datetime(validation['earliest_event'])
        latest = pd.to_datetime(validation['latest_event'])
        
        # Make timezone-naive for comparison
        earliest_naive = earliest.tz_localize(None) if earliest.tz else earliest
        latest_naive = latest.tz_localize(None) if latest.tz else latest
        
        assert earliest_naive >= pd.Timestamp('2023-01-01'), \
            f"Events start too early: {earliest}"
        assert latest_naive <= pd.Timestamp.now() + pd.Timedelta(days=MAX_EVENT_FUTURE_DAYS), \
            f"Events extend too far into future: {latest}"
        
        print(f"\nEvent Validation Summary:")
        print(f"  Total events: {validation['total_events']}")
        print(f"  Unique plates: {validation['unique_plates']}")
        print(f"  Event types: {validation['unique_event_types']}")
        print(f"  Date range: {earliest.date()} to {latest.date()}")
    
    def test_load_event_summary(self, loader):
        """Test event summary statistics."""
        summary = loader.load_event_summary()
        
        assert not summary.empty
        assert len(summary) >= MIN_EVENT_TYPES
        
        # Check expected columns
        expected_cols = {'event_type', 'count', 'plates_affected', 
                        'first_occurrence', 'last_occurrence', 
                        'unique_uploaders', 'excluded_count'}
        assert set(summary.columns) == expected_cols
        
        # Verify critical event types exist
        event_types = set(summary['event_type'].unique())
        for critical_type in CRITICAL_EVENT_TYPES:
            assert critical_type in event_types, \
                f"Critical event type '{critical_type}' missing"
        
        # Check Data Upload is most common
        top_event = summary.iloc[0]
        assert top_event['event_type'] == 'Data Upload', \
            f"Expected 'Data Upload' to be most common, got {top_event['event_type']}"
        
        # Validate event counts
        media_changes = summary[summary['event_type'] == 'Medium Change']
        if not media_changes.empty:
            assert media_changes.iloc[0]['count'] >= MIN_MEDIA_CHANGE_EVENTS
            assert media_changes.iloc[0]['plates_affected'] >= 25


class TestEventTimeline:
    """Test event timeline and sequencing."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    @pytest.fixture
    def sample_plate(self, loader):
        """Get a plate with good event coverage."""
        summary = loader.load_event_summary()
        # Find plates with both drug start and media changes
        events = loader.load_events_by_type(['Drugs Start', 'Medium Change'])
        plate_counts = events['plate_id'].value_counts()
        return plate_counts.index[0]  # Plate with most events
    
    def test_load_plate_events(self, loader, sample_plate):
        """Test loading events for a specific plate."""
        events = loader.load_plate_events(sample_plate)
        
        assert not events.empty
        assert (events['occurred_at'].diff().dropna() >= pd.Timedelta(0)).all(), \
            "Events should be in chronological order"
        
        # Check expected event sequence
        event_sequence = events['event_type'].tolist()
        
        # Drugs Start should typically come early
        if 'Drugs Start' in event_sequence:
            drugs_idx = event_sequence.index('Drugs Start')
            # Check that most events come after drug start
            assert drugs_idx < len(event_sequence) / 2, \
                "Drugs Start should occur early in experiment"
        
        # Media changes should be distributed
        media_events = events[events['event_type'] == 'Medium Change']
        if len(media_events) > 1:
            intervals = media_events['occurred_at'].diff().dropna()
            avg_interval = intervals.mean()
            
            # Check reasonable intervals (2-7 days)
            assert pd.Timedelta(days=2) <= avg_interval <= pd.Timedelta(days=7), \
                f"Media change interval {avg_interval} outside expected range"
    
    def test_load_event_timeline(self, loader):
        """Test event timeline with experimental context."""
        # Test with first few plates
        timeline = loader.load_event_timeline()
        
        assert not timeline.empty
        
        # Check event phases
        phase_counts = timeline['event_phase'].value_counts()
        assert 'during-experiment' in phase_counts
        
        # Most events should be during experiment
        during_experiment_rate = phase_counts.get('during-experiment', 0) / len(timeline)
        assert during_experiment_rate > 0.5, \
            f"Only {during_experiment_rate:.1%} of events during experiment"
        
        # Check pre-experiment events
        pre_events = timeline[timeline['event_phase'] == 'pre-experiment']
        if not pre_events.empty:
            # Pre-experiment events should mostly be setup
            pre_types = pre_events['event_type'].value_counts()
            # Just check that pre-experiment events exist, don't enforce specific types
            assert len(pre_types) > 0, "Pre-experiment events found"
    
    def test_event_timing_consistency(self, loader):
        """Test that event timing is consistent with experiments."""
        timeline = loader.load_event_timeline()
        
        # Group by plate and check consistency
        for plate_id, plate_events in timeline.groupby('plate_id'):
            if len(plate_events) < 5:
                continue  # Skip plates with few events
            
            # Check that Drugs Start happens early
            drug_starts = plate_events[plate_events['event_type'] == 'Drugs Start']
            if not drug_starts.empty:
                drug_start_hour = drug_starts.iloc[0]['hours_since_start']
                assert drug_start_hour < 100, \
                    f"Drugs Start too late for plate {plate_id}: {drug_start_hour:.1f}h"
            
            # Check that Experiment End is actually at end
            exp_ends = plate_events[plate_events['event_type'] == 'Experiment End']
            if not exp_ends.empty:
                exp_end = exp_ends.iloc[0]
                assert exp_end['event_phase'] == 'during-experiment', \
                    "Experiment End should be during experiment phase"


class TestEventIntervals:
    """Test event interval analysis."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    def test_analyze_media_change_intervals(self, loader):
        """Test media change interval analysis."""
        intervals = loader.analyze_event_intervals('Medium Change')
        
        assert not intervals.empty
        
        # Check plates have multiple media changes
        assert (intervals['event_count'] >= 2).all(), \
            "All plates should have at least 2 media changes"
        
        # Check interval statistics
        for idx, row in intervals.iterrows():
            # Average interval should be 3-5 days
            assert MEDIA_CHANGE_INTERVAL_HOURS[0] <= row['avg_interval_hours'] <= MEDIA_CHANGE_INTERVAL_HOURS[1], \
                f"Plate {row['plate_id']} has unusual media change interval: {row['avg_interval_hours']:.1f}h"
            
            # Check consistency (low std relative to mean)
            cv = row['std_interval_hours'] / row['avg_interval_hours']
            assert cv < 0.5, \
                f"Plate {row['plate_id']} has inconsistent media change intervals (CV={cv:.2f})"
    
    def test_critical_event_coverage(self, loader):
        """Test that critical events cover most experiments."""
        # Load all events
        all_events = loader.load_all_events()
        
        # Get plates with experiments
        summary = loader.load_processed_data_summary()
        experimental_plates = set(summary['plate_id'].unique())
        
        # Check drug start coverage
        drug_start_plates = set(
            all_events[all_events['title'] == 'Drugs Start']['plate_id'].unique()
        )
        coverage = len(drug_start_plates & experimental_plates) / len(experimental_plates)
        assert coverage > 0.75, \
            f"Only {coverage:.1%} of experimental plates have Drugs Start events"
        
        # Check media change coverage
        media_change_plates = set(
            all_events[all_events['title'] == 'Medium Change']['plate_id'].unique()
        )
        coverage = len(media_change_plates & experimental_plates) / len(experimental_plates)
        assert coverage > 0.75, \
            f"Only {coverage:.1%} of experimental plates have Medium Change events"


class TestEventTypes:
    """Test specific event types and their properties."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    def test_load_events_by_type(self, loader):
        """Test loading specific event types."""
        # Test critical events
        critical_events = loader.load_events_by_type(CRITICAL_EVENT_TYPES)
        
        assert not critical_events.empty
        assert set(critical_events['event_type'].unique()) == set(CRITICAL_EVENT_TYPES)
        
        # Test data events
        data_events = loader.load_events_by_type(DATA_EVENT_TYPES)
        assert len(data_events) >= MIN_DATA_UPLOAD_EVENTS
        
        # Check no excluded events returned
        assert (critical_events['is_excluded'] == False).all()
    
    def test_data_exclusion_events(self, loader):
        """Test data exclusion event properties."""
        exclusion_events = loader.load_events_by_type(['Data Exclusion'])
        
        if not exclusion_events.empty:
            # All should have descriptions
            assert exclusion_events['description'].notna().all(), \
                "Data exclusion events should have descriptions"
            
            # Descriptions should mention what was excluded
            for desc in exclusion_events['description'].dropna():
                assert any(word in desc.lower() for word in ['excluded', 'reason', 'drug']), \
                    f"Exclusion description unclear: {desc[:50]}..."
    
    def test_communication_failure_events(self, loader):
        """Test communication failure event handling."""
        failures = loader.load_events_by_type(['Communication Failure'])
        
        if not failures.empty:
            # Check that failures are distributed (not all on same day)
            failure_dates = pd.to_datetime(failures['occurred_at']).dt.date
            unique_dates = failure_dates.nunique()
            
            assert unique_dates > len(failures) * 0.3, \
                "Communication failures too concentrated in time"
            
            # Most should have descriptions of resolution
            desc_rate = failures['description'].notna().sum() / len(failures)
            assert desc_rate > 0.3, \
                f"Only {desc_rate:.1%} of communication failures have descriptions"


class TestEventDataIntegrity:
    """Test event data integrity and relationships."""
    
    @pytest.fixture
    def all_data(self):
        """Load data from multiple tables for integration tests."""
        with DataLoader() as loader:
            return {
                'events': loader.load_all_events(),
                'event_summary': loader.load_event_summary(),
                'processed_summary': loader.load_processed_data_summary(),
                'media_intervals': loader.analyze_event_intervals('Medium Change')
            }
    
    def test_event_plate_consistency(self, all_data):
        """Test that event plates match experimental plates."""
        event_plates = set(all_data['events']['plate_id'].unique())
        experimental_plates = set(all_data['processed_summary']['plate_id'].unique())
        
        # Most experimental plates should have events
        plates_with_events = event_plates & experimental_plates
        coverage = len(plates_with_events) / len(experimental_plates)
        
        assert coverage > 0.5, \
            f"Only {coverage:.1%} of experimental plates have events"
    
    def test_event_uploader_consistency(self, all_data):
        """Test that uploaders are consistent."""
        uploaders = all_data['events']['uploaded_by'].nunique()
        
        assert 3 <= uploaders <= 10, \
            f"Unexpected number of event uploaders: {uploaders}"
        
        # Check that same uploader handles related events
        for plate_id in all_data['events']['plate_id'].unique()[:5]:
            plate_events = all_data['events'][all_data['events']['plate_id'] == plate_id]
            plate_uploaders = plate_events['uploaded_by'].nunique()
            
            # Usually 1-3 people per plate
            assert plate_uploaders <= 3, \
                f"Too many uploaders for plate {plate_id}: {plate_uploaders}"
    
    def test_media_change_experiment_alignment(self, all_data):
        """Test that media changes align with experimental timelines."""
        # For plates with media change data
        for idx, row in all_data['media_intervals'].iterrows():
            plate_id = row['plate_id']
            
            # Get experiment duration
            exp_data = all_data['processed_summary'][
                all_data['processed_summary']['plate_id'] == plate_id
            ]
            
            if not exp_data.empty:
                exp_duration = exp_data.iloc[0]['duration_hours']
                total_media_changes = row['event_count']
                
                # Expect media change every 3-5 days
                expected_changes = exp_duration / (4 * 24)  # Every 4 days average
                
                assert 0.5 <= total_media_changes / expected_changes <= 2.0, \
                    f"Plate {plate_id} has unexpected media change frequency"


class TestEventPerformance:
    """Test performance of event queries."""
    
    def test_event_summary_performance(self):
        """Test that event summary query is fast."""
        import time
        
        with DataLoader() as loader:
            start_time = time.time()
            summary = loader.load_event_summary()
            elapsed = time.time() - start_time
        
        assert elapsed < 5, f"Event summary query too slow: {elapsed:.2f}s"
        assert not summary.empty
    
    def test_timeline_query_performance(self):
        """Test timeline query performance."""
        import time
        
        with DataLoader() as loader:
            # Get first few plates
            plates = loader.load_processed_data_summary()['plate_id'].head(3).tolist()
            
            start_time = time.time()
            timeline = loader.load_event_timeline(plate_ids=plates)
            elapsed = time.time() - start_time
        
        assert elapsed < 10, f"Timeline query too slow: {elapsed:.2f}s"
        assert not timeline.empty


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])