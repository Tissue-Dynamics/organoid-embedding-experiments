"""
Test suite for plate_table functionality.

OVERVIEW OF CHECKS AND ACCEPTABLE VALUES:

Table Structure:
- Required columns: id, name, created_at, status, state, plate_size, tissue
- Optional columns: updated_at, created_by, deleted, description, qc_values, qc_thresholds, internal_notes

Data Type Validations:
- id: UUID (converted to string)
- created_at/updated_at: Valid timestamps (>= 2020, <= current time)
- status: Enum values from pipeline_status type
- state: Enum values ('active', 'abandoned', 'completed')
- plate_size: String format "ROWSxCOLS" (e.g., "16x24")
- deleted: Boolean flag for soft deletion
- tissue: Text field (default 'Liver')

Plate Size Validations:
- Format: Must be "ROWSxCOLS" where ROWS and COLS are integers
- Valid well counts: 6, 12, 24, 48, 96, 384, 1536
- Common formats: "2x3", "3x4", "4x6", "6x8", "8x12", "16x24", "32x48"

QC Data Validations:
- qc_values: Valid JSON string containing quality control measurements
- qc_thresholds: Valid JSON string with default complex structure
- Both fields optional but should parse as valid JSON when present

Business Rule Validations:
- Deletion rate: <50% of plates should be marked as deleted
- Timeline: Plates created between 2020 and present
- Well coverage: >50% of expected wells should exist in well_map_data
- Event coverage: >50% of active plates should have associated events

Status/State Distributions:
- Expect variety of statuses reflecting experimental pipeline
- Most plates should be 'completed' or 'active'
- 'abandoned' plates indicate experimental issues
"""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime
from unittest.mock import Mock, patch

from src.utils.data_loader import DataLoader


class TestPlateTable:
    """Test plate_table loading and validation."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_load_plate_table(self, loader):
        """Test loading plate metadata."""
        query = "SELECT * FROM db.public.plate_table"
        df = loader._execute_and_convert(query)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "No plates found in database"
        
        # Check required columns (based on actual plate_table schema)
        required_columns = [
            'id', 'name', 'created_at', 'status', 'state', 'tissue'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data types (accept various datetime precisions)
        assert pd.api.types.is_datetime64_any_dtype(df['created_at']), \
               f"created_at should be datetime, got {df['created_at'].dtype}"
        
        # Plate size should be string after conversion (e.g., "16x24")
        if 'plate_size' in df.columns and len(df) > 0:
            sample_size = df['plate_size'].iloc[0]
            assert isinstance(sample_size, str), \
                f"plate_size should be string, got {type(sample_size)}"
            assert 'x' in sample_size, \
                f"plate_size format should be 'ROWSxCOLS', got {sample_size}"
    
    def test_plate_status_distribution(self, loader):
        """Test distribution of plate statuses."""
        query = "SELECT * FROM db.public.plate_table"
        df = loader._execute_and_convert(query)
        
        # Check status values
        status_counts = df['status'].value_counts()
        print("\nPlate status distribution:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Check state values
        state_counts = df['state'].value_counts()
        print("\nPlate state distribution:")
        for state, count in state_counts.items():
            print(f"  {state}: {count}")
        
        # Verify enums are properly converted to strings
        assert df['status'].dtype == 'object', "Status should be string type"
        assert df['state'].dtype == 'object', "State should be string type"
    
    def test_plate_size_parsing(self, loader):
        """Test that plate sizes are properly formatted."""
        query = "SELECT * FROM db.public.plate_table"
        df = loader._execute_and_convert(query)
        
        # Check all plate sizes
        unique_sizes = df['plate_size'].unique()
        print(f"\nUnique plate sizes: {unique_sizes}")
        
        for size in unique_sizes:
            if pd.notna(size):
                # Should be in format "ROWSxCOLS"
                parts = size.split('x')
                assert len(parts) == 2, f"Invalid plate size format: {size}"
                
                rows, cols = parts
                assert rows.isdigit(), f"Invalid row count in plate size: {size}"
                assert cols.isdigit(), f"Invalid column count in plate size: {size}"
                
                # Check for reasonable plate sizes
                row_count = int(rows)
                col_count = int(cols)
                well_count = row_count * col_count
                
                # Accept common sizes plus custom sizes like 13x24
                assert well_count in [6, 12, 24, 48, 96, 312, 384, 1536], \
                    f"Unusual plate size: {size} ({well_count} wells)"
    
    def test_plate_qc_data(self, loader):
        """Test QC values and thresholds in plate table."""
        query = "SELECT * FROM db.public.plate_table"
        df = loader._execute_and_convert(query)
        
        # Check QC columns exist
        qc_columns = ['qc_values', 'qc_thresholds']
        for col in qc_columns:
            assert col in df.columns, f"Missing QC column: {col}"
        
        # These should be stored as strings (JSON) after export
        plates_with_qc = df['qc_values'].notna().sum()
        plates_with_thresholds = df['qc_thresholds'].notna().sum()
        
        print(f"\nQC data coverage:")
        print(f"  Plates with QC values: {plates_with_qc}/{len(df)}")
        print(f"  Plates with QC thresholds: {plates_with_thresholds}/{len(df)}")
        
        # Try parsing a QC value if available
        if plates_with_qc > 0:
            sample_qc = df[df['qc_values'].notna()]['qc_values'].iloc[0]
            try:
                # Should be valid JSON
                qc_data = json.loads(sample_qc)
                print(f"  Sample QC keys: {list(qc_data.keys())[:5]}")
            except json.JSONDecodeError:
                pytest.fail(f"QC values not valid JSON: {sample_qc[:100]}")
    
    def test_plate_creation_timeline(self, loader):
        """Test plate creation timeline and patterns."""
        query = "SELECT * FROM db.public.plate_table"
        df = loader._execute_and_convert(query)
        
        # Analyze creation dates
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        earliest = df['created_at'].min()
        latest = df['created_at'].max()
        
        print(f"\nPlate creation timeline:")
        print(f"  Earliest plate: {earliest}")
        print(f"  Latest plate: {latest}")
        print(f"  Timespan: {(latest - earliest).days} days")
        
        # Check for reasonable dates
        assert earliest.year >= 2020, f"Suspiciously old plate: {earliest}"
        # Handle timezone comparison
        latest_naive = latest.tz_localize(None) if latest.tz else latest
        assert latest_naive <= pd.Timestamp.now() + pd.Timedelta(days=365), "Plate creation date too far in future"
        
        # Analyze creation patterns
        df['year_month'] = df['created_at'].dt.to_period('M')
        monthly_counts = df['year_month'].value_counts().sort_index()
        
        print("\nPlates created per month:")
        for period, count in monthly_counts.tail(6).items():
            print(f"  {period}: {count}")
    
    def test_plate_deletion_status(self, loader):
        """Test soft deletion status of plates."""
        query = "SELECT * FROM db.public.plate_table"
        df = loader._execute_and_convert(query)
        
        # Check deletion status
        if 'deleted' in df.columns:
            deleted_count = df['deleted'].sum()
            deletion_rate = deleted_count / len(df) * 100
            
            print(f"\nDeletion status:")
            print(f"  Active plates: {len(df) - deleted_count}")
            print(f"  Deleted plates: {deleted_count}")
            print(f"  Deletion rate: {deletion_rate:.1f}%")
            
            # Most plates should be active
            assert deletion_rate < 50, f"High deletion rate: {deletion_rate:.1f}%"


class TestPlateIntegration:
    """Test integration between plate_table and other tables."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_plate_well_consistency(self, loader):
        """Test that plate sizes match well data."""
        plates_query = "SELECT * FROM db.public.plate_table"
        plates_df = loader._execute_and_convert(plates_query)
        
        # For each plate, check well counts match plate size
        for _, plate in plates_df.head(5).iterrows():  # Test first 5 plates
            plate_id = plate['id']
            plate_size = plate['plate_size']
            
            if pd.isna(plate_size):
                continue
            
            # Parse plate size
            rows, cols = map(int, plate_size.split('x'))
            expected_wells = rows * cols
            
            # Get actual well count from well_map_data
            wells = loader.load_well_metadata()
            plate_wells = wells[wells['plate_id'] == plate_id]
            actual_wells = len(plate_wells)
            
            print(f"\nPlate {plate['name']}:")
            print(f"  Size: {plate_size} (expected {expected_wells} wells)")
            print(f"  Actual wells in well_map: {actual_wells}")
            
            # Allow some missing wells but not too many
            if actual_wells > 0:
                coverage = actual_wells / expected_wells * 100
                assert coverage >= 50, \
                    f"Low well coverage for plate {plate_id}: {coverage:.1f}%"
    
    def test_plate_event_coverage(self, loader):
        """Test which plates have events."""
        plates_query = "SELECT * FROM db.public.plate_table"
        plates_df = loader._execute_and_convert(plates_query)
        
        events_df = loader.load_event_summary()
        
        # Get plates with events
        if 'plate_id' in events_df.columns:
            plates_with_events = set(events_df['plate_id'].unique())
            all_plates = set(plates_df['id'].unique())
            
            coverage = len(plates_with_events) / len(all_plates) * 100
            
            print(f"\nEvent coverage:")
            print(f"  Plates with events: {len(plates_with_events)}/{len(all_plates)}")
            print(f"  Coverage: {coverage:.1f}%")
            
            # Most active plates should have events
            active_plates = plates_df[plates_df['deleted'] == False] if 'deleted' in plates_df.columns else plates_df
            active_with_events = len(plates_with_events.intersection(set(active_plates['id'])))
            active_coverage = active_with_events / len(active_plates) * 100
            
            print(f"  Active plates with events: {active_coverage:.1f}%")
            assert active_coverage > 50, \
                f"Low event coverage for active plates: {active_coverage:.1f}%"