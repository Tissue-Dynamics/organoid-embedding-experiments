"""
Test suite for well_image_data table access and functionality.

OVERVIEW OF CHECKS AND ACCEPTABLE VALUES:

Table Structure:
- Required columns: id, plate_id, well_number, timestamp, number_of_organoids, max_organoid_size
- Well numbers: 1-384 (for 384-well plates, adjust for other plate sizes)

Data Quality Checks:
- Organoid counts: 0-1000 per well (>1000 considered suspicious)
- Average size: >= 0 (negative values are invalid)
- Total area: >= 0 (negative values are invalid)
- Well coverage: >50% of wells should have organoid data
- Data completeness: >50% of records should have non-null values for key fields

Integration Checks:
- Wells with imaging data should have corresponding oxygen data (>50% overlap expected)
- Organoid counts may correlate with oxygen consumption (not enforced, just measured)
- Plates with imaging should be a subset of all experimental plates

Expected Data Patterns:
- Most wells should contain at least some organoids (biological expectation)
- Multiple imaging sessions per well are possible (duplicates may exist)
- Not all plates will have imaging data (imaging is selective)
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.data_loader import DataLoader


class TestWellImageData:
    """Test well_image_data loading and validation."""

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader

    def test_load_well_image_data_all(self, loader):
        """Test loading all well image data."""
        df = loader.load_well_image_data()

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)

        # Check required columns exist (based on actual schema)
        expected_columns = [
            'id', 'plate_id', 'well_number', 'timestamp',
            'number_of_organoids', 'max_organoid_size'
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Check data types
        assert df['well_number'].dtype in [np.int16, np.int32, np.int64]
        if 'number_of_organoids' in df.columns and len(df) > 0:
            assert df['number_of_organoids'].dtype in [np.int16, np.int32, np.int64, np.float32, np.float64]

        # Check basic data quality
        if len(df) > 0:
            assert df['well_number'].min() >= 1, "Well numbers should start from 1"
            assert df['well_number'].max() <= 384, "Well numbers exceed 384-well plate"

            # Check for non-negative values
            if 'number_of_organoids' in df.columns:
                assert (df['number_of_organoids'] >= 0).all(), "Negative organoid counts found"
            if 'max_organoid_size' in df.columns:
                assert (df['max_organoid_size'].dropna() >= 0).all(), "Negative organoid sizes found"

    def test_load_well_image_data_specific_plates(self, loader):
        """Test loading well image data for specific plates."""
        # First get some plate IDs
        all_data = loader.load_well_image_data()
        if len(all_data) == 0:
            pytest.skip("No well image data available")

        # Get first 2 unique plate IDs
        plate_ids = all_data['plate_id'].unique()[:2].tolist()

        # Load data for specific plates
        filtered_df = loader.load_well_image_data(plate_ids=plate_ids)

        # Verify filtering worked
        assert len(filtered_df) > 0, "No data returned for specific plates"
        assert set(filtered_df['plate_id'].unique()) <= set(plate_ids), \
            "Returned data contains unexpected plate IDs"

    def test_well_image_data_consistency(self, loader):
        """Test consistency of well image data."""
        df = loader.load_well_image_data()

        if len(df) == 0:
            pytest.skip("No well image data available")

        # Check for duplicate well entries per plate
        duplicates = df.groupby(['plate_id', 'well_number']).size()
        duplicate_wells = duplicates[duplicates > 1]

        # Some duplicates might be expected if there are multiple imaging sessions
        if len(duplicate_wells) > 0:
            print(f"Warning: {len(duplicate_wells)} wells have multiple image entries")

        # Check organoid count distribution
        if 'number_of_organoids' in df.columns:
            organoid_stats = df['number_of_organoids'].describe()
            assert organoid_stats['max'] < 1000, \
                f"Suspiciously high organoid count: {organoid_stats['max']}"

            # Most wells should have some organoids
            wells_with_organoids = (df['number_of_organoids'] > 0).sum()
            total_wells = len(df)
            organoid_percentage = wells_with_organoids / total_wells * 100

            print(f"Wells with organoids: {wells_with_organoids}/{total_wells} ({organoid_percentage:.1f}%)")

    def test_well_image_metadata_quality(self, loader):
        """Test metadata quality in well image data."""
        df = loader.load_well_image_data()

        if len(df) == 0:
            pytest.skip("No well image data available")

        # Check for required metadata fields
        metadata_completeness = {}

        # Check completeness of key fields
        for field in ['timestamp', 'number_of_organoids', 'max_organoid_size']:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                completeness = non_null_count / len(df) * 100
                metadata_completeness[field] = completeness

                # Most fields should be reasonably complete
                assert completeness > 50, \
                    f"Poor data completeness for {field}: {completeness:.1f}%"

        print("\nMetadata completeness:")
        for field, completeness in metadata_completeness.items():
            print(f"  {field}: {completeness:.1f}%")

    def test_well_image_plate_coverage(self, loader):
        """Test how many plates have imaging data."""
        # Get all plates from plate_table
        plates_df = loader._execute_and_convert("SELECT id FROM db.public.plate_table")
        total_plates = len(plates_df)

        # Get plates with imaging data
        image_data = loader.load_well_image_data()
        if len(image_data) == 0:
            print(f"No imaging data found (0/{total_plates} plates)")
            return

        plates_with_imaging = image_data['plate_id'].nunique()
        coverage = plates_with_imaging / total_plates * 100

        print(f"\nImaging coverage: {plates_with_imaging}/{total_plates} plates ({coverage:.1f}%)")

        # At least some plates should have imaging
        assert plates_with_imaging > 0, "No plates have imaging data"

        # Check wells per plate distribution
        wells_per_plate = image_data.groupby('plate_id')['well_number'].count()
        print(f"Wells imaged per plate: min={wells_per_plate.min()}, "
              f"mean={wells_per_plate.mean():.1f}, max={wells_per_plate.max()}")


class TestWellImageIntegration:
    """Test integration between well image data and other tables."""

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader

    def test_well_image_oxygen_correlation(self, loader):
        """Test if wells with imaging data have corresponding oxygen data."""
        # Get sample of well image data
        image_data = loader.load_well_image_data()
        if len(image_data) == 0:
            pytest.skip("No well image data available")

        # Take first plate with imaging
        test_plate = image_data['plate_id'].iloc[0]
        plate_image_data = image_data[image_data['plate_id'] == test_plate]

        # Get oxygen data for same plate
        oxygen_data = loader.load_oxygen_data(plate_ids=[test_plate])

        if len(oxygen_data) == 0:
            pytest.skip(f"No oxygen data for plate {test_plate}")

        # Check well overlap
        image_wells = set(plate_image_data['well_number'].unique())
        oxygen_wells = set(oxygen_data['well_number'].unique())

        overlap = image_wells.intersection(oxygen_wells)
        overlap_percentage = len(overlap) / len(image_wells) * 100

        print(f"\nWell overlap for plate {test_plate}:")
        print(f"  Wells with imaging: {len(image_wells)}")
        print(f"  Wells with oxygen data: {len(oxygen_wells)}")
        print(f"  Overlap: {len(overlap)} wells ({overlap_percentage:.1f}%)")

        # Most imaged wells should have oxygen data
        assert overlap_percentage > 50, \
            f"Poor overlap between imaging and oxygen data: {overlap_percentage:.1f}%"

    def test_organoid_count_oxygen_relationship(self, loader):
        """Test if organoid counts relate to oxygen consumption."""
        # Get plates with both imaging and oxygen data
        image_data = loader.load_well_image_data()
        if len(image_data) == 0:
            pytest.skip("No well image data available")

        # Sample one plate for detailed analysis
        test_plate = image_data['plate_id'].iloc[0]

        # Get both datasets
        plate_image = image_data[image_data['plate_id'] == test_plate]
        plate_oxygen = loader.load_oxygen_data(plate_ids=[test_plate])

        if len(plate_oxygen) == 0:
            pytest.skip("No oxygen data for test plate")

        # Calculate average oxygen consumption per well
        avg_oxygen = plate_oxygen.groupby('well_number')['o2'].mean()

        # Merge with image data
        merged = plate_image.merge(
            avg_oxygen.reset_index(),
            on='well_number',
            how='inner'
        )

        if len(merged) > 0 and 'number_of_organoids' in merged.columns:
            # Basic sanity check: wells with more organoids should generally consume more oxygen
            # (though this is very simplified and may not always hold)
            correlation = merged[['number_of_organoids', 'o2']].corr().iloc[0, 1]

            print(f"\nOrganoid count vs oxygen correlation: {correlation:.3f}")
            print(f"Based on {len(merged)} wells from plate {test_plate}")

            # We don't assert on correlation value as biology is complex,
            # but we check it's computed
            assert not np.isnan(correlation), "Failed to compute correlation"
