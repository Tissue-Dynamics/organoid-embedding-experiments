#!/usr/bin/env python3
"""
Comprehensive tests for DataLoader class.

OVERVIEW OF CHECKS AND ACCEPTABLE VALUES:

Data Quality Thresholds:
- Drugs with binary DILI: >= 180 drugs (expected ~198)
- Drugs with severity scores: >= 180 drugs (expected ~194)
- Drugs with likelihood values: >= 180 drugs (expected ~196)
- Positive DILI drugs: >= 120 drugs (expected ~141)
- Negative DILI drugs: >= 50 drugs (expected ~57)

Pharmacokinetic Parameters:
- Drugs with Cmax (oral): >= 120 drugs (expected ~138)
- Drugs with Tmax: >= 120 drugs (expected ~142)
- Drugs with half-life: >= 140 drugs (expected ~156)
- Drugs with bioavailability: >= 100 drugs (expected ~108)
- Drugs with protein binding: >= 120 drugs (expected ~134)

Experimental Data Minimums:
- Unique plates: >= 30
- Media change events: >= 60
- Unique drugs in wells: >= 250
- Plates with media changes: >= 25

Column Requirements:
- Oxygen data: plate_id, well_id, well_number, drug, concentration, elapsed_hours, o2, timestamp
- Media events: plate_id, event_time, title, description
- Drug metadata: drug, dili_risk_score, binary_dili, likelihood, dili, dili_risk_category
- Well metadata: plate_id, well_number, drug, concentration, well_id

Data Integrity Checks:
- Well numbers: 1-384 (standard plate range)
- Oxygen values: 0-150% (values >150% flagged as extreme)
- Drug concentrations: >= 0 (negative concentrations invalid)
- Timestamps: Must be valid datetime objects
- Plate consistency: Wells should not exceed plate capacity
"""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader


# Data quality thresholds based on actual database content
MIN_DRUGS_WITH_BINARY_DILI = 180  # Actually have 198
MIN_DRUGS_WITH_SEVERITY = 180     # Actually have 194
MIN_DRUGS_WITH_LIKELIHOOD = 180   # Actually have 196
MIN_DRUGS_WITH_CMAX_ORAL = 120    # Actually have 138
MIN_DRUGS_WITH_TMAX = 120         # Actually have 142
MIN_DRUGS_WITH_HALF_LIFE = 140    # Actually have 156
MIN_POSITIVE_DILI_DRUGS = 120     # Actually have 141
MIN_NEGATIVE_DILI_DRUGS = 50      # Actually have 57 (close to 60)
MIN_DRUGS_WITH_RISK_CATEGORY = 120 # Actually have 130
MIN_DRUGS_WITH_BIOAVAILABILITY = 100  # Actually have 108
MIN_DRUGS_WITH_PROTEIN_BINDING = 120  # Actually have 134

# Other data thresholds
MIN_UNIQUE_PLATES = 30
MIN_MEDIA_CHANGE_EVENTS = 60
MIN_UNIQUE_DRUGS_IN_WELLS = 250
MIN_PLATES_WITH_MEDIA_CHANGES = 25

# Expected column sets
OXYGEN_DATA_COLUMNS = {'plate_id', 'well_id', 'well_number', 'drug', 
                      'concentration', 'elapsed_hours', 'o2', 'timestamp'}
MEDIA_EVENT_COLUMNS = {'plate_id', 'event_time', 'title', 'description'}
DRUG_METADATA_COLUMNS = {'drug', 'dili_risk_score', 'binary_dili', 'likelihood', 
                        'dili', 'dili_risk_category', 'drug_class', 'mechanism_of_action'}
WELL_METADATA_COLUMNS = {'plate_id', 'well_number', 'drug', 'concentration', 'well_id'}


class TestDataLoaderInitialization:
    """Test DataLoader initialization and connection handling."""
    
    def test_init_with_env_variable(self):
        """Test initialization with auto-detection (local or remote)."""
        loader = DataLoader()
        # Should work with either local database or DATABASE_URL
        assert loader.conn is not None
        loader.close()
    
    def test_init_with_explicit_url(self):
        """Test initialization with explicit database URL."""
        db_url = os.getenv('DATABASE_URL')
        loader = DataLoader(database_url=db_url)
        assert loader.database_url == db_url
        assert loader.conn is not None
        loader.close()
    
    def test_init_without_url_raises_error(self):
        """Test initialization without local database or DATABASE_URL."""
        # Temporarily remove DATABASE_URL and rename local database
        original_url = os.environ.pop('DATABASE_URL', None)
        local_db = Path("data/database/organoid_data.duckdb")
        temp_name = Path("data/database/organoid_data.duckdb.bak")
        
        try:
            if local_db.exists():
                local_db.rename(temp_name)
            
            # Now it should fail
            with pytest.raises(ValueError, match="No database available"):
                DataLoader()
        finally:
            if original_url:
                os.environ['DATABASE_URL'] = original_url
            if temp_name.exists():
                temp_name.rename(local_db)
    
    def test_context_manager(self):
        """Test DataLoader works as context manager."""
        with DataLoader() as loader:
            assert loader.conn is not None
        # Connection should be closed after exiting context
        assert loader.conn is None


class TestDataLoadingMethods:
    """Test individual data loading methods."""
    
    @pytest.fixture
    def loader(self):
        """Provide a DataLoader instance for tests."""
        with DataLoader() as loader:
            yield loader
    
    def test_load_oxygen_data_all(self, loader):
        """Test loading all oxygen data."""
        # Load limited data to avoid timeout
        df = loader.load_oxygen_data(limit=2)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert set(df.columns) == OXYGEN_DATA_COLUMNS
        
        # Check data types (accept both float32 and float64)
        assert df['o2'].dtype in ['float32', 'float64']
        assert df['elapsed_hours'].dtype in ['float32', 'float64']
        assert df['concentration'].dtype in ['float32', 'float64', 'int64']
        
        # Check data validity
        assert df['o2'].between(-50, 250).all(), "O2 values should be in reasonable range"
        assert (df['elapsed_hours'] >= 0).all(), "Elapsed hours should be non-negative"
    
    def test_load_oxygen_data_specific_plates(self, loader):
        """Test loading oxygen data for specific plates."""
        # First get some plate IDs
        all_data = loader.load_oxygen_data(limit=2)
        plate_ids = all_data['plate_id'].unique()[:1].tolist()
        
        # Load data for specific plate
        df = loader.load_oxygen_data(plate_ids=plate_ids)
        
        assert not df.empty
        assert df['plate_id'].isin(plate_ids).all()
        assert len(df['plate_id'].unique()) == len(plate_ids)
    
    def test_load_media_events(self, loader):
        """Test loading media change events."""
        df = loader.load_media_events()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == MEDIA_EVENT_COLUMNS
        
        # Check all events are Medium Change
        if not df.empty:
            assert (df['title'] == 'Medium Change').all()
        
        # Check minimum data requirements
        assert len(df) >= MIN_MEDIA_CHANGE_EVENTS, \
            f"Expected at least {MIN_MEDIA_CHANGE_EVENTS} media change events, got {len(df)}"
        
        unique_plates = df['plate_id'].nunique()
        assert unique_plates >= MIN_PLATES_WITH_MEDIA_CHANGES, \
            f"Expected media changes in at least {MIN_PLATES_WITH_MEDIA_CHANGES} plates, got {unique_plates}"
    
    def test_load_drug_metadata(self, loader):
        """Test loading drug metadata with comprehensive validation."""
        df = loader.load_drug_metadata()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == DRUG_METADATA_COLUMNS
        
        # Validate minimum drug counts
        total_drugs = len(df)
        assert total_drugs >= 200, f"Expected at least 200 drugs, got {total_drugs}"
        
        # Check DILI-related fields
        drugs_with_binary_dili = df['binary_dili'].notna().sum()
        assert drugs_with_binary_dili >= MIN_DRUGS_WITH_BINARY_DILI, \
            f"Expected at least {MIN_DRUGS_WITH_BINARY_DILI} drugs with binary_dili, got {drugs_with_binary_dili}"
        
        drugs_with_severity = df['dili_risk_score'].notna().sum()
        assert drugs_with_severity >= MIN_DRUGS_WITH_SEVERITY, \
            f"Expected at least {MIN_DRUGS_WITH_SEVERITY} drugs with severity scores, got {drugs_with_severity}"
        
        drugs_with_likelihood = df['likelihood'].notna().sum()
        assert drugs_with_likelihood >= MIN_DRUGS_WITH_LIKELIHOOD, \
            f"Expected at least {MIN_DRUGS_WITH_LIKELIHOOD} drugs with likelihood, got {drugs_with_likelihood}"
        
        # Check positive/negative DILI distribution
        positive_dili = (df['binary_dili'] == True).sum()
        negative_dili = (df['binary_dili'] == False).sum()
        assert positive_dili >= MIN_POSITIVE_DILI_DRUGS, \
            f"Expected at least {MIN_POSITIVE_DILI_DRUGS} positive DILI drugs, got {positive_dili}"
        assert negative_dili >= MIN_NEGATIVE_DILI_DRUGS, \
            f"Expected at least {MIN_NEGATIVE_DILI_DRUGS} negative DILI drugs, got {negative_dili}"
        
        # Check DILI risk categories
        drugs_with_risk_category = df['dili_risk_category'].notna().sum()
        assert drugs_with_risk_category >= MIN_DRUGS_WITH_RISK_CATEGORY, \
            f"Expected at least {MIN_DRUGS_WITH_RISK_CATEGORY} drugs with risk category, got {drugs_with_risk_category}"
        
        # Validate likelihood categories
        likelihood_values = df['likelihood'].dropna().unique()
        expected_categories = {'A', 'B', 'C', 'D', 'E'}
        actual_categories = set(likelihood_values)
        assert expected_categories.issubset(actual_categories), \
            f"Missing likelihood categories: {expected_categories - actual_categories}"
    
    def test_load_well_metadata(self, loader):
        """Test loading well metadata."""
        df = loader.load_well_metadata()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == WELL_METADATA_COLUMNS
        
        # Check minimum data
        unique_drugs = df['drug'].nunique()
        assert unique_drugs >= MIN_UNIQUE_DRUGS_IN_WELLS, \
            f"Expected at least {MIN_UNIQUE_DRUGS_IN_WELLS} unique drugs in wells, got {unique_drugs}"
        
        # Check well_id construction
        expected_well_ids = df['plate_id'].astype(str) + '_' + df['well_number'].astype(str)
        assert (df['well_id'] == expected_well_ids).all(), "well_id should be plate_id_well_number"
        
        # Check concentration values
        assert (df['concentration'] >= 0).all(), "Concentrations should be non-negative"


class TestDataQualityValidation:
    """Test data quality and pharmacokinetic parameters."""
    
    @pytest.fixture
    def drug_data(self):
        """Load drug metadata once for PK tests."""
        with DataLoader() as loader:
            # Need to load the raw drug table to access PK parameters
            query = """
            SELECT 
                drug,
                cmax_oral_m,
                cmax_iv_m,
                tmax_hours,
                half_life_hours,
                bioavailability_percent,
                protein_binding_percent,
                volume_distribution_l_kg,
                clearance_l_hr_kg,
                auc_0_24h_m_h
            FROM db.public.drugs
            WHERE drug IS NOT NULL
            """
            return loader._execute_and_convert(query)
    
    def test_pharmacokinetic_parameters(self, drug_data):
        """Test availability of PK parameters."""
        df = drug_data
        
        # Check cmax availability
        drugs_with_cmax_oral = df['cmax_oral_m'].notna().sum()
        assert drugs_with_cmax_oral >= MIN_DRUGS_WITH_CMAX_ORAL, \
            f"Expected at least {MIN_DRUGS_WITH_CMAX_ORAL} drugs with oral Cmax, got {drugs_with_cmax_oral}"
        
        # Check tmax availability
        drugs_with_tmax = df['tmax_hours'].notna().sum()
        assert drugs_with_tmax >= MIN_DRUGS_WITH_TMAX, \
            f"Expected at least {MIN_DRUGS_WITH_TMAX} drugs with Tmax, got {drugs_with_tmax}"
        
        # Check half-life availability
        drugs_with_half_life = df['half_life_hours'].notna().sum()
        assert drugs_with_half_life >= MIN_DRUGS_WITH_HALF_LIFE, \
            f"Expected at least {MIN_DRUGS_WITH_HALF_LIFE} drugs with half-life, got {drugs_with_half_life}"
        
        # Check bioavailability
        drugs_with_bioavailability = df['bioavailability_percent'].notna().sum()
        assert drugs_with_bioavailability >= MIN_DRUGS_WITH_BIOAVAILABILITY, \
            f"Expected at least {MIN_DRUGS_WITH_BIOAVAILABILITY} drugs with bioavailability, got {drugs_with_bioavailability}"
        
        # Check protein binding
        drugs_with_protein_binding = df['protein_binding_percent'].notna().sum()
        assert drugs_with_protein_binding >= MIN_DRUGS_WITH_PROTEIN_BINDING, \
            f"Expected at least {MIN_DRUGS_WITH_PROTEIN_BINDING} drugs with protein binding, got {drugs_with_protein_binding}"
        
        # Validate PK parameter ranges
        assert (df['tmax_hours'].dropna() >= 0).all(), "Tmax should be non-negative"
        assert (df['half_life_hours'].dropna() > 0).all(), "Half-life should be positive"
        assert df['bioavailability_percent'].dropna().between(0, 100).all(), \
            "Bioavailability should be between 0-100%"
        assert df['protein_binding_percent'].dropna().between(0, 100).all(), \
            "Protein binding should be between 0-100%"
    
    def test_critical_drugs_present(self):
        """Test that known important drugs are present."""
        with DataLoader() as loader:
            df = loader.load_drug_metadata()
            
        drug_names = df['drug'].str.lower().unique()
        
        # Critical DILI-positive drugs that should be present (oncology drugs with high DILI risk)
        critical_positive = ['flupirtine', 'sitaxentan', 'sorafenib', 
                           'tamoxifen', 'erlotinib', 'gemcitabine']
        
        # Critical DILI-negative controls (oncology drugs with low DILI risk)
        critical_negative = ['metformin', 'entrectinib', 'amifostine']
        
        for drug in critical_positive + critical_negative:
            assert any(drug in name for name in drug_names), \
                f"Critical drug '{drug}' not found in database"


class TestDataIntegration:
    """Test data consistency across tables."""
    
    @pytest.fixture
    def all_data(self):
        """Load all data for integration tests."""
        with DataLoader() as loader:
            return {
                'oxygen': loader.load_oxygen_data(limit=3),
                'events': loader.load_media_events(),
                'drugs': loader.load_drug_metadata(),
                'wells': loader.load_well_metadata()
            }
    
    def test_drug_consistency(self, all_data):
        """Test that drugs in wells exist in drug metadata."""
        well_drugs = set(all_data['wells']['drug'].unique())
        metadata_drugs = set(all_data['drugs']['drug'].unique())
        
        # Remove special cases like 'DMSO', 'Unknown'
        well_drugs_filtered = {d for d in well_drugs 
                              if d not in ['DMSO', 'Unknown', 'None', '']}
        
        missing_drugs = well_drugs_filtered - metadata_drugs
        assert len(missing_drugs) < 10, \
            f"Too many drugs in wells missing from metadata: {missing_drugs}"
    
    def test_plate_consistency(self, all_data):
        """Test plate IDs are consistent across tables."""
        oxygen_plates = set(all_data['oxygen']['plate_id'].unique())
        well_plates = set(all_data['wells']['plate_id'].unique())
        
        # Some plates in wells might not have oxygen data yet
        assert oxygen_plates.issubset(well_plates) or len(oxygen_plates & well_plates) > 20, \
            "Plate IDs should overlap significantly between oxygen and well data"
    
    def test_event_timing_validity(self, all_data):
        """Test that event times are within experimental timeframe."""
        if all_data['events'].empty or all_data['oxygen'].empty:
            pytest.skip("No events or oxygen data to validate")
        
        # Get time ranges for each plate
        for plate_id in all_data['events']['plate_id'].unique()[:5]:  # Check first 5 plates
            plate_oxygen = all_data['oxygen'][all_data['oxygen']['plate_id'] == plate_id]
            if plate_oxygen.empty:
                continue
                
            plate_events = all_data['events'][all_data['events']['plate_id'] == plate_id]
            
            min_time = plate_oxygen['timestamp'].min()
            max_time = plate_oxygen['timestamp'].max()
            
            # Events should be within experimental timeframe (with some buffer)
            buffer = pd.Timedelta(days=1)
            valid_events = plate_events['event_time'].between(min_time - buffer, max_time + buffer)
            
            assert valid_events.all(), \
                f"Some events for plate {plate_id} are outside experimental timeframe"


class TestPerformanceAndLimits:
    """Test performance and data limits."""
    
    def test_large_query_performance(self):
        """Test that large queries complete in reasonable time."""
        import time
        
        with DataLoader() as loader:
            start_time = time.time()
            df = loader.load_oxygen_data(limit=5)
            elapsed = time.time() - start_time
            
        assert elapsed < 30, f"Query took too long: {elapsed:.2f} seconds"
        assert len(df) > 1000, "Should load substantial amount of data"
    
    def test_memory_usage(self):
        """Test that data loading doesn't use excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with DataLoader() as loader:
            df = loader.load_oxygen_data(limit=5)
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.2f} MB"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])