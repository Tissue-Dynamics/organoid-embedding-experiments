"""
Test the filtered dataset quality.

This tests the clean, filtered dataset to ensure it meets all quality criteria.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.analysis.load_clean_data import (
    load_clean_wells, 
    get_treatment_wells, 
    get_control_wells,
    load_clean_oxygen_data
)
from src.utils.data_loader import DataLoader


class TestFilteredDataQuality:
    """Test quality of filtered dataset."""
    
    def test_filtered_wells_exist(self):
        """Test that filtered wells file exists and loads."""
        wells = load_clean_wells()
        assert wells is not None, "Filtered wells file not found"
        assert isinstance(wells, pd.DataFrame)
        assert not wells.empty
        
    def test_drug_consistency_filtered(self):
        """Test that all drugs in filtered data exist in metadata."""
        wells = get_treatment_wells()
        assert wells is not None
        
        with DataLoader() as loader:
            drugs = loader.load_drug_metadata()
            
        well_drugs = set(wells['drug'].unique())
        metadata_drugs = set(drugs['drug'].unique())
        
        missing_drugs = well_drugs - metadata_drugs
        assert len(missing_drugs) == 0, f"Missing drugs in filtered data: {missing_drugs}"
        
    def test_plate_duration_filtered(self):
        """Test that all filtered plates have sufficient duration."""
        wells = load_clean_wells()
        plate_ids = wells['plate_id'].unique()
        
        # Load oxygen data to check durations
        oxygen_data = load_clean_oxygen_data()
        plate_durations = oxygen_data.groupby('plate_id')['elapsed_hours'].max()
        
        for plate_id in plate_ids:
            duration = plate_durations[plate_id]
            assert duration >= 300, f"Plate {plate_id} duration {duration}h is too short"
            
    def test_media_changes_filtered(self):
        """Test that all filtered plates have media changes."""
        wells = load_clean_wells()
        plate_ids = wells['plate_id'].unique()
        
        with DataLoader() as loader:
            events = loader.load_media_events()
            
        plates_with_media = set(events['plate_id'].unique())
        
        for plate_id in plate_ids:
            assert plate_id in plates_with_media, f"Plate {plate_id} has no media changes"
            
    def test_dili_annotations_complete(self):
        """Test that all treatment wells have DILI annotations."""
        wells = get_treatment_wells()
        
        missing_dili = wells['binary_dili'].isna().sum()
        assert missing_dili == 0, f"{missing_dili} wells missing DILI annotations"
        
    def test_control_wells_identified(self):
        """Test that control wells are properly identified."""
        wells = load_clean_wells()
        
        # All control wells should have concentration = 0
        control_wells = wells[wells['is_control']]
        assert (control_wells['concentration'] == 0).all(), "Control wells with non-zero concentration"
        
        # All treatment wells should have concentration > 0
        treatment_wells = wells[~wells['is_control']]
        assert (treatment_wells['concentration'] > 0).all(), "Treatment wells with zero concentration"
        
    def test_excluded_drugs_removed(self):
        """Test that excluded drug patterns are removed."""
        wells = get_treatment_wells()
        
        # Check for .number suffixes
        numeric_suffix_drugs = [drug for drug in wells['drug'].unique() 
                               if drug and '.' in drug and drug.split('.')[-1].isdigit()]
        assert len(numeric_suffix_drugs) == 0, f"Drugs with numeric suffixes: {numeric_suffix_drugs}"
        
        # Check for (mg/ml) patterns
        mg_ml_drugs = [drug for drug in wells['drug'].unique() 
                      if drug and '(mg/ml)' in drug.lower()]
        assert len(mg_ml_drugs) == 0, f"Drugs with (mg/ml): {mg_ml_drugs}"
        
    def test_minimum_data_available(self):
        """Test that we have sufficient data for analysis."""
        wells = load_clean_wells()
        treatment_wells = get_treatment_wells()
        
        assert len(wells) >= 4000, f"Total wells {len(wells)} too low"
        assert len(treatment_wells) >= 3000, f"Treatment wells {len(treatment_wells)} too low"
        
        # Check DILI balance
        dili_positive = (treatment_wells['binary_dili'] == 1).sum()
        dili_negative = (treatment_wells['binary_dili'] == 0).sum()
        
        assert dili_positive >= 500, f"DILI positive wells {dili_positive} too low"
        assert dili_negative >= 500, f"DILI negative wells {dili_negative} too low"
        
        # Check drug diversity
        unique_drugs = treatment_wells['drug'].nunique()
        assert unique_drugs >= 100, f"Unique drugs {unique_drugs} too low"


class TestFilteredDataStructure:
    """Test structure of filtered dataset."""
    
    def test_required_columns(self):
        """Test that filtered data has required columns."""
        wells = load_clean_wells()
        
        required_cols = [
            'plate_id', 'well_id', 'drug', 'concentration', 
            'is_control', 'binary_dili', 'dili_risk_score'
        ]
        
        for col in required_cols:
            assert col in wells.columns, f"Missing required column: {col}"
            
    def test_data_types(self):
        """Test that data types are correct."""
        wells = load_clean_wells()
        
        assert wells['concentration'].dtype in ['float64', 'int64'], "Concentration should be numeric"
        assert wells['is_control'].dtype == 'bool', "is_control should be boolean"
        # binary_dili can be boolean or numeric (True/False/NaN for controls)
        assert wells['binary_dili'].dtype in ['float64', 'int64', 'bool', 'object'], "binary_dili should be boolean or numeric"
        
    def test_no_null_critical_fields(self):
        """Test that critical fields don't have nulls."""
        wells = load_clean_wells()
        
        # These fields should never be null
        critical_fields = ['plate_id', 'well_id', 'concentration', 'is_control']
        
        for field in critical_fields:
            null_count = wells[field].isnull().sum()
            assert null_count == 0, f"Field {field} has {null_count} null values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])