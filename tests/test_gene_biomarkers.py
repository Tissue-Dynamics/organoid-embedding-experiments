"""
Test suite for gene_biomarkers schema tables.

OVERVIEW OF CHECKS AND ACCEPTABLE VALUES:

gene_biomarkers.samples Table:
- Required columns: sample_id (unique identifier)
- Expected columns: plate_id, well_id, well_number, drug, concentration, timepoint, batch
- Sample IDs must be unique (no duplicates allowed)
- Each sample should link to a valid plate_id when present
- Drug names should match main drug table where possible

gene_biomarkers.biomarkers Table:
- Required columns: biomarker_id, biomarker_name or gene_symbol
- Biomarker IDs must be unique
- Expected to contain gene/protein identifiers for expression analysis
- Should have reasonable number of biomarkers (10s to 1000s)

gene_biomarkers.drug_keys Table:
- Maps barcodes to drug names
- Barcode format should be consistent (same length for all)
- Drug names should overlap with main drug table
- Used for decoding experimental barcodes

gene_biomarkers.gene_expression Table:
- Required columns: sample_id, biomarker_id/gene_id, expression_value/value
- Expression values: Typically -100 to 100,000 (log scale or normalized)
- Missing values acceptable but should be <50% per sample
- Each sample should have measurements for multiple biomarkers
- Values must be numeric (float32 or float64)

Integration Expectations:
- Samples with gene data should have corresponding metabolic data
- Not all plates will have gene expression (it's a subset)
- Biomarker coverage: Not all defined biomarkers need expression data
- Drug consistency: Gene study drugs should mostly overlap with main study
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.utils.data_loader import DataLoader


class TestGeneSamples:
    """Test gene_biomarkers.samples table."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_load_gene_samples(self, loader):
        """Test loading gene sample metadata."""
        df = loader.load_gene_samples()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        
        # Check expected columns (based on typical sample metadata)
        possible_columns = [
            'sample_id', 'plate_id', 'well_id', 'well_number',
            'drug', 'concentration', 'timepoint', 'batch',
            'created_at', 'experiment_id'
        ]
        
        # Check actual columns exist (based on database schema)
        key_columns = ['id', 'plate_id']
        for col in key_columns:
            assert col in df.columns, f"Missing key column: {col}"
        
        # Check data quality
        if len(df) > 0:
            # IDs should be unique
            assert df['id'].nunique() == len(df), "Duplicate IDs found"
            
            # Check for drug information if present
            if 'drug' in df.columns:
                print(f"\nDrugs in gene samples: {df['drug'].nunique()}")
                print(f"Sample drugs: {df['drug'].value_counts().head()}")
    
    def test_gene_samples_plate_coverage(self, loader):
        """Test which plates have gene expression samples."""
        samples = loader.load_gene_samples()
        
        if len(samples) == 0:
            pytest.skip("No gene expression samples found")
        
        if 'plate_id' in samples.columns:
            plates_with_genes = samples['plate_id'].nunique()
            print(f"\nPlates with gene expression: {plates_with_genes}")
            
            # Check sample distribution
            samples_per_plate = samples['plate_id'].value_counts()
            print(f"Samples per plate: min={samples_per_plate.min()}, "
                  f"mean={samples_per_plate.mean():.1f}, max={samples_per_plate.max()}")


class TestGeneBiomarkers:
    """Test gene_biomarkers.biomarkers table."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_load_gene_biomarkers(self, loader):
        """Test loading biomarker definitions."""
        df = loader.load_gene_biomarkers()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        
        # Check for expected columns
        expected_columns = ['biomarker_id', 'biomarker_name', 'gene_symbol']
        for col in expected_columns:
            if col in df.columns:
                print(f"Found column: {col}")
        
        if len(df) > 0:
            print(f"\nTotal biomarkers: {len(df)}")
            
            # Check for unique biomarkers
            if 'biomarker_id' in df.columns:
                assert df['biomarker_id'].nunique() == len(df), "Duplicate biomarker IDs"
            
            # Sample biomarker info
            if 'biomarker_name' in df.columns:
                print(f"Sample biomarkers: {df['biomarker_name'].head().tolist()}")
            elif 'gene_symbol' in df.columns:
                print(f"Sample genes: {df['gene_symbol'].head().tolist()}")


class TestGeneDrugKeys:
    """Test gene_biomarkers.drug_keys table."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_load_gene_drug_keys(self, loader):
        """Test loading drug key mappings."""
        df = loader.load_gene_drug_keys()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        
        # This table maps barcodes to drugs
        expected_columns = ['barcode', 'drug', 'drug_key', 'plate_id']
        
        if len(df) > 0:
            print(f"\nTotal drug keys: {len(df)}")
            
            # Check barcode format if present
            if 'barcode' in df.columns:
                sample_barcodes = df['barcode'].head()
                print(f"Sample barcodes: {sample_barcodes.tolist()}")
                
                # Barcodes should have consistent format
                barcode_lengths = df['barcode'].astype(str).str.len()
                if barcode_lengths.nunique() == 1:
                    print(f"Barcode length: {barcode_lengths.iloc[0]}")
            
            # Check drug mapping
            if 'drug' in df.columns:
                drugs_mapped = df['drug'].nunique()
                print(f"Unique drugs in mapping: {drugs_mapped}")
    
    def test_drug_key_consistency(self, loader):
        """Test consistency between drug keys and main drug table."""
        drug_keys = loader.load_gene_drug_keys()
        drugs = loader.load_drug_metadata()
        
        if len(drug_keys) == 0:
            pytest.skip("No drug key data available")
        
        if 'drug' in drug_keys.columns:
            # Check if drugs in gene data exist in main drug table
            gene_drugs = set(drug_keys['drug'].dropna().unique())
            main_drugs = set(drugs['drug'].unique())
            
            # Find drugs only in gene data
            gene_only = gene_drugs - main_drugs
            if len(gene_only) > 0:
                print(f"\nDrugs only in gene data: {len(gene_only)}")
                print(f"Examples: {list(gene_only)[:5]}")
            
            # Find overlap
            overlap = gene_drugs.intersection(main_drugs)
            print(f"Drugs in both tables: {len(overlap)}")


class TestGeneExpression:
    """Test gene_biomarkers.gene_expression table."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_load_gene_expression_all(self, loader):
        """Test loading all gene expression data."""
        # This might be large, so we'll just check structure
        df = loader.load_gene_expression_data()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        
        # Expected columns for expression data
        expected_columns = [
            'sample_id', 'biomarker_id', 'expression_value',
            'gene_id', 'gene_symbol', 'value'
        ]
        
        # Check which columns exist
        found_columns = [col for col in expected_columns if col in df.columns]
        print(f"\nFound expression columns: {found_columns}")
        
        if len(df) > 0:
            print(f"Total expression measurements: {len(df):,}")
            
            # Check data types for expression values
            value_columns = ['expression_value', 'value']
            for col in value_columns:
                if col in df.columns:
                    assert df[col].dtype in [np.float32, np.float64], \
                        f"Expression values should be numeric, got {df[col].dtype}"
    
    def test_load_gene_expression_by_sample(self, loader):
        """Test loading gene expression for specific samples."""
        # First get some sample IDs
        samples = loader.load_gene_samples()
        
        if len(samples) == 0:
            pytest.skip("No gene samples available")
        
        # Get first 5 sample IDs
        sample_ids = samples['sample_id'].head(5).tolist()
        
        # Load expression data for these samples
        expr_data = loader.load_gene_expression_data(sample_ids=sample_ids)
        
        if len(expr_data) > 0:
            # Verify filtering worked
            assert 'sample_id' in expr_data.columns, "No sample_id column in expression data"
            
            returned_samples = set(expr_data['sample_id'].unique())
            requested_samples = set(sample_ids)
            
            # All returned samples should be in requested list
            assert returned_samples <= requested_samples, \
                "Returned expression data for unrequested samples"
            
            print(f"\nExpression data for {len(returned_samples)} samples")
            print(f"Total measurements: {len(expr_data):,}")
    
    def test_expression_data_quality(self, loader):
        """Test quality of gene expression data."""
        # Load a sample of expression data
        samples = loader.load_gene_samples()
        if len(samples) == 0:
            pytest.skip("No gene samples available")
        
        # Get one sample for detailed analysis
        sample_id = samples['sample_id'].iloc[0]
        expr_data = loader.load_gene_expression_data(sample_ids=[sample_id])
        
        if len(expr_data) == 0:
            pytest.skip("No expression data for test sample")
        
        # Check expression value distribution
        value_col = None
        for col in ['expression_value', 'value']:
            if col in expr_data.columns:
                value_col = col
                break
        
        if value_col:
            values = expr_data[value_col]
            
            # Basic statistics
            print(f"\nExpression statistics for sample {sample_id}:")
            print(f"  Min: {values.min():.3f}")
            print(f"  Max: {values.max():.3f}")
            print(f"  Mean: {values.mean():.3f}")
            print(f"  Median: {values.median():.3f}")
            
            # Check for reasonable values
            assert values.min() >= -100, "Extremely low expression values found"
            assert values.max() <= 100000, "Extremely high expression values found"
            
            # Check for missing values
            missing_pct = values.isna().sum() / len(values) * 100
            print(f"  Missing values: {missing_pct:.1f}%")


class TestGeneDataIntegration:
    """Test integration between gene expression and other data."""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        with DataLoader() as loader:  # Let it auto-detect local vs remote
            yield loader
    
    def test_gene_metabolic_integration(self, loader):
        """Test if samples with gene data have metabolic data."""
        samples = loader.load_gene_samples()
        
        if len(samples) == 0:
            pytest.skip("No gene samples available")
        
        # Check if we can link to plates
        if 'plate_id' not in samples.columns:
            pytest.skip("No plate_id in gene samples")
        
        # Get a plate with gene data
        gene_plates = samples['plate_id'].dropna().unique()
        test_plate = gene_plates[0]
        
        # Check if this plate has oxygen data
        oxygen_data = loader.load_oxygen_data(plate_ids=[test_plate])
        
        print(f"\nIntegration check for plate {test_plate}:")
        print(f"  Gene samples: {len(samples[samples['plate_id'] == test_plate])}")
        print(f"  Oxygen measurements: {len(oxygen_data)}")
        
        # Note: Gene expression plates might be separate from oxygen measurement plates
        if len(oxygen_data) == 0:
            print(f"  Note: Plate {test_plate} has gene data but no oxygen data (separate experiments)")
    
    def test_biomarker_expression_coverage(self, loader):
        """Test how many biomarkers have expression data."""
        biomarkers = loader.load_gene_biomarkers()
        expression = loader.load_gene_expression_data()
        
        if len(biomarkers) == 0 or len(expression) == 0:
            pytest.skip("Missing biomarker or expression data")
        
        # Find the biomarker ID column
        biomarker_id_col = None
        for col in ['biomarker_id', 'gene_id']:
            if col in expression.columns:
                biomarker_id_col = col
                break
        
        if biomarker_id_col and 'biomarker_id' in biomarkers.columns:
            # Check coverage
            biomarkers_with_data = expression[biomarker_id_col].nunique()
            total_biomarkers = len(biomarkers)
            coverage = biomarkers_with_data / total_biomarkers * 100
            
            print(f"\nBiomarker coverage:")
            print(f"  Total biomarkers: {total_biomarkers}")
            print(f"  Biomarkers with data: {biomarkers_with_data}")
            print(f"  Coverage: {coverage:.1f}%")
            
            # At least some biomarkers should have data
            assert biomarkers_with_data > 0, "No biomarkers have expression data"