#!/usr/bin/env python3
"""Test loading real organoid data from Supabase."""

import os
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_supabase_connection():
    """Test basic Supabase connection and explore data structure."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    # Initialize data loader
    loader = SupabaseDataLoader()
    logger.info("✅ Connected to Supabase successfully!")
    
    # Test basic connection by listing available tables
    logger.info("Testing database connection...")
    
    try:
        # Try to get basic info from tables
        logger.info("Checking available data...")
        
        # Get plate info
        plates_query = loader.client.table('plate_table').select('*').limit(5).execute()
        logger.info(f"Found {len(plates_query.data)} plates (showing first 5)")
        
        if plates_query.data:
            plate_df = pd.DataFrame(plates_query.data)
            logger.info(f"Plate columns: {list(plate_df.columns)}")
            logger.info(f"Sample plate data:\n{plate_df.head()}")
            
            # Get first plate ID for testing
            first_plate_id = plates_query.data[0]['id']
            logger.info(f"Testing with first plate ID: {first_plate_id}")
            
            return first_plate_id, loader
        else:
            logger.warning("No plates found in database")
            return None, loader
            
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return None, loader

def test_data_loading(plate_id, loader):
    """Test loading real organoid data."""
    logger.info(f"Loading data for plate: {plate_id}")
    
    try:
        # Get well mapping
        logger.info("Loading well mapping...")
        well_map = loader.get_well_map(plate_id)
        logger.info(f"Well map shape: {well_map.shape}")
        logger.info(f"Well map columns: {list(well_map.columns)}")
        logger.info(f"Sample wells:\n{well_map.head()}")
        
        # Get time series data
        logger.info("Loading time series data...")
        time_series = loader.get_time_series_data(plate_id)
        logger.info(f"Time series shape: {time_series.shape}")
        logger.info(f"Time series columns: {list(time_series.columns)}")
        
        # Check data quality
        logger.info("Analyzing data quality...")
        if 'value' in time_series.columns:
            values = time_series['value']
            logger.info(f"Value range: {values.min():.2f} to {values.max():.2f}")
            logger.info(f"Missing values: {values.isnull().sum()} / {len(values)} ({100*values.isnull().sum()/len(values):.1f}%)")
        
        # Check time coverage
        if 'timestamp' in time_series.columns:
            timestamps = pd.to_datetime(time_series['timestamp'])
            logger.info(f"Time range: {timestamps.min()} to {timestamps.max()}")
            logger.info(f"Duration: {timestamps.max() - timestamps.min()}")
        
        # Check number of wells
        if 'well_number' in time_series.columns:
            n_wells = time_series['well_number'].nunique()
            logger.info(f"Number of wells with data: {n_wells}")
        
        return time_series, well_map
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return None, None

def prepare_matrix_data(time_series, well_map, loader):
    """Test preparing matrix format for embeddings."""
    logger.info("Preparing time series matrix...")
    
    try:
        # Use the loader's method to prepare matrix
        X, metadata_df = loader.prepare_time_series_matrix(
            time_series, 
            well_map,
            fill_method='interpolate',
            normalize=True
        )
        
        logger.info(f"Matrix shape: {X.shape}")
        logger.info(f"Metadata shape: {metadata_df.shape}")
        logger.info(f"Metadata columns: {list(metadata_df.columns)}")
        
        # Check data quality
        logger.info("Matrix data quality:")
        logger.info(f"  NaN values: {np.isnan(X).sum()} / {X.size} ({100*np.isnan(X).sum()/X.size:.1f}%)")
        logger.info(f"  Value range: {np.nanmin(X):.2f} to {np.nanmax(X):.2f}")
        logger.info(f"  Mean: {np.nanmean(X):.2f}, Std: {np.nanstd(X):.2f}")
        
        # Show treatment distribution if available
        if 'treatment' in metadata_df.columns:
            treatment_counts = metadata_df['treatment'].value_counts()
            logger.info(f"Treatment distribution:\n{treatment_counts}")
        
        return X, metadata_df
        
    except Exception as e:
        logger.error(f"Matrix preparation failed: {e}")
        return None, None

def test_embeddings_on_real_data(X, metadata_df):
    """Test a few embedding methods on real data."""
    logger.info("Testing embedding methods on real data...")
    
    from embeddings import (
        FourierEmbedder, CustomFeaturesEmbedder, SAXEmbedder
    )
    
    # Use subset for faster testing
    n_samples = min(100, len(X))
    X_subset = X[:n_samples]
    metadata_subset = metadata_df.iloc[:n_samples]
    
    logger.info(f"Testing with {n_samples} samples...")
    
    # Test fast methods first
    methods = {
        'fourier': FourierEmbedder(n_components=30),
        'custom': CustomFeaturesEmbedder(),
        'sax': SAXEmbedder(n_segments=15, n_symbols=4)
    }
    
    results = {}
    
    for name, embedder in methods.items():
        logger.info(f"Testing {name} on real data...")
        try:
            start_time = time.time()
            embeddings = embedder.fit_transform(X_subset)
            duration = time.time() - start_time
            
            logger.info(f"✅ {name}: {embeddings.shape} in {duration:.2f}s")
            logger.info(f"   Embedding range: {embeddings.min():.3f} to {embeddings.max():.3f}")
            
            results[name] = {
                'embeddings': embeddings,
                'shape': embeddings.shape,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    import time
    
    logger.info("🧬 Testing real organoid data loading...")
    
    try:
        # Test connection
        plate_id, loader = test_supabase_connection()
        
        if plate_id and loader:
            # Test data loading
            time_series, well_map = test_data_loading(plate_id, loader)
            
            if time_series is not None and well_map is not None:
                # Test matrix preparation
                X, metadata_df = prepare_matrix_data(time_series, well_map, loader)
                
                if X is not None and metadata_df is not None:
                    # Test embeddings
                    results = test_embeddings_on_real_data(X, metadata_df)
                    
                    logger.info("\n🎉 REAL DATA TEST SUMMARY:")
                    logger.info(f"  Data shape: {X.shape}")
                    logger.info(f"  Embedding results:")
                    for method, result in results.items():
                        if 'error' in result:
                            logger.info(f"    ❌ {method}: {result['error']}")
                        else:
                            logger.info(f"    ✅ {method}: {result['shape']} in {result['duration']:.2f}s")
                    
                    logger.info("\n✅ Ready to run full experiments on real organoid data!")
                else:
                    logger.error("Failed to prepare matrix data")
            else:
                logger.error("Failed to load time series data")
        else:
            logger.error("Failed to connect to database or find plates")
            
    except Exception as e:
        logger.error(f"❌ Real data test failed: {e}")
        raise