#!/usr/bin/env python3
"""Prepare real oxygen data for embedding experiments."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_plate_data(plate_id, limit_wells=None):
    """Load oxygen data for a specific plate."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    
    # Connect to database
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    
    attach_query = f"""
    ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
    user={parsed.username} password={parsed.password}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    
    conn.execute(attach_query)
    
    # Query for specific plate
    well_limit = f"AND well_number <= {limit_wells}" if limit_wells else ""
    
    query = f"""
    SELECT 
        plate_id,
        well_number,
        timestamp,
        median_o2,
        is_excluded
    FROM supabase.public.processed_data
    WHERE plate_id = '{plate_id}'
    {well_limit}
    AND is_excluded = false
    ORDER BY well_number, timestamp
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    logger.info(f"Loaded {len(df):,} measurements for plate {plate_id}")
    return df

def prepare_time_series_matrix(df):
    """Convert oxygen data to time series matrix format."""
    logger.info("Converting to time series matrix...")
    
    # Pivot data: rows = wells, columns = time points
    pivot_df = df.pivot_table(
        index='well_number',
        columns='timestamp',
        values='median_o2',
        aggfunc='mean'  # Handle any duplicates
    )
    
    logger.info(f"Matrix shape: {pivot_df.shape} (wells × time points)")
    
    # Convert to numpy array
    X = pivot_df.values
    
    # Get metadata
    metadata = pd.DataFrame({
        'well_number': pivot_df.index,
        'n_timepoints': (~np.isnan(X)).sum(axis=1)
    })
    
    # Handle missing values
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        logger.info(f"Missing values: {n_missing:,} ({100*n_missing/X.size:.1f}%)")
        
        # Interpolate missing values
        from scipy.interpolate import interp1d
        
        X_filled = X.copy()
        for i in range(len(X)):
            ts = X[i]
            valid_idx = ~np.isnan(ts)
            
            if valid_idx.sum() > 1:  # Need at least 2 points
                # Interpolate
                x_valid = np.where(valid_idx)[0]
                y_valid = ts[valid_idx]
                
                # Linear interpolation
                f = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
                X_filled[i] = f(np.arange(len(ts)))
            else:
                # Too few points, use median
                X_filled[i] = np.nanmedian(ts)
        
        X = X_filled
        logger.info("✅ Missing values interpolated")
    
    # Get time points for reference
    time_points = pd.to_datetime(pivot_df.columns)
    
    return X, metadata, time_points

def analyze_plate_data(X, metadata, time_points):
    """Analyze the prepared time series data."""
    logger.info("\n📊 Data Analysis:")
    logger.info(f"  Wells: {len(X)}")
    logger.info(f"  Time points: {len(time_points)}")
    logger.info(f"  Duration: {(time_points[-1] - time_points[0]).days} days")
    logger.info(f"  Sampling rate: ~{(time_points[1:] - time_points[:-1]).mean()}")
    
    # Value statistics
    logger.info(f"\nOxygen values (median_o2):")
    logger.info(f"  Range: {X.min():.2f} to {X.max():.2f}")
    logger.info(f"  Mean: {X.mean():.2f}")
    logger.info(f"  Std: {X.std():.2f}")
    
    # Show sample time series
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, well_idx in enumerate([0, 10, 20, 30][:len(X)]):
        ax = axes[i]
        ax.plot(X[well_idx], alpha=0.7)
        ax.set_title(f'Well {metadata.iloc[well_idx]["well_number"]}')
        ax.set_xlabel('Time point')
        ax.set_ylabel('Median O2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oxygen_time_series_samples.png', dpi=150)
    logger.info("Saved sample plots to: oxygen_time_series_samples.png")
    plt.close()
    
    return X, metadata

def run_embedding_experiments(X, metadata):
    """Run embedding methods on the real oxygen data."""
    logger.info("\n🧬 Running embedding experiments on real oxygen data...")
    
    from embeddings import (
        FourierEmbedder, SAXEmbedder, CustomFeaturesEmbedder,
        AutoencoderEmbedder, TransformerEmbedder
    )
    
    # Select methods optimized for real data
    methods = {
        'fourier': FourierEmbedder(n_components=50),
        'sax': SAXEmbedder(n_segments=20, n_symbols=4),
        'custom': CustomFeaturesEmbedder(
            response_window_hours=24,  # 24 hour response window
            sampling_rate_hours=1      # Hourly sampling
        ),
        'autoencoder': AutoencoderEmbedder(
            embedding_dim=32,
            epochs=20,
            batch_size=32,
            architecture='lstm'  # LSTM for time series
        ),
        'transformer': TransformerEmbedder(
            embedding_dim=32,
            epochs=20,
            batch_size=32
        )
    }
    
    results = {}
    
    for name, embedder in methods.items():
        logger.info(f"\nTesting {name} embedder...")
        try:
            import time
            start_time = time.time()
            
            embeddings = embedder.fit_transform(X)
            duration = time.time() - start_time
            
            logger.info(f"✅ {name}: {embeddings.shape} in {duration:.2f}s")
            
            results[name] = {
                'embeddings': embeddings,
                'metadata': metadata,
                'duration': duration,
                'config': embedder.get_config()
            }
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

def save_results(results, plate_id):
    """Save embedding results."""
    import joblib
    
    output_dir = Path('oxygen_embedding_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save each successful embedding
    for method, result in results.items():
        if 'embeddings' in result:
            output_file = output_dir / f'{method}_embeddings_{plate_id}.joblib'
            joblib.dump(result, output_file)
            logger.info(f"Saved {method} results to: {output_file}")

if __name__ == "__main__":
    logger.info("🧬 Preparing real oxygen data for embedding experiments...")
    
    try:
        # Use a specific plate with good data coverage
        # From the analysis, this plate has 182,243 measurements
        plate_id = "c47cadd1-c079-45c0-9863-33d62fb021fa"
        
        # Load data for first 96 wells (one quadrant) for faster testing
        df = load_plate_data(plate_id, limit_wells=96)
        
        if len(df) > 0:
            # Prepare time series matrix
            X, metadata, time_points = prepare_time_series_matrix(df)
            
            # Analyze the data
            X, metadata = analyze_plate_data(X, metadata, time_points)
            
            # Run embedding experiments
            results = run_embedding_experiments(X, metadata)
            
            # Save results
            save_results(results, plate_id)
            
            logger.info("\n✅ SUCCESS! Embedding experiments completed on real oxygen data!")
            
            # Summary
            logger.info("\n📊 RESULTS SUMMARY:")
            for method, result in results.items():
                if 'error' in result:
                    logger.info(f"  {method}: FAILED - {result['error']}")
                else:
                    logger.info(f"  {method}: SUCCESS - {result['embeddings'].shape} in {result['duration']:.2f}s")
            
        else:
            logger.error("No data loaded!")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise