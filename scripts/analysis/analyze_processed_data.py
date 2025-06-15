#!/usr/bin/env python3
"""Analyze the processed_data table with real organoid experimental data."""

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

def analyze_processed_data_table():
    """Analyze the processed_data table structure and real organoid data."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info("🧬 Analyzing processed_data table with real organoid data...")
    
    try:
        # Check table exists and get row count
        count_result = loader.client.table('processed_data').select('*', count='exact').execute()
        total_rows = count_result.count
        logger.info(f"✅ processed_data table found with {total_rows:,} rows")
        
        if total_rows == 0:
            logger.warning("❌ processed_data table is empty!")
            return None
        
        # Get sample data to understand structure
        logger.info("📊 Getting sample data...")
        sample_result = loader.client.table('processed_data').select('*').limit(20).execute()
        
        if not sample_result.data:
            logger.warning("❌ No data returned from processed_data table")
            return None
        
        df = pd.DataFrame(sample_result.data)
        logger.info(f"✅ Sample data shape: {df.shape}")
        logger.info(f"✅ Columns ({len(df.columns)}): {list(df.columns)}")
        
        # Analyze column types and patterns
        logger.info("\n🔬 Column Analysis:")
        for col in df.columns:
            sample_values = df[col].dropna().head(3).tolist()
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            logger.info(f"  {col}:")
            logger.info(f"    - Type: {df[col].dtype}")
            logger.info(f"    - Unique values: {unique_count}")
            logger.info(f"    - Null values: {null_count}")
            logger.info(f"    - Sample: {sample_values}")
        
        # Look for organoid-specific patterns
        logger.info("\n🎯 Organoid Data Pattern Analysis:")
        
        # Time-related columns
        time_cols = [col for col in df.columns if any(word in col.lower() for word in ['time', 'date', 'timestamp', 'hour', 'minute', 'created_at', 'measured_at'])]
        logger.info(f"  Time columns: {time_cols}")
        
        # Value/measurement columns  
        value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'oxygen', 'o2', 'measurement', 'reading', 'signal', 'consumption', 'level'])]
        logger.info(f"  Value columns: {value_cols}")
        
        # Identifier columns
        id_cols = [col for col in df.columns if any(word in col.lower() for word in ['well', 'sample', 'id', 'row', 'col', 'plate', 'position'])]
        logger.info(f"  ID columns: {id_cols}")
        
        # Treatment columns
        treatment_cols = [col for col in df.columns if any(word in col.lower() for word in ['treatment', 'drug', 'compound', 'condition', 'dose', 'concentration', 'control'])]
        logger.info(f"  Treatment columns: {treatment_cols}")
        
        # Show sample data
        logger.info(f"\n📋 Sample processed_data records:")
        logger.info(df.head().to_string())
        
        # Analyze data distribution
        if value_cols:
            main_value_col = value_cols[0]
            values = df[main_value_col].dropna()
            if len(values) > 0:
                logger.info(f"\n📈 {main_value_col} statistics:")
                logger.info(f"  Range: {values.min():.3f} to {values.max():.3f}")
                logger.info(f"  Mean: {values.mean():.3f}")
                logger.info(f"  Std: {values.std():.3f}")
        
        # Check for time series structure
        if time_cols and id_cols:
            time_col = time_cols[0]
            id_col = id_cols[0]
            
            # Check how many unique time points per ID
            time_points_per_id = df.groupby(id_col)[time_col].count()
            logger.info(f"\n⏰ Time series structure:")
            logger.info(f"  Unique {id_col}s: {df[id_col].nunique()}")
            logger.info(f"  Time points per {id_col}: {time_points_per_id.describe()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze processed_data table: {e}")
        return None

def check_data_quality(df):
    """Check data quality issues in the organoid data."""
    if df is None:
        return
        
    logger.info("\n🔍 Data Quality Assessment:")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    total_rows = len(df)
    
    logger.info("Missing data by column:")
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            percentage = (missing_count / total_rows) * 100
            logger.info(f"  {col}: {missing_count}/{total_rows} ({percentage:.1f}%)")
    
    # Check for duplicate records
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows: {duplicates}")
    
    # Data type consistency
    logger.info("Data type issues:")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric data is stored as string
            sample_vals = df[col].dropna().head(10)
            if len(sample_vals) > 0:
                try:
                    pd.to_numeric(sample_vals)
                    logger.info(f"  {col}: Appears to be numeric but stored as text")
                except:
                    pass

def suggest_data_preparation():
    """Suggest how to prepare the data for embedding experiments."""
    logger.info("\n💡 Data Preparation Suggestions:")
    logger.info("1. Time Series Matrix: Convert to (n_samples, n_timepoints) format")
    logger.info("2. Missing Data: Handle gaps with interpolation or forward-fill")
    logger.info("3. Normalization: Scale values for fair comparison across wells")
    logger.info("4. Metadata: Extract treatment, concentration, replicate info")
    logger.info("5. Quality Control: Remove wells with >50% missing data")
    logger.info("6. Time Alignment: Ensure consistent time sampling")

if __name__ == "__main__":
    logger.info("🧬 Analyzing real organoid data in processed_data table...")
    
    try:
        # Analyze the processed_data table
        df = analyze_processed_data_table()
        
        if df is not None:
            # Check data quality
            check_data_quality(df)
            
            # Suggest preparation steps
            suggest_data_preparation()
            
            logger.info(f"\n🎉 SUCCESS! Found real organoid data!")
            logger.info(f"✅ {len(df)} sample records analyzed")
            logger.info(f"✅ Ready to adapt data loader for your format")
            logger.info(f"✅ Can proceed with embedding experiments")
            
        else:
            logger.error("❌ Could not access processed_data table")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise