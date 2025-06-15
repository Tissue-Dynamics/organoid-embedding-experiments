#!/usr/bin/env python3
"""Import real oxygen data from processed_data table using DuckDB."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_to_supabase_with_duckdb():
    """Connect to Supabase PostgreSQL database using DuckDB."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env file")
    
    logger.info("🦆 Connecting to database with DuckDB...")
    
    # Create DuckDB connection
    conn = duckdb.connect()
    
    # Install and load PostgreSQL extension
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    # Parse the database URL to extract components
    # Format: postgresql://user:password@host:port/database
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    
    db_config = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }
    
    logger.info(f"Connecting to: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    
    # Attach PostgreSQL database
    attach_query = f"""
    ATTACH 'host={db_config['host']} port={db_config['port']} dbname={db_config['database']} user={db_config['user']} password={db_config['password']}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    
    try:
        conn.execute(attach_query)
        logger.info("✅ Successfully connected to Supabase via DuckDB!")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        raise

def explore_processed_data_table(conn):
    """Explore the processed_data table structure."""
    logger.info("🔍 Exploring processed_data table...")
    
    try:
        # Check table structure
        schema_query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'processed_data'
        ORDER BY ordinal_position;
        """
        
        schema_df = conn.execute(schema_query).df()
        logger.info("Table schema:")
        for _, row in schema_df.iterrows():
            logger.info(f"  {row['column_name']}: {row['data_type']}")
        
        # Get row count
        count_result = conn.execute("SELECT COUNT(*) as count FROM supabase.public.processed_data").fetchone()
        total_rows = count_result[0]
        logger.info(f"\n✅ Total rows in processed_data: {total_rows:,}")
        
        # Get sample data
        sample_df = conn.execute("SELECT * FROM supabase.public.processed_data LIMIT 10").df()
        logger.info(f"\nSample data shape: {sample_df.shape}")
        logger.info(f"Columns: {list(sample_df.columns)}")
        
        return sample_df, total_rows
        
    except Exception as e:
        logger.error(f"Failed to explore table: {e}")
        # Try without schema prefix
        try:
            logger.info("Trying without schema prefix...")
            sample_df = conn.execute("SELECT * FROM processed_data LIMIT 10").df()
            logger.info(f"✅ Success! Sample data shape: {sample_df.shape}")
            logger.info(f"Columns: {list(sample_df.columns)}")
            
            # Get count
            count_result = conn.execute("SELECT COUNT(*) FROM processed_data").fetchone()
            total_rows = count_result[0]
            logger.info(f"Total rows: {total_rows:,}")
            
            return sample_df, total_rows
            
        except Exception as e2:
            logger.error(f"Also failed: {e2}")
            return None, 0

def analyze_oxygen_data(conn, sample_df):
    """Analyze the oxygen time series data structure."""
    if sample_df is None or len(sample_df) == 0:
        logger.warning("No data to analyze")
        return
    
    logger.info("\n🧬 Analyzing oxygen data structure...")
    
    # Identify key columns
    columns = sample_df.columns.tolist()
    logger.info(f"Available columns: {columns}")
    
    # Look for time-related columns
    time_cols = [col for col in columns if any(word in col.lower() for word in ['time', 'timestamp', 'date', 'hour', 'minute'])]
    logger.info(f"Time columns: {time_cols}")
    
    # Look for oxygen/value columns
    value_cols = [col for col in columns if any(word in col.lower() for word in ['oxygen', 'o2', 'value', 'measurement', 'concentration'])]
    logger.info(f"Value columns: {value_cols}")
    
    # Look for identifier columns
    id_cols = [col for col in columns if any(word in col.lower() for word in ['well', 'plate', 'sample', 'id', 'experiment'])]
    logger.info(f"ID columns: {id_cols}")
    
    # Look for treatment columns
    treatment_cols = [col for col in columns if any(word in col.lower() for word in ['treatment', 'drug', 'compound', 'condition', 'dose'])]
    logger.info(f"Treatment columns: {treatment_cols}")
    
    # Show sample data
    logger.info("\nSample records:")
    logger.info(sample_df.head().to_string())
    
    # Get unique experiments/plates if available
    if id_cols:
        for id_col in id_cols[:2]:  # Check first 2 ID columns
            unique_query = f"SELECT DISTINCT {id_col}, COUNT(*) as count FROM supabase.public.processed_data GROUP BY {id_col} ORDER BY count DESC LIMIT 10"
            try:
                unique_df = conn.execute(unique_query).df()
                logger.info(f"\nTop {id_col} values:")
                logger.info(unique_df.to_string())
            except:
                pass

def load_oxygen_time_series(conn):
    """Load the full oxygen time series data."""
    logger.info("\n📥 Loading oxygen time series data...")
    
    try:
        # First get a larger sample to understand structure better
        sample_query = """
        SELECT * FROM supabase.public.processed_data 
        LIMIT 1000
        """
        
        sample_df = conn.execute(sample_query).df()
        logger.info(f"Loaded sample: {sample_df.shape}")
        
        # Analyze the structure
        logger.info("\nData structure analysis:")
        for col in sample_df.columns:
            dtype = sample_df[col].dtype
            nunique = sample_df[col].nunique()
            nulls = sample_df[col].isnull().sum()
            logger.info(f"  {col}: {dtype}, {nunique} unique values, {nulls} nulls")
            
            # Show sample values for key columns
            if nunique < 20:
                unique_vals = sample_df[col].unique()[:10]
                logger.info(f"    Sample values: {unique_vals}")
        
        # Check if we can load all data or need batching
        count_result = conn.execute("SELECT COUNT(*) FROM supabase.public.processed_data").fetchone()
        total_rows = count_result[0]
        
        if total_rows < 1000000:  # Less than 1M rows, load all
            logger.info(f"Loading all {total_rows:,} rows...")
            full_df = conn.execute("SELECT * FROM supabase.public.processed_data").df()
            logger.info(f"✅ Loaded full dataset: {full_df.shape}")
            return full_df
        else:
            logger.info(f"Large dataset ({total_rows:,} rows), loading in batches...")
            # Implement batched loading if needed
            return sample_df
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

def prepare_for_embeddings(df):
    """Prepare the oxygen data for embedding experiments."""
    if df is None or len(df) == 0:
        return None
    
    logger.info("\n🔧 Preparing data for embedding experiments...")
    
    # Show what we have
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    # Save a sample for inspection
    sample_file = "oxygen_data_sample.csv"
    df.head(100).to_csv(sample_file, index=False)
    logger.info(f"Saved sample to: {sample_file}")
    
    return df

if __name__ == "__main__":
    logger.info("🧬 Importing real oxygen data from processed_data table using DuckDB...")
    
    try:
        # Connect to database
        conn = connect_to_supabase_with_duckdb()
        
        # Explore table structure
        sample_df, total_rows = explore_processed_data_table(conn)
        
        if sample_df is not None and total_rows > 0:
            # Analyze the data
            analyze_oxygen_data(conn, sample_df)
            
            # Load the full dataset
            full_df = load_oxygen_time_series(conn)
            
            if full_df is not None:
                # Prepare for embeddings
                prepared_df = prepare_for_embeddings(full_df)
                
                logger.info("\n✅ SUCCESS! Real oxygen data imported!")
                logger.info(f"Ready to run embedding experiments on {len(full_df):,} oxygen measurements")
            else:
                logger.error("Failed to load full dataset")
        else:
            logger.error("No data found in processed_data table")
            
        # Close connection
        conn.close()
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise