#!/usr/bin/env python3
"""Find and analyze the processes dat table with real organoid data."""

import logging
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_processes_table():
    """Find the processes dat table in the current or sonic project."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info("🔍 Searching for processes dat table...")
    
    # Possible table name variations
    possible_names = [
        'processes dat table',
        'processes_dat_table', 
        'processesDatTable',
        'processes-dat-table',
        'processes dat',
        'processes_dat',
        'processes',
        'process_dat',
        'sonic_processes',
        'organoid_processes',
        'dat_table',
        'data_table'
    ]
    
    found_tables = []
    
    for table_name in possible_names:
        try:
            logger.info(f"Checking: {table_name}")
            result = loader.client.table(table_name).select('*').limit(1).execute()
            
            if hasattr(result, 'data') and result.data:
                logger.info(f"✅ FOUND TABLE WITH DATA: {table_name}")
                found_tables.append(table_name)
                
                # Get sample data
                sample = result.data[0]
                logger.info(f"   Columns: {list(sample.keys())}")
                logger.info(f"   Sample values: {sample}")
                
                # Get row count
                try:
                    count_result = loader.client.table(table_name).select('*', count='exact').execute()
                    logger.info(f"   Row count: {count_result.count}")
                except:
                    logger.info(f"   Row count: Unknown")
                    
            elif hasattr(result, 'data'):
                logger.info(f"   Table {table_name} exists but is empty")
                
        except Exception as e:
            # Table doesn't exist, continue searching
            logger.debug(f"   {table_name}: {e}")
    
    return found_tables

def analyze_processes_table(table_name):
    """Analyze the processes dat table structure and data."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info(f"📊 Analyzing table: {table_name}")
    
    try:
        # Get sample data
        result = loader.client.table(table_name).select('*').limit(10).execute()
        
        if result.data:
            df = pd.DataFrame(result.data)
            logger.info(f"✅ Table shape: {df.shape}")
            logger.info(f"✅ Columns: {list(df.columns)}")
            
            # Analyze columns for organoid data patterns
            logger.info("\n🔬 Data Analysis:")
            for col in df.columns:
                sample_values = df[col].dropna().head(3).tolist()
                data_type = df[col].dtype
                logger.info(f"  {col}: {data_type} - {sample_values}")
            
            # Look for time series patterns
            time_cols = [col for col in df.columns if any(word in col.lower() for word in ['time', 'date', 'timestamp', 'hour', 'minute'])]
            value_cols = [col for col in df.columns if any(word in col.lower() for word in ['value', 'oxygen', 'o2', 'measurement', 'reading', 'signal'])]
            id_cols = [col for col in df.columns if any(word in col.lower() for word in ['well', 'sample', 'id', 'row', 'col', 'plate'])]
            treatment_cols = [col for col in df.columns if any(word in col.lower() for word in ['treatment', 'drug', 'compound', 'condition', 'dose'])]
            
            logger.info(f"\n🎯 Potential organoid data columns:")
            logger.info(f"  Time columns: {time_cols}")
            logger.info(f"  Value columns: {value_cols}")
            logger.info(f"  ID columns: {id_cols}")
            logger.info(f"  Treatment columns: {treatment_cols}")
            
            # Show sample data
            logger.info(f"\n📋 Sample data:")
            logger.info(df.head().to_string())
            
            return df
            
    except Exception as e:
        logger.error(f"❌ Failed to analyze table: {e}")
        return None

def check_sonic_project_connection():
    """Use existing credentials - they should point to sonic analysis project."""
    logger.info("\n🔗 Re-checking with existing credentials (should be sonic project)...")
    
    # The existing SUPABASE_URL and SUPABASE_KEY should be for sonic project
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info(f"Using existing connection to search for processes dat table...")
    
    # Try exact table name variations for processes dat table
    exact_variations = [
        'processes dat table',
        'processes_dat_table',
        'ProcessesDatTable', 
        'processes-dat-table',
        'PROCESSES_DAT_TABLE',
        'processesdattable'
    ]
    
    for table_name in exact_variations:
        try:
            logger.info(f"Trying exact table name: '{table_name}'")
            result = loader.client.table(table_name).select('*').limit(1).execute()
            
            if hasattr(result, 'data') and result.data is not None:
                logger.info(f"🎉 FOUND! Table '{table_name}' exists!")
                
                if result.data:  # Has data
                    logger.info(f"✅ Table has data: {len(result.data)} sample rows")
                    return table_name, loader
                else:  # Empty but exists
                    logger.info(f"⚠️  Table '{table_name}' exists but is empty")
                    
        except Exception as e:
            logger.debug(f"Table '{table_name}' not found: {e}")
    
    return None, None

if __name__ == "__main__":
    logger.info("🧬 Searching for real organoid data in processes dat table...")
    
    try:
        # First, search in current database
        found_tables = find_processes_table()
        
        if found_tables:
            logger.info(f"\n🎉 Found {len(found_tables)} potential tables:")
            for table in found_tables:
                logger.info(f"  - {table}")
                
            # Analyze the first found table
            main_table = found_tables[0]
            df = analyze_processes_table(main_table)
            
            if df is not None:
                logger.info(f"\n✅ SUCCESS! Found real organoid data in '{main_table}'")
                logger.info("Ready to adapt data loader and run embedding experiments!")
            
        else:
            logger.info("\n❌ No processes dat table found in current database")
            
            # Check for sonic project connection
            sonic_client = check_sonic_project_connection()
            
            if sonic_client:
                logger.info("Checking sonic project for processes dat table...")
                # Would need to adapt the search for sonic client
            else:
                logger.info("\n💡 NEXT STEPS:")
                logger.info("1. Verify the exact table name")
                logger.info("2. Check if data is in a different Supabase project")
                logger.info("3. Provide sonic analysis project credentials")
                logger.info("4. Or confirm the table location")
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise