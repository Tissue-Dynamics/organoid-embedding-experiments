#!/usr/bin/env python3
"""Find ALL tables with actual data in the current database."""

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

def find_all_tables_with_data():
    """Exhaustively search for ANY tables with data."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info("🔍 Comprehensive search for ANY tables with data...")
    
    # Expanded list of possible table names
    all_possible_tables = [
        # Organoid/experimental tables
        'processed_data', 'raw_data', 'plate_table', 'well_map', 'time_series',
        'experiments', 'measurements', 'readings', 'data', 'organoid_data',
        'oxygen_data', 'well_data', 'sensor_data', 'sonic_data',
        
        # Process-related tables
        'processes', 'process_data', 'processed', 'process_results',
        'dat_table', 'dat_processed', 'processed_results',
        
        # Common database tables
        'users', 'profiles', 'auth', 'sessions', 'logs', 'events',
        'drugs', 'compounds', 'treatments', 'conditions',
        
        # Generic data tables
        'table1', 'table2', 'main', 'primary', 'test', 'temp',
        'export', 'import', 'backup', 'archive', 'staging',
        
        # Sonic-specific possibilities
        'sonic_processed', 'sonic_raw', 'sonic_experiments', 'sonic_results',
        'analysis_data', 'analysis_results', 'sonic_analysis',
        
        # Alternative naming conventions
        'ProcessedData', 'RawData', 'PlateTable', 'WellMap',
        'ExperimentData', 'SensorReadings', 'OxygenMeasurements'
    ]
    
    tables_with_data = []
    
    for table_name in all_possible_tables:
        try:
            # Get count first (faster)
            count_result = loader.client.table(table_name).select('*', count='exact').execute()
            row_count = count_result.count
            
            if row_count > 0:
                logger.info(f"✅ FOUND DATA: {table_name} ({row_count:,} rows)")
                tables_with_data.append((table_name, row_count))
                
                # Get sample data
                sample_result = loader.client.table(table_name).select('*').limit(3).execute()
                if sample_result.data:
                    sample = sample_result.data[0]
                    logger.info(f"   Columns: {list(sample.keys())}")
                    logger.info(f"   Sample: {sample}")
                    
            else:
                logger.debug(f"   {table_name}: exists but empty")
                
        except Exception as e:
            # Table doesn't exist or access denied
            logger.debug(f"   {table_name}: {e}")
    
    return tables_with_data

def analyze_data_table(table_name, row_count):
    """Analyze a specific table that contains data."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info(f"\n📊 Detailed analysis of {table_name} ({row_count:,} rows)...")
    
    try:
        # Get larger sample
        sample_size = min(50, row_count)
        result = loader.client.table(table_name).select('*').limit(sample_size).execute()
        
        if result.data:
            df = pd.DataFrame(result.data)
            logger.info(f"Sample shape: {df.shape}")
            
            # Analyze for time series patterns
            time_patterns = []
            value_patterns = []
            id_patterns = []
            
            for col in df.columns:
                col_lower = col.lower()
                
                # Time-related
                if any(word in col_lower for word in ['time', 'date', 'timestamp', 'created', 'updated', 'measured']):
                    time_patterns.append(col)
                
                # Value-related  
                if any(word in col_lower for word in ['value', 'oxygen', 'o2', 'measurement', 'reading', 'level', 'concentration', 'dose']):
                    value_patterns.append(col)
                    
                # ID-related
                if any(word in col_lower for word in ['id', 'well', 'sample', 'plate', 'row', 'col', 'position']):
                    id_patterns.append(col)
            
            logger.info(f"Time columns: {time_patterns}")
            logger.info(f"Value columns: {value_patterns}")  
            logger.info(f"ID columns: {id_patterns}")
            
            # Check if this looks like organoid time series data
            organoid_score = 0
            if time_patterns: organoid_score += 3
            if value_patterns: organoid_score += 3
            if id_patterns: organoid_score += 2
            if any('oxygen' in col.lower() or 'o2' in col.lower() for col in df.columns): organoid_score += 5
            if any('well' in col.lower() for col in df.columns): organoid_score += 3
            if any('plate' in col.lower() for col in df.columns): organoid_score += 2
            
            logger.info(f"Organoid likelihood score: {organoid_score}/18")
            
            if organoid_score >= 8:
                logger.info("🧬 HIGH PROBABILITY: This looks like organoid experimental data!")
            elif organoid_score >= 5:
                logger.info("🤔 MEDIUM PROBABILITY: Could be relevant experimental data")
            else:
                logger.info("📋 LOW PROBABILITY: Probably not organoid time series data")
                
            # Show sample data
            logger.info(f"Sample records:\n{df.head(3).to_string()}")
            
            return df, organoid_score
            
    except Exception as e:
        logger.error(f"Failed to analyze {table_name}: {e}")
        return None, 0

if __name__ == "__main__":
    logger.info("🔍 Comprehensive search for real organoid data...")
    logger.info("This will check ALL possible table names for data...")
    
    try:
        # Find all tables with data
        tables_with_data = find_all_tables_with_data()
        
        if tables_with_data:
            logger.info(f"\n🎉 Found {len(tables_with_data)} tables with data:")
            for table_name, row_count in tables_with_data:
                logger.info(f"  - {table_name}: {row_count:,} rows")
            
            # Analyze each table for organoid data
            best_candidates = []
            
            for table_name, row_count in tables_with_data:
                df, score = analyze_data_table(table_name, row_count)
                if score >= 5:  # Potential organoid data
                    best_candidates.append((table_name, score, row_count))
            
            if best_candidates:
                best_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by score
                logger.info(f"\n🧬 BEST ORGANOID DATA CANDIDATES:")
                for table_name, score, row_count in best_candidates:
                    logger.info(f"  {table_name}: Score {score}/18, {row_count:,} rows")
                
                best_table = best_candidates[0][0]
                logger.info(f"\n✅ RECOMMENDATION: Use '{best_table}' for organoid experiments")
                
            else:
                logger.info(f"\n❌ No tables appear to contain organoid time series data")
                logger.info("Available tables contain:")
                for table_name, row_count in tables_with_data:
                    logger.info(f"  - {table_name}: {row_count:,} rows (non-organoid data)")
            
        else:
            logger.warning("❌ NO TABLES WITH DATA FOUND!")
            logger.info("This suggests:")
            logger.info("1. Wrong database/project")
            logger.info("2. Data hasn't been uploaded yet") 
            logger.info("3. Different table naming convention")
            logger.info("4. Access permission issues")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise