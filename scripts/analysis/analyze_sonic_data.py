#!/usr/bin/env python3
"""
Analyze real organoid data from the sonic analysis project
Examine the "processes dat table" structure and contents
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SonicAnalysisDataLoader:
    """Load organoid data from the sonic analysis project"""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client for sonic analysis project"""
        load_dotenv()
        
        # Try to use environment variables, or use same credentials but different table structure
        self.url = url or os.getenv('SONIC_SUPABASE_URL') or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SONIC_SUPABASE_KEY') or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided for sonic analysis project")
            
        self.client: Client = create_client(self.url, self.key)
        logger.info(f"Connected to Sonic Analysis Supabase: {self.url}")
    
    def explore_database_structure(self) -> Dict[str, List[str]]:
        """Explore the database structure to find available tables"""
        logger.info("Exploring database structure...")
        
        # Try to get table information from information_schema
        try:
            # First, let's try to see what tables exist
            result = self.client.rpc('get_table_names').execute()
            logger.info("Available tables:")
            for table in result.data:
                logger.info(f"  - {table}")
            return {"tables": result.data}
        except Exception as e:
            logger.warning(f"Could not get table names via RPC: {e}")
            
        # Alternative approach: try common table names
        common_tables = [
            "processes_dat_table", 
            "processes dat table",
            "processes_data",
            "organoid_data",
            "time_series_data",
            "experimental_data"
        ]
        
        existing_tables = []
        for table in common_tables:
            try:
                # Try to get first row to see if table exists
                result = self.client.table(table).select("*").limit(1).execute()
                existing_tables.append(table)
                logger.info(f"Found table: {table}")
            except Exception as e:
                logger.debug(f"Table {table} not found: {e}")
                
        return {"existing_tables": existing_tables}
    
    def analyze_processes_dat_table(self) -> Dict:
        """Analyze the processes dat table structure and contents"""
        logger.info("Analyzing processes dat table...")
        
        # Try different possible table names
        possible_names = [
            "processes_dat_table",
            "processes dat table", 
            "processes_data",
            "processes_dat",
            "process_data"
        ]
        
        table_name = None
        for name in possible_names:
            try:
                # Try to access the table
                result = self.client.table(name).select("*").limit(1).execute()
                table_name = name
                logger.info(f"Found processes table: {name}")
                break
            except Exception as e:
                logger.debug(f"Table {name} not accessible: {e}")
        
        if not table_name:
            logger.error("Could not find processes dat table")
            return {"error": "Processes dat table not found"}
        
        analysis = {"table_name": table_name}
        
        # Get table schema/structure
        try:
            # Get a few sample rows to understand structure
            sample_result = self.client.table(table_name).select("*").limit(10).execute()
            sample_data = pd.DataFrame(sample_result.data)
            
            if sample_data.empty:
                logger.warning("Processes dat table appears to be empty")
                return {"error": "Table is empty", "table_name": table_name}
            
            logger.info(f"Sample data shape: {sample_data.shape}")
            logger.info(f"Columns: {list(sample_data.columns)}")
            
            analysis.update({
                "sample_shape": sample_data.shape,
                "columns": list(sample_data.columns),
                "sample_data": sample_data.head().to_dict('records'),
                "column_types": sample_data.dtypes.to_dict()
            })
            
            # Get total row count
            try:
                count_result = self.client.table(table_name).select("*", count="exact").execute()
                total_rows = count_result.count
                logger.info(f"Total rows in table: {total_rows}")
                analysis["total_rows"] = total_rows
            except Exception as e:
                logger.warning(f"Could not get row count: {e}")
            
            # Analyze time-related columns
            time_columns = [col for col in sample_data.columns if any(time_word in col.lower() for time_word in ['time', 'date', 'timestamp', 'hour', 'day'])]
            if time_columns:
                logger.info(f"Time-related columns found: {time_columns}")
                analysis["time_columns"] = time_columns
                
                # Try to parse timestamps
                for col in time_columns:
                    try:
                        sample_data[col] = pd.to_datetime(sample_data[col])
                        time_range = f"{sample_data[col].min()} to {sample_data[col].max()}"
                        logger.info(f"Time range in {col}: {time_range}")
                        analysis[f"{col}_range"] = time_range
                    except Exception as e:
                        logger.debug(f"Could not parse {col} as datetime: {e}")
            
            # Analyze potential organoid/well identifiers
            id_columns = [col for col in sample_data.columns if any(id_word in col.lower() for id_word in ['id', 'well', 'organoid', 'sample', 'plate', 'position'])]
            if id_columns:
                logger.info(f"ID/identifier columns found: {id_columns}")
                analysis["id_columns"] = id_columns
                
                for col in id_columns:
                    unique_count = sample_data[col].nunique()
                    logger.info(f"Unique values in {col}: {unique_count}")
                    analysis[f"{col}_unique_count"] = unique_count
            
            # Analyze measurement columns (likely oxygen or other sensor data)
            numeric_columns = sample_data.select_dtypes(include=[np.number]).columns.tolist()
            measurement_columns = [col for col in numeric_columns if col not in id_columns + time_columns]
            if measurement_columns:
                logger.info(f"Potential measurement columns: {measurement_columns}")
                analysis["measurement_columns"] = measurement_columns
                
                for col in measurement_columns:
                    stats = sample_data[col].describe()
                    logger.info(f"Statistics for {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
                    analysis[f"{col}_stats"] = stats.to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing table structure: {e}")
            return {"error": str(e), "table_name": table_name}
    
    def get_experimental_overview(self, table_name: str) -> Dict:
        """Get overview of experimental design from the data"""
        logger.info("Getting experimental overview...")
        
        try:
            # Get more data for analysis
            result = self.client.table(table_name).select("*").limit(1000).execute()
            data = pd.DataFrame(result.data)
            
            if data.empty:
                return {"error": "No data available"}
            
            overview = {
                "total_samples": len(data),
                "data_shape": data.shape
            }
            
            # Identify experimental variables
            categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_columns:
                unique_values = data[col].unique()
                if len(unique_values) < 50:  # Reasonable number for experimental conditions
                    overview[f"{col}_conditions"] = {
                        "count": len(unique_values),
                        "values": unique_values.tolist()
                    }
                    logger.info(f"{col}: {len(unique_values)} unique values")
            
            # Look for treatment/drug information
            treatment_columns = [col for col in data.columns if any(word in col.lower() for word in ['drug', 'treatment', 'compound', 'dose', 'concentration'])]
            if treatment_columns:
                logger.info(f"Treatment-related columns: {treatment_columns}")
                overview["treatment_columns"] = treatment_columns
                
                for col in treatment_columns:
                    if data[col].dtype == 'object':
                        treatments = data[col].value_counts()
                        logger.info(f"Treatments in {col}:")
                        for treatment, count in treatments.head(10).items():
                            logger.info(f"  {treatment}: {count} samples")
                        overview[f"{col}_treatments"] = treatments.head(20).to_dict()
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting experimental overview: {e}")
            return {"error": str(e)}
    
    def load_sample_time_series(self, table_name: str, limit: int = 1000) -> pd.DataFrame:
        """Load sample time series data for analysis"""
        logger.info(f"Loading sample time series data (limit: {limit})...")
        
        try:
            result = self.client.table(table_name).select("*").limit(limit).execute()
            data = pd.DataFrame(result.data)
            
            # Try to identify and parse time columns
            time_columns = [col for col in data.columns if any(time_word in col.lower() for time_word in ['time', 'date', 'timestamp', 'hour'])]
            
            for col in time_columns:
                try:
                    data[col] = pd.to_datetime(data[col])
                    logger.info(f"Parsed {col} as datetime")
                except:
                    logger.debug(f"Could not parse {col} as datetime")
            
            # Sort by time if possible
            if time_columns:
                data = data.sort_values(time_columns[0])
            
            logger.info(f"Loaded {len(data)} rows of sample data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return pd.DataFrame()

def main():
    """Main analysis function"""
    print("=" * 80)
    print("SONIC ANALYSIS PROJECT - ORGANOID DATA ANALYSIS")
    print("=" * 80)
    
    try:
        # Initialize data loader
        loader = SonicAnalysisDataLoader()
        print(f"✅ Connected to Supabase: {loader.url}")
        
        # Explore database structure
        print("\n" + "=" * 60)
        print("1. EXPLORING DATABASE STRUCTURE")
        print("=" * 60)
        
        db_structure = loader.explore_database_structure()
        print(f"Database structure: {db_structure}")
        
        # Analyze processes dat table
        print("\n" + "=" * 60)
        print("2. ANALYZING PROCESSES DAT TABLE")
        print("=" * 60)
        
        table_analysis = loader.analyze_processes_dat_table()
        
        if "error" in table_analysis:
            print(f"❌ Error: {table_analysis['error']}")
            return
        
        print(f"✅ Found table: {table_analysis['table_name']}")
        print(f"📊 Total rows: {table_analysis.get('total_rows', 'Unknown')}")
        print(f"📋 Columns ({len(table_analysis['columns'])}): {table_analysis['columns']}")
        
        if "time_columns" in table_analysis:
            print(f"⏰ Time columns: {table_analysis['time_columns']}")
        
        if "id_columns" in table_analysis:
            print(f"🔍 ID columns: {table_analysis['id_columns']}")
        
        if "measurement_columns" in table_analysis:
            print(f"📈 Measurement columns: {table_analysis['measurement_columns']}")
        
        # Get experimental overview
        print("\n" + "=" * 60)
        print("3. EXPERIMENTAL DESIGN OVERVIEW")
        print("=" * 60)
        
        exp_overview = loader.get_experimental_overview(table_analysis['table_name'])
        
        if "error" not in exp_overview:
            print(f"📝 Total samples: {exp_overview['total_samples']}")
            
            if "treatment_columns" in exp_overview:
                print(f"💊 Treatment columns: {exp_overview['treatment_columns']}")
        
        # Load sample data
        print("\n" + "=" * 60)
        print("4. SAMPLE DATA ANALYSIS")
        print("=" * 60)
        
        sample_data = loader.load_sample_time_series(table_analysis['table_name'], limit=5000)
        
        if not sample_data.empty:
            print(f"✅ Loaded sample data: {sample_data.shape}")
            print("\nFirst 5 rows:")
            print(sample_data.head())
            
            print("\nData types:")
            print(sample_data.dtypes)
            
            print("\nBasic statistics:")
            print(sample_data.describe())
            
            # Save sample to CSV for inspection
            sample_file = "/Users/shaunie/Documents/Code/organoid-embedding-experiments/sonic_sample_data.csv"
            sample_data.to_csv(sample_file, index=False)
            print(f"💾 Sample data saved to: {sample_file}")
        
        # Summary and recommendations
        print("\n" + "=" * 60)
        print("5. ANALYSIS SUMMARY & RECOMMENDATIONS")
        print("=" * 60)
        
        print("\n🔍 DATA STRUCTURE ANALYSIS:")
        print(f"  • Table: {table_analysis['table_name']}")
        print(f"  • Total rows: {table_analysis.get('total_rows', 'Unknown')}")
        print(f"  • Columns: {len(table_analysis['columns'])}")
        
        if "time_columns" in table_analysis:
            print(f"  • Time tracking: {len(table_analysis['time_columns'])} time columns")
        
        if "id_columns" in table_analysis:
            print(f"  • Identifiers: {len(table_analysis['id_columns'])} ID columns")
        
        if "measurement_columns" in table_analysis:
            print(f"  • Measurements: {len(table_analysis['measurement_columns'])} numeric columns")
        
        print("\n💡 INTEGRATION RECOMMENDATIONS:")
        print("  1. Adapt SupabaseDataLoader to use processes_dat_table structure")
        print("  2. Create mapping between sonic table columns and expected schema")
        print("  3. Implement data preprocessing for the real experimental data")
        print("  4. Validate time series continuity and missing data patterns")
        print("  5. Identify control conditions and treatment groups")
        
        print("\n🚀 NEXT STEPS:")
        print("  1. Create sonic-specific data loader class")
        print("  2. Implement preprocessing pipeline for real data")
        print("  3. Adapt embedding experiments to work with actual data structure")
        print("  4. Validate data quality and experimental design assumptions")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()