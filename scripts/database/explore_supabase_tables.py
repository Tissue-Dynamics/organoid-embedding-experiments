#!/usr/bin/env python3
"""
Explore what tables and functions are available in the Supabase instance
"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    client = create_client(url, key)
    
    print("=" * 80)
    print("EXPLORING SUPABASE DATABASE")
    print("=" * 80)
    print(f"URL: {url}")
    
    # From the error log, we saw a hint about 'get_metabolic_timeseries' function
    print("\n1. Trying available RPC functions...")
    
    try:
        result = client.rpc('get_metabolic_timeseries').execute()
        print("✅ get_metabolic_timeseries function found!")
        print(f"Result type: {type(result.data)}")
        if result.data:
            df = pd.DataFrame(result.data)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Save sample
            df.to_csv('/Users/shaunie/Documents/Code/organoid-embedding-experiments/metabolic_timeseries_sample.csv', index=False)
            print("💾 Sample saved to metabolic_timeseries_sample.csv")
        
    except Exception as e:
        print(f"❌ get_metabolic_timeseries error: {e}")
    
    # Try to find tables by attempting common organoid-related table names
    print("\n2. Trying common table names...")
    
    common_tables = [
        'metabolic_data',
        'metabolic_timeseries',
        'organoid_metabolic_data',
        'time_series',
        'plate_data',
        'well_data',
        'measurement_data',
        'sensor_data',
        'oxygen_data',
        'process_data',
        'processes',
        'dat_table',
        'data_table',
        'timeseries_data',
        'experimental_data',
        'organoid_data',
        'organoids',
        'measurements',
        'readings',
        'results'
    ]
    
    found_tables = []
    
    for table in common_tables:
        try:
            result = client.table(table).select("*").limit(1).execute()
            found_tables.append(table)
            print(f"✅ Found table: {table}")
            
            # Get column info
            if result.data:
                df = pd.DataFrame(result.data)
                print(f"   Columns: {list(df.columns)}")
                
        except Exception as e:
            print(f"❌ {table}: {str(e)[:50]}...")
    
    print(f"\n📊 Found {len(found_tables)} tables: {found_tables}")
    
    # Try to get table list using SQL if possible
    print("\n3. Trying to get table schema information...")
    
    try:
        # Try different approaches to get schema
        schema_queries = [
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ]
        
        for query in schema_queries:
            try:
                result = client.rpc('sql_query', {'query': query}).execute()
                print(f"✅ Schema query worked: {query}")
                print(f"Tables: {result.data}")
                break
            except Exception as e:
                print(f"❌ Schema query failed: {str(e)[:50]}...")
                
    except Exception as e:
        print(f"❌ Schema exploration failed: {e}")
    
    # Check the existing tables we know work from the original loader
    print("\n4. Checking original project tables...")
    
    original_tables = [
        'plate_table',
        'well_map_data', 
        'processed_data',
        'event_table',
        'drugs'
    ]
    
    for table in original_tables:
        try:
            result = client.table(table).select("*").limit(1).execute()
            print(f"✅ Original table exists: {table}")
            if result.data:
                df = pd.DataFrame(result.data)
                print(f"   Columns: {list(df.columns)}")
                print(f"   Sample data available: {len(result.data)} rows")
        except Exception as e:
            print(f"❌ {table}: Not found")

if __name__ == "__main__":
    main()