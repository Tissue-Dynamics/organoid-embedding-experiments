#!/usr/bin/env python3
"""Check available tables in the database."""

import os
import duckdb
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def check_tables():
    """Check what tables are available in the database."""
    
    # Connect to database via DuckDB
    database_url = os.getenv('DATABASE_URL')
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    attach_query = f"""
    ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
    user={parsed.username} password={parsed.password}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    conn.execute(attach_query)
    
    # Get all tables
    print("=== AVAILABLE TABLES ===")
    try:
        tables = conn.execute("SELECT table_name FROM supabase.information_schema.tables WHERE table_schema = 'public';").df()
        table_names = tables['table_name'].tolist()
    except:
        # Alternative approach
        tables = conn.execute("SELECT * FROM information_schema.tables WHERE table_catalog = 'supabase' AND table_schema = 'public';").df()
        table_names = tables['table_name'].tolist()
    
    print("Tables found:", table_names)
    
    # Look for tables that might contain experiment/well data
    
    print("\n=== TABLES WITH 'WELL' OR 'EXPERIMENT' ===")
    relevant_tables = [t for t in table_names if 'well' in t.lower() or 'experiment' in t.lower()]
    print(relevant_tables)
    
    # Check each relevant table
    for table in relevant_tables:
        print(f"\n=== {table.upper()} TABLE STRUCTURE ===")
        try:
            columns = conn.execute(f"DESCRIBE supabase.public.{table};").df()
            print("Columns:", columns['column_name'].tolist())
            
            # Check if it has the columns we need
            col_names = columns['column_name'].tolist()
            required_cols = ['drug_id', 'drug_name', 'drug_concentration', 'well_id', 'is_excluded']
            available_required = [col for col in required_cols if col in col_names]
            print(f"Required columns present: {available_required}")
            
            # Get a sample
            sample = conn.execute(f"SELECT * FROM supabase.public.{table} LIMIT 3;").df()
            print("Sample data:")
            print(sample)
            
        except Exception as e:
            print(f"Error accessing {table}: {e}")

if __name__ == "__main__":
    check_tables()