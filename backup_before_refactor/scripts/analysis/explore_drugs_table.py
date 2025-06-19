#!/usr/bin/env python3
"""Explore the drugs table structure and content."""

import duckdb
import pandas as pd
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

def explore_drugs_table():
    """Explore the drugs table to understand available drug information."""
    
    load_dotenv()
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
    
    # Explore available tables
    print("=== AVAILABLE TABLES ===")
    try:
        tables = conn.execute("SELECT table_name FROM supabase.information_schema.tables WHERE table_schema = 'public';").df()
        table_names = tables['table_name'].tolist()
        print("Tables:", table_names)
    except Exception as e:
        print(f"Error getting tables: {e}")
        # Try simpler approach
        try:
            tables = conn.execute("SHOW TABLES FROM supabase;").df()
            table_names = tables['name'].tolist() if 'name' in tables.columns else tables.iloc[:, 0].tolist()
            print("Tables:", table_names)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            return conn, None, None
    
    # Look for drug-related tables  
    drug_tables = [t for t in table_names if 'drug' in t.lower()]
    print(f"\nDrug-related tables: {drug_tables}")
    
    # Check if drugs table exists
    if 'drugs' in table_names:
        print("\n=== DRUGS TABLE STRUCTURE ===")
        columns = conn.execute("DESCRIBE supabase.public.drugs;").df()
        print(columns[['column_name', 'column_type']])
        
        print("\n=== DRUGS TABLE SAMPLE ===")
        sample = conn.execute("SELECT * FROM supabase.public.drugs LIMIT 5;").df()
        print(sample)
        
        print("\n=== KEY DRUG PROPERTIES ===")
        key_props = conn.execute("""
        SELECT drug, dili, dili_risk_category, experimental_names, 
               pathway, target, mechanism, moa, 
               hepatotoxicity, nephrotoxicity, cardiotoxicity
        FROM supabase.public.drugs 
        LIMIT 10;
        """).df()
        print(key_props)
        
        print("\n=== DRUGS TABLE SUMMARY ===")
        count = conn.execute("SELECT COUNT(*) as total_drugs FROM supabase.public.drugs;").df()
        print(f"Total drugs in table: {count['total_drugs'].iloc[0]}")
        
        # Check for key columns we might want
        key_columns = ['name', 'compound_name', 'moa', 'target', 'mechanism', 'pathway', 'class', 'category']
        available_columns = columns['column_name'].str.lower().tolist()
        
        print("\n=== AVAILABLE KEY COLUMNS ===")
        for col in key_columns:
            matches = [c for c in available_columns if col in c]
            if matches:
                print(f"  {col}: {matches}")
        
        return conn, columns, sample
    else:
        print("No 'drugs' table found.")
        
        # Check other tables that might contain drug info
        for table in drug_tables:
            print(f"\n=== {table.upper()} TABLE ===")
            try:
                columns = conn.execute(f"DESCRIBE supabase.public.{table};").df()
                print("Columns:", columns['column_name'].tolist())
                
                sample = conn.execute(f"SELECT * FROM supabase.public.{table} LIMIT 3;").df()
                print("Sample data:")
                print(sample)
            except Exception as e:
                print(f"Error accessing {table}: {e}")
        
        return conn, None, None

if __name__ == "__main__":
    explore_drugs_table()