#!/usr/bin/env python3
"""Query drugs with concentrations over 2.2E+1 and count their wells."""

import os
import duckdb
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def query_high_concentration_drugs():
    """Query drugs with concentrations > 22 and count their wells."""
    
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
    
    # SQL query to find drugs with high concentrations
    query = """
    SELECT 
        drug_id,
        drug_name,
        COUNT(DISTINCT well_id) as well_count,
        COUNT(DISTINCT drug_concentration) as concentration_count,
        MIN(drug_concentration) as min_concentration,
        MAX(drug_concentration) as max_concentration
    FROM 
        supabase.public.experiment_wells
    WHERE 
        drug_concentration > 22
        AND is_excluded = false
    GROUP BY 
        drug_id, drug_name
    ORDER BY 
        well_count DESC
    """
    
    print("Querying drugs with concentrations > 2.2E+1 (22)...")
    print("=" * 80)
    
    # Execute query
    result = conn.execute(query).df()
    
    if result.empty:
        print("No drugs found with concentrations > 22")
        return
    
    # Display results
    print(f"\nFound {len(result)} drugs with concentrations > 22:\n")
    print(f"{'Drug Name':<40} {'Drug ID':<15} {'Well Count':<12} {'Conc. Count':<12} {'Min Conc.':<12} {'Max Conc.':<12}")
    print("-" * 115)
    
    for _, row in result.iterrows():
        print(f"{row['drug_name']:<40} {row['drug_id']:<15} {row['well_count']:<12} {row['concentration_count']:<12} {row['min_concentration']:<12.2f} {row['max_concentration']:<12.2f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print(f"Total drugs with high concentrations: {len(result)}")
    print(f"Total wells affected: {result['well_count'].sum()}")
    print(f"Average wells per drug: {result['well_count'].mean():.1f}")
    print(f"Max concentration found: {result['max_concentration'].max():.2f}")
    
    # Additional analysis: Check concentration distribution
    print("\n" + "=" * 80)
    print("Checking concentration distribution for these drugs...")
    
    conc_query = """
    SELECT 
        drug_concentration,
        COUNT(DISTINCT drug_id) as drug_count,
        COUNT(DISTINCT well_id) as well_count
    FROM 
        supabase.public.experiment_wells
    WHERE 
        drug_concentration > 22
        AND is_excluded = false
    GROUP BY 
        drug_concentration
    ORDER BY 
        drug_concentration
    """
    
    conc_result = conn.execute(conc_query).df()
    
    if not conc_result.empty:
        print(f"\n{'Concentration':<15} {'Drug Count':<12} {'Well Count':<12}")
        print("-" * 40)
        for _, row in conc_result.iterrows():
            print(f"{row['drug_concentration']:<15.2f} {row['drug_count']:<12} {row['well_count']:<12}")

if __name__ == "__main__":
    query_high_concentration_drugs()