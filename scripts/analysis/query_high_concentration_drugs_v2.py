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
    
    # First, let's understand the data structure better
    print("=== CHECKING DATA STRUCTURE ===")
    sample = conn.execute("""
    SELECT drug, concentration, units, is_excluded, well_number, plate_id
    FROM supabase.public.well_map_data 
    WHERE concentration IS NOT NULL 
    LIMIT 10;
    """).df()
    print("Sample data from well_map_data:")
    print(sample)
    
    # Check concentration units
    units_check = conn.execute("""
    SELECT DISTINCT units, COUNT(*) as count
    FROM supabase.public.well_map_data
    WHERE concentration IS NOT NULL
    GROUP BY units;
    """).df()
    print("\nConcentration units found:")
    print(units_check)
    
    # SQL query to find drugs with high concentrations
    query = """
    SELECT 
        drug,
        COUNT(DISTINCT well_number || '-' || plate_id) as well_count,
        COUNT(DISTINCT concentration) as concentration_count,
        MIN(concentration) as min_concentration,
        MAX(concentration) as max_concentration,
        STRING_AGG(DISTINCT units, ', ') as units_used
    FROM 
        supabase.public.well_map_data
    WHERE 
        concentration > 22
        AND is_excluded = false
        AND drug IS NOT NULL
    GROUP BY 
        drug
    ORDER BY 
        well_count DESC
    """
    
    print("\n" + "=" * 80)
    print("Querying drugs with concentrations > 2.2E+1 (22)...")
    print("=" * 80)
    
    # Execute query
    result = conn.execute(query).df()
    
    if result.empty:
        print("No drugs found with concentrations > 22")
        return
    
    # Display results
    print(f"\nFound {len(result)} drugs with concentrations > 22:\n")
    print(f"{'Drug Name':<40} {'Well Count':<12} {'Conc. Count':<12} {'Min Conc.':<12} {'Max Conc.':<12} {'Units':<10}")
    print("-" * 115)
    
    for _, row in result.iterrows():
        print(f"{row['drug']:<40} {row['well_count']:<12} {row['concentration_count']:<12} {row['min_concentration']:<12.2f} {row['max_concentration']:<12.2f} {row['units_used']:<10}")
    
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
        concentration,
        COUNT(DISTINCT drug) as drug_count,
        COUNT(DISTINCT well_number || '-' || plate_id) as well_count
    FROM 
        supabase.public.well_map_data
    WHERE 
        concentration > 22
        AND is_excluded = false
        AND drug IS NOT NULL
    GROUP BY 
        concentration
    ORDER BY 
        concentration
    """
    
    conc_result = conn.execute(conc_query).df()
    
    if not conc_result.empty:
        print(f"\n{'Concentration':<15} {'Drug Count':<12} {'Well Count':<12}")
        print("-" * 40)
        for _, row in conc_result.iterrows():
            print(f"{row['concentration']:<15.2f} {row['drug_count']:<12} {row['well_count']:<12}")
    
    # Check specific high concentrations
    print("\n" + "=" * 80)
    print("Top 10 highest concentrations:")
    
    high_conc_query = """
    SELECT 
        drug,
        concentration,
        units,
        COUNT(DISTINCT well_number || '-' || plate_id) as well_count
    FROM 
        supabase.public.well_map_data
    WHERE 
        is_excluded = false
        AND drug IS NOT NULL
    GROUP BY 
        drug, concentration, units
    ORDER BY 
        concentration DESC
    LIMIT 10
    """
    
    high_conc_result = conn.execute(high_conc_query).df()
    print(f"\n{'Drug':<40} {'Concentration':<15} {'Units':<10} {'Wells':<10}")
    print("-" * 80)
    for _, row in high_conc_result.iterrows():
        print(f"{row['drug']:<40} {row['concentration']:<15.2f} {row['units']:<10} {row['well_count']:<10}")

if __name__ == "__main__":
    query_high_concentration_drugs()