#!/usr/bin/env python3
"""
Check data exclusion statistics to understand what data we should be using.
"""

import duckdb
import pandas as pd

def check_exclusion_stats():
    """Check exclusion statistics in the database."""
    
    # Connect to database
    DATABASE_URL = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    conn.execute(f"ATTACH '{DATABASE_URL}' AS supabase (TYPE POSTGRES, READ_ONLY);")
    
    print("=== WELL MAP EXCLUSION STATS ===")
    result = conn.execute("""
    SELECT 
        is_excluded,
        COUNT(*) as count,
        COUNT(DISTINCT drug) as n_drugs,
        COUNT(DISTINCT plate_id) as n_plates,
        COUNT(DISTINCT concentration) as n_concentrations
    FROM supabase.public.well_map_data
    WHERE drug != '' AND drug IS NOT NULL
    GROUP BY is_excluded
    ORDER BY is_excluded
    """).df()
    print(result)
    
    print("\n=== PROCESSED DATA EXCLUSION STATS ===")
    result = conn.execute("""
    SELECT 
        is_excluded,
        COUNT(*) as count,
        COUNT(DISTINCT plate_id || '_' || well_number) as n_wells
    FROM supabase.public.processed_data
    GROUP BY is_excluded
    ORDER BY is_excluded
    """).df()
    print(result)
    
    print("\n=== CROSS-CHECK: WELLS WITH MIXED EXCLUSION STATUS ===")
    result = conn.execute("""
    SELECT 
        w.is_excluded as well_excluded,
        p.is_excluded as data_excluded,
        COUNT(DISTINCT w.plate_id || '_' || w.well_number) as n_wells
    FROM supabase.public.well_map_data w
    JOIN supabase.public.processed_data p
        ON w.plate_id = p.plate_id AND w.well_number = p.well_number
    WHERE w.drug != '' AND w.drug IS NOT NULL
    GROUP BY w.is_excluded, p.is_excluded
    ORDER BY w.is_excluded, p.is_excluded
    """).df()
    print(result)
    
    print("\n=== SAMPLE OF EXCLUDED DATA ===")
    result = conn.execute("""
    SELECT 
        w.drug,
        w.concentration,
        w.is_excluded as well_excluded,
        COUNT(DISTINCT w.well_number) as n_wells
    FROM supabase.public.well_map_data w
    WHERE w.is_excluded = true
        AND w.drug != '' 
        AND w.drug IS NOT NULL
    GROUP BY w.drug, w.concentration, w.is_excluded
    ORDER BY n_wells DESC
    LIMIT 10
    """).df()
    print(result)
    
    # Check if we're missing important data
    print("\n=== DRUGS ONLY IN EXCLUDED DATA ===")
    result = conn.execute("""
    WITH excluded_drugs AS (
        SELECT DISTINCT drug
        FROM supabase.public.well_map_data
        WHERE is_excluded = true AND drug != '' AND drug IS NOT NULL
    ),
    included_drugs AS (
        SELECT DISTINCT drug
        FROM supabase.public.well_map_data
        WHERE is_excluded = false AND drug != '' AND drug IS NOT NULL
    )
    SELECT 
        e.drug,
        COUNT(DISTINCT w.concentration) as n_concentrations,
        COUNT(DISTINCT w.well_number) as n_wells
    FROM excluded_drugs e
    LEFT JOIN included_drugs i ON e.drug = i.drug
    JOIN supabase.public.well_map_data w ON e.drug = w.drug
    WHERE i.drug IS NULL  -- Only in excluded
        AND w.is_excluded = true
    GROUP BY e.drug
    ORDER BY n_wells DESC
    LIMIT 20
    """).df()
    
    if len(result) > 0:
        print(result)
    else:
        print("No drugs found that are only in excluded data.")
    
    # Check what happens if we use both exclusion flags
    print("\n=== IMPACT OF USING BOTH EXCLUSION FLAGS ===")
    
    # Only well_map exclusion
    result1 = conn.execute("""
    SELECT COUNT(DISTINCT w.drug || '_' || w.concentration) as drug_conc_combos
    FROM supabase.public.well_map_data w
    JOIN supabase.public.processed_data p
        ON w.plate_id = p.plate_id AND w.well_number = p.well_number
    WHERE w.drug != '' AND w.drug IS NOT NULL
        AND w.is_excluded = false
    """).fetchone()
    
    # Both exclusion flags
    result2 = conn.execute("""
    SELECT COUNT(DISTINCT w.drug || '_' || w.concentration) as drug_conc_combos
    FROM supabase.public.well_map_data w
    JOIN supabase.public.processed_data p
        ON w.plate_id = p.plate_id AND w.well_number = p.well_number
    WHERE w.drug != '' AND w.drug IS NOT NULL
        AND w.is_excluded = false
        AND p.is_excluded = false
    """).fetchone()
    
    print(f"Using only well_map.is_excluded = false: {result1[0]} drug/conc combinations")
    print(f"Using both is_excluded = false: {result2[0]} drug/conc combinations")
    print(f"Difference: {result1[0] - result2[0]} combinations lost")
    
    conn.close()

if __name__ == "__main__":
    check_exclusion_stats()