#!/usr/bin/env python3
"""
Download updated processed_data with latest exclusions using DuckDB.
"""

import os
import sys
import duckdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def download_processed_data():
    """Download the full processed_data table with updated exclusions."""
    print("Downloading updated processed_data with latest exclusions...")
    
    # Connect to database
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
    
    # Download full processed_data table
    query = """
    SELECT 
        plate_id,
        well_number,
        timestamp,
        median_o2,
        is_excluded,
        exclusion_reason
    FROM supabase.public.processed_data
    ORDER BY plate_id, well_number, timestamp
    """
    
    print("  Executing query...")
    processed_df = conn.execute(query).df()
    
    print(f"  Downloaded {len(processed_df):,} rows")
    print(f"  Excluded rows: {processed_df['is_excluded'].sum():,}")
    print(f"  Non-excluded rows: {(~processed_df['is_excluded']).sum():,}")
    
    # Check exclusion reasons
    if 'exclusion_reason' in processed_df.columns:
        exclusion_counts = processed_df[processed_df['is_excluded']]['exclusion_reason'].value_counts()
        if len(exclusion_counts) > 0:
            print("\n  Exclusion reasons:")
            for reason, count in exclusion_counts.items():
                print(f"    {reason}: {count:,}")
    
    # Convert UUIDs to strings for parquet compatibility
    processed_df['plate_id'] = processed_df['plate_id'].astype(str)
    
    # Save to parquet for efficient storage
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "processed_data_updated.parquet"
    processed_df.to_parquet(output_path, index=False, compression='snappy')
    print(f"\n  Saved to: {output_path}")
    
    # Also download well_map_data for reference
    well_query = """
    SELECT 
        plate_id,
        well_number,
        drug,
        concentration,
        is_excluded,
        exclusion_reason
    FROM supabase.public.well_map_data
    WHERE drug IS NOT NULL AND drug != ''
    ORDER BY drug, concentration, plate_id, well_number
    """
    
    print("\nDownloading well_map_data...")
    well_df = conn.execute(well_query).df()
    
    print(f"  Downloaded {len(well_df):,} wells")
    print(f"  Excluded wells: {well_df['is_excluded'].sum():,}")
    print(f"  Non-excluded wells: {(~well_df['is_excluded']).sum():,}")
    
    # Convert UUIDs to strings for parquet compatibility
    well_df['plate_id'] = well_df['plate_id'].astype(str)
    
    well_output_path = output_dir / "well_map_data_updated.parquet"
    well_df.to_parquet(well_output_path, index=False, compression='snappy')
    print(f"  Saved to: {well_output_path}")
    
    conn.close()
    
    return processed_df, well_df


def analyze_data_quality(processed_df, well_df):
    """Analyze the quality of downloaded data."""
    print("\nAnalyzing data quality...")
    
    # Time range
    time_range = processed_df['timestamp'].max() - processed_df['timestamp'].min()
    print(f"  Time range: {time_range.days} days")
    
    # Unique plates and wells
    n_plates = processed_df['plate_id'].nunique()
    n_wells = len(processed_df[['plate_id', 'well_number']].drop_duplicates())
    print(f"  Unique plates: {n_plates}")
    print(f"  Unique plate-well combinations: {n_wells}")
    
    # Drug statistics
    drug_stats = well_df.groupby('drug').agg({
        'concentration': 'nunique',
        'well_number': 'count',
        'is_excluded': 'sum'
    }).rename(columns={
        'concentration': 'n_concentrations',
        'well_number': 'n_wells',
        'is_excluded': 'n_excluded'
    })
    
    drug_stats['n_valid_wells'] = drug_stats['n_wells'] - drug_stats['n_excluded']
    
    print(f"\n  Total drugs: {len(drug_stats)}")
    print(f"  Drugs with ≥4 concentrations: {(drug_stats['n_concentrations'] >= 4).sum()}")
    print(f"  Drugs with ≥8 valid wells: {(drug_stats['n_valid_wells'] >= 8).sum()}")


if __name__ == "__main__":
    processed_df, well_df = download_processed_data()
    analyze_data_quality(processed_df, well_df)