#!/usr/bin/env python3
"""
Export entire database to local DuckDB file for offline analysis.

This script creates a complete local copy of the database to avoid
timeout issues and enable faster analysis.
"""

import os
import sys
from pathlib import Path
import duckdb
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output path for DuckDB file
OUTPUT_DIR = Path("data/database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = OUTPUT_DIR / "organoid_data.duckdb"

def export_to_duckdb():
    """Export all tables from PostgreSQL to local DuckDB file."""
    
    print("=" * 70)
    print("Database Export to DuckDB")
    print("=" * 70)
    print(f"Output file: {DB_FILE}")
    print()
    
    # Remove existing file if it exists
    if DB_FILE.exists():
        print(f"Removing existing database file...")
        DB_FILE.unlink()
    
    # Create new DuckDB connection
    local_conn = duckdb.connect(str(DB_FILE))
    
    with DataLoader() as loader:
        print("Connected to remote database")
        
        # Define tables to export with their queries
        tables = [
            ("drugs", "SELECT * FROM db.public.drugs"),
            ("event_table", "SELECT * FROM db.public.event_table"),
            ("well_map_data", "SELECT * FROM db.public.well_map_data"),
            ("plate_table", "SELECT * FROM db.public.plate_table"),
        ]
        
        # Export each table
        for table_name, query in tables:
            try:
                print(f"\nExporting {table_name}...")
                start_time = time.time()
                
                # Execute query and get dataframe
                df = loader._execute_and_convert(query)
                
                # Create table in DuckDB
                local_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                
                # Get row count
                count = local_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                elapsed = time.time() - start_time
                print(f"  ✓ Exported {count:,} rows in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Error exporting {table_name}: {e}")
                continue
        
        # Export processed_data in chunks (it's large)
        print(f"\nExporting processed_data (this may take a while)...")
        try:
            start_time = time.time()
            
            # First get total count
            count_query = "SELECT COUNT(*) as count FROM db.public.processed_data"
            total_rows = loader._execute_and_convert(count_query)['count'].iloc[0]
            print(f"  Total rows to export: {total_rows:,}")
            
            # Get list of plates to export one at a time
            plates_query = """
                SELECT DISTINCT plate_id, COUNT(*) as row_count 
                FROM db.public.processed_data 
                GROUP BY plate_id 
                ORDER BY row_count
            """
            plates_df = loader._execute_and_convert(plates_query)
            print(f"  Found {len(plates_df)} plates to export")
            
            # Create table structure first
            print("  Creating table structure...")
            # Get a small sample to determine the schema
            schema_query = """
                SELECT * FROM db.public.processed_data 
                LIMIT 1
            """
            schema_df = loader._execute_and_convert(schema_query)
            
            # Create empty dataframe with correct types
            empty_df = schema_df.iloc[0:0].copy()
            
            # Create table from empty dataframe - this preserves column types
            local_conn.execute("CREATE TABLE processed_data AS SELECT * FROM empty_df")
            
            # Export each plate
            exported_rows = 0
            for idx, (plate_id, row_count) in enumerate(zip(plates_df['plate_id'], plates_df['row_count'])):
                print(f"  Exporting plate {idx+1}/{len(plates_df)} ({row_count:,} rows)...", end='', flush=True)
                
                plate_query = f"""
                    SELECT * FROM db.public.processed_data 
                    WHERE plate_id = '{plate_id}'
                """
                
                try:
                    plate_df = loader._execute_and_convert(plate_query)
                    local_conn.execute("INSERT INTO processed_data SELECT * FROM plate_df")
                    exported_rows += len(plate_df)
                    print(f" ✓")
                except Exception as e:
                    print(f" ✗ Error: {e}")
                    continue
            
            elapsed = time.time() - start_time
            print(f"  ✓ Exported {exported_rows:,} rows in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error exporting processed_data: {e}")
        
        # Create indexes for better query performance
        print("\nCreating indexes...")
        indexes = [
            ("CREATE INDEX idx_processed_plate ON processed_data(plate_id)", "processed_data.plate_id"),
            ("CREATE INDEX idx_processed_well ON processed_data(plate_id, well_number)", "processed_data.well"),
            ("CREATE INDEX idx_processed_time ON processed_data(timestamp)", "processed_data.timestamp"),
            ("CREATE INDEX idx_event_plate ON event_table(plate_id)", "event_table.plate_id"),
            ("CREATE INDEX idx_event_time ON event_table(occurred_at)", "event_table.occurred_at"),
            ("CREATE INDEX idx_wellmap ON well_map_data(plate_id, well_number)", "well_map_data"),
        ]
        
        for idx_query, idx_name in indexes:
            try:
                local_conn.execute(idx_query)
                print(f"  ✓ Created index: {idx_name}")
            except Exception as e:
                print(f"  ✗ Error creating index {idx_name}: {e}")
    
    # Analyze database for query optimization
    print("\nOptimizing database...")
    local_conn.execute("ANALYZE")
    
    # Get database statistics
    print("\nDatabase statistics:")
    stats = local_conn.execute("""
        SELECT 
            table_name,
            COUNT(*) as column_count
        FROM information_schema.columns
        WHERE table_schema = 'main'
        GROUP BY table_name
    """).fetchall()
    
    for table, col_count in stats:
        row_count = local_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {row_count:,} rows, {col_count} columns")
    
    # Get file size
    file_size_mb = DB_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDatabase file size: {file_size_mb:.1f} MB")
    
    # Close connection
    local_conn.close()
    
    print(f"\n✓ Export complete! Database saved to: {DB_FILE}")
    print("\nTo use the local database, modify DataLoader to use:")
    print(f"  duckdb.connect('{DB_FILE}')")
    
    return DB_FILE

def verify_export():
    """Verify the exported database contains expected data."""
    
    print("\n" + "=" * 70)
    print("Verifying exported database...")
    print("=" * 70)
    
    conn = duckdb.connect(str(DB_FILE))
    
    # Check each table
    tables = ["drugs", "event_table", "well_map_data", "plate_table", "processed_data"]
    
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"✓ {table}: {count:,} rows")
        except Exception as e:
            print(f"✗ {table}: Error - {e}")
    
    # Test a sample query
    print("\nTesting sample query...")
    try:
        result = conn.execute("""
            SELECT 
                p.plate_id,
                COUNT(DISTINCT p.well_number) as wells,
                COUNT(*) as measurements,
                MIN(p.timestamp) as start_time,
                MAX(p.timestamp) as end_time
            FROM processed_data p
            GROUP BY p.plate_id
            LIMIT 5
        """).fetchdf()
        
        print("✓ Sample query successful:")
        print(result)
        
    except Exception as e:
        print(f"✗ Sample query failed: {e}")
    
    conn.close()

if __name__ == "__main__":
    # Export database
    db_file = export_to_duckdb()
    
    # Verify export
    verify_export()