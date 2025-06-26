#!/usr/bin/env python3
"""
Export a sample of the database to local DuckDB file for testing.
This exports only a few plates to verify the approach works.
"""

import os
import sys
from pathlib import Path
import duckdb
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output path for DuckDB file
OUTPUT_DIR = Path("data/database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = OUTPUT_DIR / "organoid_data_sample.duckdb"

def export_sample_to_duckdb():
    """Export a sample of tables from PostgreSQL to local DuckDB file."""
    
    print("=" * 70)
    print("Database Sample Export to DuckDB")
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
        
        # Export small tables completely
        tables = [
            ("drugs", "SELECT * FROM db.public.drugs"),
            ("event_table", "SELECT * FROM db.public.event_table"),
            ("plate_table", "SELECT * FROM db.public.plate_table"),
        ]
        
        for table_name, query in tables:
            try:
                print(f"\nExporting {table_name}...")
                start_time = time.time()
                
                df = loader._execute_and_convert(query)
                local_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                
                count = local_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                elapsed = time.time() - start_time
                print(f"  ✓ Exported {count:,} rows in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  ✗ Error exporting {table_name}: {e}")
        
        # Export sample of well_map_data (first 3 plates)
        print(f"\nExporting well_map_data sample...")
        try:
            sample_plates_query = """
                SELECT DISTINCT plate_id 
                FROM db.public.well_map_data 
                LIMIT 3
            """
            sample_plates = loader._execute_and_convert(sample_plates_query)
            plate_ids = "', '".join(sample_plates['plate_id'].astype(str).tolist())
            
            well_query = f"""
                SELECT * FROM db.public.well_map_data 
                WHERE plate_id IN ('{plate_ids}')
            """
            
            df = loader._execute_and_convert(well_query)
            local_conn.execute("CREATE TABLE well_map_data AS SELECT * FROM df")
            
            count = local_conn.execute("SELECT COUNT(*) FROM well_map_data").fetchone()[0]
            print(f"  ✓ Exported {count:,} rows")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Export sample of processed_data (first 3 plates)
        print(f"\nExporting processed_data sample...")
        try:
            # Use the same plates from well_map_data
            processed_query = f"""
                SELECT * FROM db.public.processed_data 
                WHERE plate_id IN ('{plate_ids}')
            """
            
            df = loader._execute_and_convert(processed_query)
            local_conn.execute("CREATE TABLE processed_data AS SELECT * FROM df")
            
            count = local_conn.execute("SELECT COUNT(*) FROM processed_data").fetchone()[0]
            print(f"  ✓ Exported {count:,} rows")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Create indexes
        print("\nCreating indexes...")
        indexes = [
            "CREATE INDEX idx_processed_plate ON processed_data(plate_id)",
            "CREATE INDEX idx_event_plate ON event_table(plate_id)",
            "CREATE INDEX idx_wellmap ON well_map_data(plate_id, well_number)",
        ]
        
        for idx_query in indexes:
            try:
                local_conn.execute(idx_query)
                print(f"  ✓ {idx_query}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Analyze database
    local_conn.execute("ANALYZE")
    
    # Get statistics
    print("\nDatabase statistics:")
    tables = ["drugs", "event_table", "plate_table", "well_map_data", "processed_data"]
    
    for table in tables:
        try:
            count = local_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count:,} rows")
        except:
            pass
    
    # Get file size
    file_size_mb = DB_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDatabase file size: {file_size_mb:.1f} MB")
    
    local_conn.close()
    
    print(f"\n✓ Sample export complete!")
    return DB_FILE

if __name__ == "__main__":
    export_sample_to_duckdb()