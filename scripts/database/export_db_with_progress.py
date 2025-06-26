#!/usr/bin/env python3
"""
Export database to local DuckDB file with live progress display.
Single script that handles both export and progress monitoring.
Uses correct data types based on database schema documentation.
"""

import os
import sys
from pathlib import Path
import duckdb
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output path for DuckDB file
OUTPUT_DIR = Path("data/database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = OUTPUT_DIR / "organoid_data.duckdb"

def format_progress_bar(current, total, width=40):
    """Create a visual progress bar."""
    if total == 0:
        return f"[{'?' * width}]"
    
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {current:,}/{total:,} ({progress*100:.1f}%)"

def export_with_progress():
    """Export database with integrated progress display."""
    
    start_time = time.time()
    
    print("=" * 80)
    print("DATABASE EXPORT TO DUCKDB")
    print("=" * 80)
    print(f"Output: {DB_FILE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Remove existing file if it exists
    if DB_FILE.exists():
        print("Removing existing database file...")
        DB_FILE.unlink()
    
    # Create new DuckDB connection
    local_conn = duckdb.connect(str(DB_FILE))
    
    # Force close any existing DuckDB connections to PostgreSQL
    try:
        # This will close all DuckDB connections in the current process
        import gc
        gc.collect()  # Force garbage collection to clean up any orphaned connections
    except:
        pass
    
    try:
        # Use context manager to ensure proper connection cleanup
        with DataLoader(use_local=False) as loader:  # Force remote
            print("✓ Connected to remote database")
            
            # Force garbage collection to clean up any previous connections
            import gc
            gc.collect()
            
            # Quick connection test
            print("\nTesting database connection...", end=' ', flush=True)
            try:
                test_result = loader._execute_and_convert("SELECT COUNT(*) as count FROM db.public.drugs")
                test_count = test_result['count'].iloc[0]
                print(f"✓ Found {test_count} drugs\n")
            except Exception as e:
                print(f"✗ Connection test failed: {e}")
                return False
            
            # Small tables to export with proper queries
            small_tables = [
                ("drugs", "SELECT * FROM db.public.drugs"),
                ("event_table", "SELECT * FROM db.public.event_table"),
                ("well_map_data", "SELECT * FROM db.public.well_map_data"),
                ("plate_table", """
                    SELECT 
                        id::VARCHAR as id,
                        name,
                        created_at,
                        updated_at,
                        created_by::VARCHAR as created_by,
                        deleted,
                        status::VARCHAR as status,
                        state::VARCHAR as state,
                        tissue,
                        description,
                        array_to_string(plate_size, 'x') as plate_size,
                        qc_values::VARCHAR as qc_values,
                        qc_thresholds::VARCHAR as qc_thresholds,
                        internal_notes::VARCHAR as internal_notes
                    FROM db.public.plate_table
                """),
                ("well_image_data", "SELECT * FROM db.public.well_image_data"),
                # Gene biomarkers tables
                ("gene_samples", "SELECT * FROM db.gene_biomarkers.samples"),
                ("gene_biomarkers", "SELECT * FROM db.gene_biomarkers.biomarkers"),
                ("gene_drug_keys", "SELECT * FROM db.gene_biomarkers.drug_keys"),
                ("gene_expression", "SELECT * FROM db.gene_biomarkers.gene_expression"),
            ]
            
            # Export small tables
            print("EXPORTING SMALL TABLES")
            print("-" * 80)
            
            for table_name, query in small_tables:
                print(f"\n{table_name}:", end=' ', flush=True)
                
                try:
                    table_start = time.time()
                    df = loader._execute_and_convert(query)
                    local_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                    
                    elapsed = time.time() - table_start
                    print(f"✓ {len(df):,} rows in {elapsed:.1f}s")
                    
                except Exception as e:
                    print(f"✗ ERROR: {e}")
            
            # Export processed_data
            print("\n\nEXPORTING PROCESSED_DATA")
            print("-" * 80)
            
            try:
                # Create table structure based on documented schema
                print("Creating table structure...", end=' ', flush=True)
                create_table_query = """
                    CREATE TABLE processed_data (
                        id BIGINT,
                        plate_id VARCHAR,           -- UUID stored as VARCHAR
                        well_number SMALLINT,
                        timestamp TIMESTAMP WITH TIME ZONE,
                        median_o2 FLOAT,
                        cycle_time_stamp TIMESTAMP WITH TIME ZONE,
                        cycle_num SMALLINT,
                        is_excluded BOOLEAN,
                        exclusion_reason VARCHAR,   -- Can be NULL, integer, or string
                        excluded_by VARCHAR,        -- Can be NULL, integer (user_id), or string
                        excluded_at TIMESTAMP WITH TIME ZONE
                    )
                """
                local_conn.execute(create_table_query)
                print("✓")
                
                # OPTIMIZED: Row-based chunking (much faster than plate-by-plate)
                print("Getting total row count...", end=' ', flush=True)
                start = time.time()
                count_query = """
                    SELECT COUNT(*) as total_count 
                    FROM db.public.processed_data 
                    WHERE is_excluded = false
                """
                count_result = loader._execute_and_convert(count_query)
                total_rows = count_result['total_count'].iloc[0]
                elapsed = time.time() - start
                print(f"✓ Found {total_rows:,} rows ({elapsed:.1f}s)")
                
                if total_rows == 0:
                    print("✗ No data to export")
                    exported_rows = 0
                else:
                    # Export in chunks - smaller chunks for SSL stability
                    chunk_size = 5000  # 5K rows per chunk - more stable for SSL connections
                    total_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
                    
                    print(f"\nExporting {total_rows:,} rows in {total_chunks} chunks of {chunk_size:,}...")
                    print("=" * 70)
                    
                    exported_rows = 0
                    
                    for chunk_num in range(total_chunks):
                        offset = chunk_num * chunk_size
                        
                        # Retry logic for SSL/connection errors
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                chunk_start = time.time()
                                
                                # Optimized query - no COALESCE, use UUID directly, simple filtering
                                chunk_query = f"""
                                    SELECT 
                                        id,
                                        plate_id,
                                        well_number,
                                        timestamp,
                                        median_o2,
                                        cycle_time_stamp,
                                        cycle_num,
                                        is_excluded,
                                        exclusion_reason,
                                        excluded_by,
                                        excluded_at
                                    FROM db.public.processed_data 
                                    WHERE is_excluded = false
                                    ORDER BY id
                                    LIMIT {chunk_size} OFFSET {offset}
                                """
                                
                                chunk_df = loader._execute_and_convert(chunk_query)
                                chunk_rows = len(chunk_df)
                                
                                if chunk_rows > 0:
                                    # Insert into local database
                                    local_conn.execute("INSERT INTO processed_data SELECT * FROM chunk_df")
                                    exported_rows += chunk_rows
                                    
                                    elapsed = time.time() - chunk_start
                                    progress_bar = format_progress_bar(chunk_num + 1, total_chunks)
                                    
                                    print(f"  {progress_bar} Chunk {chunk_num+1}/{total_chunks}: {chunk_rows:,} rows ({elapsed:.1f}s) - Total: {exported_rows:,}")
                                
                                # Minimal cleanup (skip aggressive gc.collect())
                                del chunk_df
                                break  # Success - exit retry loop
                                    
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    # Retry with backoff
                                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                                    print(f"  Chunk {chunk_num+1}/{total_chunks}: ⚠️  SSL error, retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                                    time.sleep(wait_time)
                                    continue
                                else:
                                    # Final failure
                                    print(f"  Chunk {chunk_num+1}/{total_chunks}: ✗ FAILED after {max_retries} attempts: {e}")
                                    import gc
                                    gc.collect()
                                    break
                
                # Final progress
                print(f"\n{'=' * 80}")
                print(f"PROCESSED_DATA EXPORT COMPLETE")
                print(f"{'=' * 80}")
                print(f"Total rows exported: {exported_rows:,}")
                
                if exported_rows > 0:
                    # Get some stats
                    stats_query = "SELECT COUNT(DISTINCT plate_id) as plates, MIN(timestamp) as min_time, MAX(timestamp) as max_time FROM processed_data"
                    stats = local_conn.execute(stats_query).fetchone()
                    print(f"Unique plates: {stats[0]}")
                    print(f"Time range: {stats[1]} to {stats[2]}")
                
            except Exception as e:
                print(f"\n✗ ERROR exporting processed_data: {e}")
            
            # Create indexes
            print("\n\nCREATING INDEXES")
            print("-" * 80)
            
            indexes = [
                ("processed_data(plate_id)", "CREATE INDEX idx_processed_plate ON processed_data(plate_id)"),
                ("processed_data(plate_id, well_number)", "CREATE INDEX idx_processed_well ON processed_data(plate_id, well_number)"),
                ("processed_data(timestamp)", "CREATE INDEX idx_processed_time ON processed_data(timestamp)"),
                ("event_table(plate_id)", "CREATE INDEX idx_event_plate ON event_table(plate_id)"),
                ("event_table(occurred_at)", "CREATE INDEX idx_event_time ON event_table(occurred_at)"),
                ("well_map_data(plate_id, well_number)", "CREATE INDEX idx_wellmap ON well_map_data(plate_id, well_number)"),
            ]
            
            for idx_name, idx_query in indexes:
                print(f"\n{idx_name}:", end=' ', flush=True)
                try:
                    local_conn.execute(idx_query)
                    print("✓")
                except Exception as e:
                    print(f"✗ {e}")
            
            # Optimize
            print("\n\nOPTIMIZING DATABASE")
            print("-" * 80)
            print("Running ANALYZE...", end=' ', flush=True)
            local_conn.execute("ANALYZE")
            print("✓")
            
            # Final statistics
            print("\n\nFINAL STATISTICS")
            print("-" * 80)
            
            tables_to_check = [
                "drugs", "event_table", "well_map_data", "plate_table", 
                "well_image_data", "gene_samples", "gene_biomarkers", 
                "gene_drug_keys", "gene_expression", "processed_data"
            ]
            
            for table in tables_to_check:
                try:
                    count = local_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    print(f"{table}: {count:,} rows")
                except:
                    print(f"{table}: ERROR")
            
            # File size
            file_size_mb = DB_FILE.stat().st_size / (1024 * 1024)
            print(f"\nDatabase size: {file_size_mb:.1f} MB")
            
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        return False
    
    finally:
        local_conn.close()
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Output file: {DB_FILE}")
    print("\nTo use the local database:")
    print("  DataLoader(use_local=True)")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    # Use Transaction pooler with IPv4 add-on (many more connections)
    if not os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = "postgres://postgres:eTEEoWWGExovyChe@db.ooqjakwyfawahvnzcllk.supabase.co:6543/postgres"
    
    print("Using Supabase Transaction Pooler with IPv4 add-on")
    print("Plate-by-plate export with progress updates")
    print()
    
    success = export_with_progress()
    sys.exit(0 if success else 1)