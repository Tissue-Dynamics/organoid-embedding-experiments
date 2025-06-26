#!/usr/bin/env python3
"""
Export database to local DuckDB file with live progress display.
Single script that handles both export and progress monitoring.
"""

import os
import sys
from pathlib import Path
import duckdb
import time
from datetime import datetime, timedelta

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
    
    try:
        with DataLoader(use_local=False) as loader:  # Force remote
            print("✓ Connected to remote database")
            
            # Quick connection test
            print("\nTesting database connection...", end=' ', flush=True)
            try:
                test_result = loader._execute_and_convert("SELECT COUNT(*) as count FROM db.public.drugs")
                test_count = test_result['count'].iloc[0]
                print(f"✓ Found {test_count} drugs\n")
            except Exception as e:
                print(f"✗ Connection test failed: {e}")
                return False
            
            # Small tables to export
            small_tables = [
                ("drugs", "SELECT * FROM db.public.drugs"),
                ("event_table", "SELECT * FROM db.public.event_table"),
                ("well_map_data", "SELECT * FROM db.public.well_map_data"),
                ("plate_table", """
                    SELECT 
                        id, name, created_at, updated_at, created_by, 
                        deleted, status, state, tissue, description,
                        array_to_string(plate_size, 'x') as plate_size,
                        qc_values, qc_thresholds, internal_notes
                    FROM db.public.plate_table
                """),
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
                # Get total count
                print("Counting rows...", end=' ', flush=True)
                count_query = "SELECT COUNT(*) as count FROM db.public.processed_data"
                total_rows = loader._execute_and_convert(count_query)['count'].iloc[0]
                print(f"{total_rows:,} total rows")
                
                # Get plates
                print("Getting plate list...", end=' ', flush=True)
                plates_query = """
                    SELECT DISTINCT plate_id, COUNT(*) as row_count 
                    FROM db.public.processed_data 
                    GROUP BY plate_id 
                    ORDER BY row_count
                """
                plates_df = loader._execute_and_convert(plates_query)
                num_plates = len(plates_df)
                print(f"{num_plates} plates\n")
                
                # Create table structure
                print("Creating table structure...", end=' ', flush=True)
                # Create table with explicit schema to avoid type issues
                create_table_query = """
                    CREATE TABLE processed_data (
                        id BIGINT,
                        plate_id VARCHAR,
                        well_number SMALLINT,
                        timestamp TIMESTAMP WITH TIME ZONE,
                        median_o2 FLOAT,
                        cycle_time_stamp TIMESTAMP WITH TIME ZONE,
                        cycle_num SMALLINT,
                        is_excluded BOOLEAN,
                        exclusion_reason INTEGER,
                        excluded_by INTEGER,
                        excluded_at TIMESTAMP WITH TIME ZONE
                    )
                """
                local_conn.execute(create_table_query)
                print("✓")
                
                # Export each plate with progress
                print("\nExporting plates:")
                exported_rows = 0
                failed_plates = []
                successful_plates = 0
                
                for idx, (plate_id, row_count) in enumerate(zip(plates_df['plate_id'], plates_df['row_count'])):
                    # Calculate ETA
                    if successful_plates > 0 and exported_rows > 0:
                        elapsed = time.time() - start_time
                        rate = exported_rows / elapsed
                        remaining_rows = total_rows - exported_rows
                        eta_seconds = remaining_rows / rate if rate > 0 else 0
                        eta_str = f" | ETA: {str(timedelta(seconds=int(eta_seconds)))}"
                    else:
                        eta_str = " | Calculating..."
                    
                    # Progress display
                    print(f"\rPlate {idx+1}/{num_plates} {format_progress_bar(exported_rows, total_rows)}{eta_str}", 
                          end='', flush=True)
                    
                    # Export plate
                    plate_query = f"""
                        SELECT * FROM db.public.processed_data 
                        WHERE plate_id = '{plate_id}'
                    """
                    
                    try:
                        # Debug: show what we're doing
                        plate_start = time.time()
                        
                        # Execute query
                        plate_df = loader._execute_and_convert(plate_query)
                        rows_fetched = len(plate_df)
                        
                        if rows_fetched == 0:
                            failed_plates.append((plate_id, f"No rows returned (expected {row_count})"))
                            continue
                        
                        # Insert into local database
                        local_conn.execute("INSERT INTO processed_data SELECT * FROM plate_df")
                        
                        # Verify insertion
                        verify_count = local_conn.execute(f"SELECT COUNT(*) FROM processed_data").fetchone()[0]
                        
                        exported_rows += rows_fetched
                        successful_plates += 1
                        
                        # Show success for this plate
                        plate_time = time.time() - plate_start
                        print(f"\rPlate {idx+1}/{num_plates} ({plate_id[:8]}...) ✓ {rows_fetched:,} rows in {plate_time:.1f}s", end='', flush=True)
                        print()  # New line for next plate
                        
                    except Exception as e:
                        failed_plates.append((plate_id, str(e)))
                        print(f"\rPlate {idx+1}/{num_plates} ({plate_id[:8]}...) ✗ ERROR: {str(e)[:50]}...", end='', flush=True)
                        print()  # New line for next plate
                
                # Final progress
                print(f"\rPlate {num_plates}/{num_plates} {format_progress_bar(exported_rows, total_rows)} | COMPLETE")
                
                if failed_plates:
                    print(f"\n\n⚠️  Failed to export {len(failed_plates)} plates:")
                    for pid, err in failed_plates[:5]:  # Show first 5
                        print(f"  - {pid}: {err[:50]}...")
                    if len(failed_plates) > 5:
                        print(f"  ... and {len(failed_plates) - 5} more")
                
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
            
            for table in ["drugs", "event_table", "well_map_data", "plate_table", "processed_data"]:
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
    # Set DATABASE_URL if not already set
    if not os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    
    success = export_with_progress()
    sys.exit(0 if success else 1)