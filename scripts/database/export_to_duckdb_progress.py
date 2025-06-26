#!/usr/bin/env python3
"""
Export entire database to local DuckDB file with interactive progress updates.

This script creates a complete local copy of the database to avoid
timeout issues and enable faster analysis. Includes real-time progress
updates suitable for Zed task runner.
"""

import os
import sys
from pathlib import Path
import duckdb
import time
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Output path for DuckDB file
OUTPUT_DIR = Path("data/database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = OUTPUT_DIR / "organoid_data.duckdb"
PROGRESS_FILE = OUTPUT_DIR / "export_progress.json"

class ProgressTracker:
    """Track and report export progress."""
    
    def __init__(self):
        self.start_time = time.time()
        self.current_task = ""
        self.total_tasks = 0
        self.completed_tasks = 0
        self.current_rows = 0
        self.total_rows = 0
        self.errors = []
        
    def update(self, task="", rows_done=0, rows_total=0, status="running"):
        """Update progress and write to file."""
        self.current_task = task
        self.current_rows = rows_done
        self.total_rows = rows_total
        
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if rows_total > 0 and rows_done > 0:
            rate = rows_done / elapsed
            remaining_rows = rows_total - rows_done
            eta_seconds = remaining_rows / rate if rate > 0 else 0
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
        
        # Progress percentage
        if self.total_tasks > 0:
            task_progress = (self.completed_tasks / self.total_tasks) * 100
        else:
            task_progress = 0
            
        if rows_total > 0:
            row_progress = (rows_done / rows_total) * 100
        else:
            row_progress = 0
        
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_formatted": str(timedelta(seconds=int(elapsed))),
            "current_task": task,
            "status": status,
            "task_progress": round(task_progress, 1),
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "current_rows": rows_done,
            "total_rows": rows_total,
            "row_progress": round(row_progress, 1),
            "eta": eta,
            "errors": len(self.errors),
            "last_error": self.errors[-1] if self.errors else None
        }
        
        # Write to file
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Also print to console with clear formatting
        print(f"\r[{elapsed:.0f}s] {task}: {rows_done:,}/{rows_total:,} rows ({row_progress:.1f}%) - ETA: {eta}", 
              end='', flush=True)
        
        if status == "completed":
            print()  # New line after completion
    
    def complete_task(self):
        """Mark current task as complete."""
        self.completed_tasks += 1
        
    def add_error(self, error):
        """Add an error to the list."""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "task": self.current_task,
            "error": str(error)
        })

def export_to_duckdb():
    """Export all tables from PostgreSQL to local DuckDB file."""
    
    progress = ProgressTracker()
    
    print("=" * 70)
    print("Database Export to DuckDB - Interactive Progress")
    print("=" * 70)
    print(f"Output file: {DB_FILE}")
    print(f"Progress tracking: {PROGRESS_FILE}")
    print()
    
    # Remove existing file if it exists
    if DB_FILE.exists():
        print(f"Removing existing database file...")
        DB_FILE.unlink()
    
    # Create new DuckDB connection
    local_conn = duckdb.connect(str(DB_FILE))
    
    with DataLoader(use_local=False) as loader:  # Force remote
        print("Connected to remote database")
        
        # Define tables to export
        small_tables = [
            ("drugs", "SELECT * FROM db.public.drugs"),
            ("event_table", "SELECT * FROM db.public.event_table"),
            ("well_map_data", "SELECT * FROM db.public.well_map_data"),
            ("plate_table", "SELECT * FROM db.public.plate_table"),
        ]
        
        # Count total tasks (small tables + processed_data plates)
        progress.total_tasks = len(small_tables) + 1  # +1 for counting plates
        
        # Export small tables
        for table_name, query in small_tables:
            try:
                progress.update(f"Exporting {table_name}", 0, 0)
                start_time = time.time()
                
                # Get count first
                count_query = query.replace("SELECT *", "SELECT COUNT(*) as count")
                count_result = loader._execute_and_convert(count_query)
                total_count = count_result['count'].iloc[0] if not count_result.empty else 0
                
                progress.update(f"Exporting {table_name}", 0, total_count)
                
                # Execute query and get dataframe
                df = loader._execute_and_convert(query)
                
                # Create table in DuckDB
                local_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                
                elapsed = time.time() - start_time
                progress.update(f"Completed {table_name}", len(df), len(df), "completed")
                progress.complete_task()
                
                print(f"  ✓ Exported {len(df):,} rows in {elapsed:.2f}s")
                
            except Exception as e:
                progress.add_error(f"{table_name}: {e}")
                print(f"\n  ✗ Error exporting {table_name}: {e}")
                continue
        
        # Export processed_data in chunks
        print(f"\nPreparing to export processed_data...")
        try:
            # Get total count
            count_query = "SELECT COUNT(*) as count FROM db.public.processed_data"
            total_rows = loader._execute_and_convert(count_query)['count'].iloc[0]
            
            progress.update("Counting plates in processed_data", 0, total_rows)
            
            # Get list of plates
            plates_query = """
                SELECT DISTINCT plate_id, COUNT(*) as row_count 
                FROM db.public.processed_data 
                GROUP BY plate_id 
                ORDER BY row_count
            """
            plates_df = loader._execute_and_convert(plates_query)
            num_plates = len(plates_df)
            
            progress.total_tasks = len(small_tables) + num_plates
            progress.complete_task()  # Counting was a task
            
            print(f"\n  Total rows: {total_rows:,}")
            print(f"  Plates to export: {num_plates}")
            
            # Create table structure
            progress.update("Creating processed_data table structure", 0, 0)
            schema_query = "SELECT * FROM db.public.processed_data LIMIT 1"
            schema_df = loader._execute_and_convert(schema_query)
            empty_df = schema_df.iloc[0:0].copy()
            local_conn.execute("CREATE TABLE processed_data AS SELECT * FROM empty_df")
            
            # Export each plate
            exported_rows = 0
            export_start = time.time()
            
            for idx, (plate_id, row_count) in enumerate(zip(plates_df['plate_id'], plates_df['row_count'])):
                progress.update(
                    f"Exporting plate {idx+1}/{num_plates}", 
                    exported_rows, 
                    total_rows
                )
                
                plate_query = f"""
                    SELECT * FROM db.public.processed_data 
                    WHERE plate_id = '{plate_id}'
                """
                
                try:
                    plate_start = time.time()
                    plate_df = loader._execute_and_convert(plate_query)
                    local_conn.execute("INSERT INTO processed_data SELECT * FROM plate_df")
                    exported_rows += len(plate_df)
                    plate_time = time.time() - plate_start
                    
                    progress.complete_task()
                    
                    # Update with actual exported count
                    progress.update(
                        f"Completed plate {idx+1}/{num_plates} ({plate_time:.1f}s)", 
                        exported_rows, 
                        total_rows
                    )
                    
                except Exception as e:
                    progress.add_error(f"Plate {plate_id}: {e}")
                    print(f"\n  ✗ Error on plate {plate_id}: {e}")
                    continue
            
            total_elapsed = time.time() - export_start
            progress.update("Completed processed_data export", exported_rows, exported_rows, "completed")
            print(f"\n  ✓ Exported {exported_rows:,} rows in {total_elapsed:.2f}s")
            
        except Exception as e:
            progress.add_error(f"processed_data: {e}")
            print(f"\n  ✗ Error exporting processed_data: {e}")
        
        # Create indexes
        print("\nCreating indexes...")
        indexes = [
            ("idx_processed_plate", "CREATE INDEX idx_processed_plate ON processed_data(plate_id)"),
            ("idx_processed_well", "CREATE INDEX idx_processed_well ON processed_data(plate_id, well_number)"),
            ("idx_processed_time", "CREATE INDEX idx_processed_time ON processed_data(timestamp)"),
            ("idx_event_plate", "CREATE INDEX idx_event_plate ON event_table(plate_id)"),
            ("idx_event_time", "CREATE INDEX idx_event_time ON event_table(occurred_at)"),
            ("idx_wellmap", "CREATE INDEX idx_wellmap ON well_map_data(plate_id, well_number)"),
        ]
        
        for idx_name, idx_query in indexes:
            try:
                progress.update(f"Creating index {idx_name}", 0, 0)
                local_conn.execute(idx_query)
                print(f"  ✓ Created index: {idx_name}")
            except Exception as e:
                progress.add_error(f"Index {idx_name}: {e}")
                print(f"  ✗ Error creating index {idx_name}: {e}")
    
    # Optimize database
    progress.update("Optimizing database", 0, 0)
    local_conn.execute("ANALYZE")
    
    # Get final statistics
    print("\nDatabase statistics:")
    for table in ["drugs", "event_table", "well_map_data", "plate_table", "processed_data"]:
        try:
            count = local_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count:,} rows")
        except:
            pass
    
    # Get file size
    file_size_mb = DB_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDatabase file size: {file_size_mb:.1f} MB")
    
    local_conn.close()
    
    # Final progress update
    progress.update("Export complete!", progress.current_rows, progress.current_rows, "completed")
    
    print(f"\n✓ Export complete! Database saved to: {DB_FILE}")
    print(f"Progress tracking saved to: {PROGRESS_FILE}")
    
    return DB_FILE

if __name__ == "__main__":
    export_to_duckdb()