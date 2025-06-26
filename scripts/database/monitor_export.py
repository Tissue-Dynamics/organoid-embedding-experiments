#!/usr/bin/env python3
"""
Monitor database export progress with a nice display.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

PROGRESS_FILE = Path("data/database/export_progress.json")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_progress_bar(progress, width=50):
    """Create a visual progress bar."""
    filled = int(width * progress / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {progress:.1f}%"

def monitor_progress():
    """Monitor export progress with live updates."""
    
    print("Monitoring database export progress...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    last_update = None
    
    try:
        while True:
            if PROGRESS_FILE.exists():
                try:
                    with open(PROGRESS_FILE, 'r') as f:
                        data = json.load(f)
                    
                    # Clear screen for clean display
                    clear_screen()
                    
                    print("=" * 70)
                    print("Database Export Progress Monitor")
                    print("=" * 70)
                    print()
                    
                    # Basic info
                    print(f"Status: {data.get('status', 'unknown').upper()}")
                    print(f"Elapsed: {data.get('elapsed_formatted', 'unknown')}")
                    print(f"ETA: {data.get('eta', 'unknown')}")
                    print()
                    
                    # Current task
                    current_task = data.get('current_task', 'Initializing...')
                    print(f"Current Task: {current_task}")
                    
                    # Task progress
                    task_progress = data.get('task_progress', 0)
                    completed = data.get('completed_tasks', 0)
                    total = data.get('total_tasks', 0)
                    print(f"Tasks: {completed}/{total} completed")
                    print(format_progress_bar(task_progress))
                    print()
                    
                    # Row progress for current task
                    if data.get('total_rows', 0) > 0:
                        current_rows = data.get('current_rows', 0)
                        total_rows = data.get('total_rows', 0)
                        row_progress = data.get('row_progress', 0)
                        
                        print(f"Rows: {current_rows:,}/{total_rows:,}")
                        print(format_progress_bar(row_progress))
                        print()
                    
                    # Errors
                    error_count = data.get('errors', 0)
                    if error_count > 0:
                        print(f"⚠️  Errors: {error_count}")
                        last_error = data.get('last_error')
                        if last_error:
                            print(f"   Last: {last_error.get('task', 'unknown')} - {last_error.get('error', '')[:50]}...")
                        print()
                    
                    # Last update
                    timestamp = data.get('timestamp')
                    if timestamp:
                        update_time = datetime.fromisoformat(timestamp)
                        print(f"Last Update: {update_time.strftime('%H:%M:%S')}")
                    
                    # Check if complete
                    if data.get('status') == 'completed':
                        print()
                        print("✅ Export Complete!")
                        break
                    
                    last_update = data
                    
                except json.JSONDecodeError:
                    print("Waiting for valid progress data...")
                except Exception as e:
                    print(f"Error reading progress: {e}")
            else:
                print("Waiting for export to start...")
                print(f"Progress file: {PROGRESS_FILE}")
            
            time.sleep(1)  # Update every second
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if last_update and last_update.get('status') != 'completed':
            print("Export is still running in the background.")

if __name__ == "__main__":
    monitor_progress()