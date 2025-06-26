#!/usr/bin/env python3
"""
Quick check of processed_data schema and sample values
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import DataLoader

# Set database URL if not already set
if not os.getenv('DATABASE_URL'):
    os.environ['DATABASE_URL'] = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"

def check_schema():
    """Check the actual data types in processed_data."""
    
    with DataLoader(use_local=False) as loader:
        print("Checking processed_data schema...")
        
        # Get a sample with potential problematic values
        query = """
            SELECT *
            FROM db.public.processed_data
            WHERE 
                exclusion_reason IS NOT NULL 
                OR excluded_by IS NOT NULL
                OR excluded_at IS NOT NULL
            LIMIT 10
        """
        
        df = loader._execute_and_convert(query)
        
        print("\nDataFrame dtypes:")
        print(df.dtypes)
        
        print("\nSample values for potentially problematic columns:")
        for col in ['exclusion_reason', 'excluded_by', 'excluded_at']:
            if col in df.columns:
                print(f"\n{col}:")
                print(f"  Unique values: {df[col].unique()}")
                print(f"  Data type: {df[col].dtype}")

if __name__ == "__main__":
    check_schema()