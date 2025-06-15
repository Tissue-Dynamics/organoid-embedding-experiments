#!/usr/bin/env python3
"""
Template for importing processes dat table data into Supabase
Customize this script based on your data source format
"""

import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import os
from datetime import datetime

def import_processes_dat_table(data_file_path: str):
    """
    Import processes dat table data from CSV/Excel file
    
    Args:
        data_file_path: Path to your processes dat table file
    """
    load_dotenv()
    
    # Connect to Supabase
    url = os.getenv('SUPABASE_URL')  # or SONIC_SUPABASE_URL
    key = os.getenv('SUPABASE_KEY')  # or SONIC_SUPABASE_KEY
    client = create_client(url, key)
    
    print(f"Loading data from: {data_file_path}")
    
    # Load data (adapt based on your file format)
    if data_file_path.endswith('.csv'):
        df = pd.read_csv(data_file_path)
    elif data_file_path.endswith('.xlsx') or data_file_path.endswith('.xls'):
        df = pd.read_excel(data_file_path)
    else:
        raise ValueError("Supported formats: CSV, Excel")
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data preprocessing and mapping
    # TODO: Customize these mappings based on your actual column names
    
    # Expected columns after mapping:
    # - plate_id: Unique identifier for each plate
    # - well_number: Well position (1-96 for 96-well plate)
    # - timestamp: Time of measurement
    # - median_o2: Oxygen measurement value
    # - drug: Drug name
    # - concentration: Drug concentration
    # - replicate: Replicate number
    
    # Example mapping (customize based on your data):
    column_mapping = {
        # 'your_plate_column': 'plate_id',
        # 'your_well_column': 'well_number',
        # 'your_time_column': 'timestamp',
        # 'your_o2_column': 'median_o2',
        # 'your_drug_column': 'drug',
        # 'your_concentration_column': 'concentration'
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Data cleaning and validation
    # TODO: Add your specific cleaning steps
    
    # Convert timestamp to proper format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add metadata columns if missing
    df['is_excluded'] = False
    df['created_at'] = datetime.now().isoformat()
    
    # Split data for different tables
    
    # 1. Plate information
    plate_data = df.groupby('plate_id').first().reset_index()
    plate_records = plate_data[['plate_id', 'created_at']].to_dict('records')
    
    # 2. Well mapping (experimental design)
    well_map_columns = ['plate_id', 'well_number', 'drug', 'concentration', 'is_excluded']
    well_map_data = df[well_map_columns].drop_duplicates().to_dict('records')
    
    # 3. Time series data
    timeseries_columns = ['plate_id', 'well_number', 'timestamp', 'median_o2', 'is_excluded']
    timeseries_data = df[timeseries_columns].to_dict('records')
    
    # Insert data into Supabase tables
    print("Inserting plate data...")
    try:
        client.table('plate_table').insert(plate_records).execute()
        print(f"✅ Inserted {len(plate_records)} plates")
    except Exception as e:
        print(f"❌ Error inserting plates: {e}")
    
    print("Inserting well mapping data...")
    try:
        # Insert in batches
        batch_size = 1000
        for i in range(0, len(well_map_data), batch_size):
            batch = well_map_data[i:i+batch_size]
            client.table('well_map_data').insert(batch).execute()
        print(f"✅ Inserted {len(well_map_data)} well mappings")
    except Exception as e:
        print(f"❌ Error inserting well mappings: {e}")
    
    print("Inserting time series data...")
    try:
        # Insert in batches
        batch_size = 1000
        for i in range(0, len(timeseries_data), batch_size):
            batch = timeseries_data[i:i+batch_size]
            client.table('processed_data').insert(batch).execute()
            if i % 10000 == 0:
                print(f"  Inserted {i}/{len(timeseries_data)} time series points...")
        print(f"✅ Inserted {len(timeseries_data)} time series data points")
    except Exception as e:
        print(f"❌ Error inserting time series data: {e}")
    
    print("\n✅ Data import completed!")
    print("Run analyze_real_organoid_data.py to verify the import")

if __name__ == "__main__":
    # TODO: Update this path to your processes dat table file
    data_file = "/path/to/your/processes_dat_table.csv"
    
    if os.path.exists(data_file):
        import_processes_dat_table(data_file)
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please update the data_file path in this script")
