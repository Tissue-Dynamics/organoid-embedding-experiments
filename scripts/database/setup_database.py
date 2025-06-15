#!/usr/bin/env python3
"""Set up database schema and help upload organoid data."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_organoid_data():
    """Create sample organoid data that matches your expected schema."""
    logger.info("🧬 Creating sample organoid data...")
    
    # Create a sample plate
    plate_data = {
        'id': 'PLATE_001',
        'name': 'Liver Organoid Toxicity Screen',
        'description': 'Test plate for organoid embedding experiments',
        'created_at': datetime.now().isoformat(),
        'experiment_type': 'toxicity_screen'
    }
    
    # Create well mapping (96-well plate)
    well_map_data = []
    treatments = [
        'Control', 'APAP_10uM', 'APAP_100uM', 'APAP_1mM',
        'Paracetamol_10uM', 'Paracetamol_100uM', 'Rifampicin_10uM', 'Rifampicin_100uM'
    ]
    
    well_counter = 1
    for row in range(8):  # A-H
        for col in range(12):  # 1-12
            well_name = f"{chr(65+row)}{col+1:02d}"
            treatment = treatments[well_counter % len(treatments)]
            concentration = "Control" if treatment == "Control" else treatment.split('_')[1]
            
            well_map_data.append({
                'plate_id': 'PLATE_001',
                'well_number': well_counter,
                'well_name': well_name,
                'row': chr(65+row),
                'column': col+1,
                'treatment': treatment.split('_')[0],
                'concentration': concentration,
                'replicate': (well_counter % 4) + 1
            })
            well_counter += 1
    
    # Create time series data (2 weeks of hourly measurements)
    logger.info("Generating time series data...")
    time_series_data = []
    
    start_time = datetime.now() - timedelta(days=14)
    n_timepoints = 24 * 14  # 2 weeks hourly
    
    for well in well_map_data[:24]:  # Use first 24 wells for demo
        well_id = well['well_number']
        treatment = well['treatment']
        
        for hour in range(n_timepoints):
            timestamp = start_time + timedelta(hours=hour)
            
            # Generate realistic oxygen consumption patterns
            baseline = 80 + np.random.normal(0, 3)
            
            # Add circadian rhythm
            circadian = 8 * np.sin(2 * np.pi * hour / 24)
            
            # Add treatment effects
            if treatment == 'Control':
                treatment_effect = 0
            elif 'APAP' in treatment:
                # Gradual decline after 48h
                treatment_effect = -max(0, (hour - 48) / 20) * np.random.uniform(0.5, 2.0)
            else:
                # Other treatments
                treatment_effect = -max(0, (hour - 24) / 15) * np.random.uniform(0.3, 1.5)
            
            # Add noise
            noise = np.random.normal(0, 2)
            
            # Final value
            value = baseline + circadian + treatment_effect + noise
            value = max(10, min(100, value))  # Keep in reasonable range
            
            # Add some missing values (5% chance)
            if np.random.random() < 0.05:
                value = None
            
            time_series_data.append({
                'plate_id': 'PLATE_001',
                'well_number': well_id,
                'timestamp': timestamp.isoformat(),
                'value': value,
                'measurement_type': 'oxygen_consumption'
            })
    
    logger.info(f"Created {len(time_series_data)} time series measurements")
    
    return plate_data, well_map_data, time_series_data

def upload_to_supabase(plate_data, well_map_data, time_series_data):
    """Upload the sample data to Supabase."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info("📤 Uploading data to Supabase...")
    
    try:
        # Upload plate data
        logger.info("Uploading plate data...")
        plate_result = loader.client.table('plate_table').insert(plate_data).execute()
        logger.info(f"✅ Uploaded 1 plate record")
        
        # Upload well mapping
        logger.info("Uploading well mapping...")
        # Create well_map table if it doesn't exist and upload in batches
        batch_size = 50
        for i in range(0, len(well_map_data), batch_size):
            batch = well_map_data[i:i+batch_size]
            try:
                well_result = loader.client.table('well_map').insert(batch).execute()
                logger.info(f"✅ Uploaded well mapping batch {i//batch_size + 1}")
            except Exception as e:
                logger.warning(f"Well mapping batch {i//batch_size + 1} failed: {e}")
        
        # Upload time series data
        logger.info("Uploading time series data...")
        batch_size = 100
        for i in range(0, len(time_series_data), batch_size):
            batch = time_series_data[i:i+batch_size]
            # Filter out None values
            batch = [record for record in batch if record['value'] is not None]
            
            try:
                ts_result = loader.client.table('time_series').insert(batch).execute()
                logger.info(f"✅ Uploaded time series batch {i//batch_size + 1}")
            except Exception as e:
                logger.warning(f"Time series batch {i//batch_size + 1} failed: {e}")
        
        logger.info("🎉 Data upload completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        logger.info("💡 Note: You may need to create the tables in Supabase first:")
        logger.info("1. Go to your Supabase dashboard")
        logger.info("2. Go to Table Editor")
        logger.info("3. Create tables: well_map, time_series")
        return False

def show_table_schemas():
    """Show the expected table schemas."""
    logger.info("\n📋 Expected table schemas:")
    
    logger.info("\n1. plate_table (already exists):")
    logger.info("   - id (text, primary key)")
    logger.info("   - name (text)")
    logger.info("   - description (text)")
    logger.info("   - created_at (timestamp)")
    logger.info("   - experiment_type (text)")
    
    logger.info("\n2. well_map (needs to be created):")
    logger.info("   - plate_id (text, foreign key)")
    logger.info("   - well_number (integer)")
    logger.info("   - well_name (text)")
    logger.info("   - row (text)")
    logger.info("   - column (integer)")
    logger.info("   - treatment (text)")
    logger.info("   - concentration (text)")
    logger.info("   - replicate (integer)")
    
    logger.info("\n3. time_series (needs to be created):")
    logger.info("   - plate_id (text, foreign key)")
    logger.info("   - well_number (integer)")
    logger.info("   - timestamp (timestamp)")
    logger.info("   - value (real/float)")
    logger.info("   - measurement_type (text)")

def create_sql_schemas():
    """Generate SQL to create the required tables."""
    logger.info("\n📝 SQL to create tables in Supabase:")
    
    sql_commands = [
        """
-- Well mapping table
CREATE TABLE well_map (
    id SERIAL PRIMARY KEY,
    plate_id TEXT REFERENCES plate_table(id),
    well_number INTEGER NOT NULL,
    well_name TEXT NOT NULL,
    row TEXT NOT NULL,
    column INTEGER NOT NULL,
    treatment TEXT,
    concentration TEXT,
    replicate INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
""",
        """
-- Time series data table
CREATE TABLE time_series (
    id SERIAL PRIMARY KEY,
    plate_id TEXT REFERENCES plate_table(id),
    well_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    value REAL,
    measurement_type TEXT DEFAULT 'oxygen_consumption',
    created_at TIMESTAMP DEFAULT NOW()
);
""",
        """
-- Indexes for better performance
CREATE INDEX idx_time_series_plate_well ON time_series(plate_id, well_number);
CREATE INDEX idx_time_series_timestamp ON time_series(timestamp);
CREATE INDEX idx_well_map_plate ON well_map(plate_id);
"""
    ]
    
    for i, sql in enumerate(sql_commands, 1):
        logger.info(f"\n{i}. {sql.strip()}")

if __name__ == "__main__":
    logger.info("🛠️  Setting up organoid database...")
    
    try:
        # Show what we need to create
        show_table_schemas()
        create_sql_schemas()
        
        # Create sample data
        plate_data, well_map_data, time_series_data = create_sample_organoid_data()
        
        # Try to upload (will fail if tables don't exist)
        logger.info("\n🚀 Attempting to upload sample data...")
        success = upload_to_supabase(plate_data, well_map_data, time_series_data)
        
        if success:
            logger.info("\n✅ SUCCESS! Sample organoid data uploaded.")
            logger.info("Now you can run: python test_real_data.py")
        else:
            logger.info("\n⚠️  Upload failed. Please:")
            logger.info("1. Go to your Supabase dashboard")
            logger.info("2. Use SQL Editor to run the SQL commands above")
            logger.info("3. Run this script again")
            
        logger.info("\n💡 Or upload your own data files:")
        logger.info("- Modify this script to load from CSV/Excel files")
        logger.info("- Update the data loader to match your schema")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise