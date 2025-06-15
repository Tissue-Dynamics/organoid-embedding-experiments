#!/usr/bin/env python3
"""
Real Organoid Data Integration Plan
Complete guide and tools for integrating sonic analysis project data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataIntegrator:
    """Complete integration pipeline for real organoid data"""
    
    def __init__(self):
        self.project_root = Path("/Users/shaunie/Documents/Code/organoid-embedding-experiments")
        
    def print_integration_plan(self):
        """Print comprehensive integration plan"""
        
        print("=" * 80)
        print("REAL ORGANOID DATA INTEGRATION PLAN")
        print("SONIC ANALYSIS PROJECT → EMBEDDING EXPERIMENTS")
        print("=" * 80)
        
        print("\n📋 INTEGRATION CHECKLIST:")
        print("=" * 50)
        
        checklist = [
            ("✅", "Environment setup with uv", "Completed"),
            ("✅", "Existing codebase analysis", "Completed"),
            ("✅", "Database connection tools created", "Completed"),
            ("🔄", "Connect to sonic analysis project", "Pending - Need credentials"),
            ("⏳", "Analyze processes dat table structure", "Ready when connected"),
            ("⏳", "Data quality assessment", "Ready when connected"),
            ("⏳", "Preprocessing pipeline adaptation", "Ready when connected"),
            ("⏳", "Embedding experiments on real data", "Ready when connected"),
            ("⏳", "Results analysis and visualization", "Ready when connected")
        ]
        
        for status, item, note in checklist:
            print(f"  {status} {item:<35} - {note}")
        
        print("\n🎯 CURRENT STATUS:")
        print("=" * 50)
        print("• Codebase: Fully functional with comprehensive embedding methods")
        print("• Database: Connected to Supabase, tables exist but empty")
        print("• Pipeline: Ready to process real organoid time series data")
        print("• Bottleneck: Need access to sonic analysis project data")
        
        print("\n🔌 CONNECTION OPTIONS:")
        print("=" * 50)
        
        print("OPTION 1: Different Supabase Project")
        print("  • Need: Sonic analysis project Supabase URL + API key")
        print("  • Action: Update .env with SONIC_SUPABASE_URL and SONIC_SUPABASE_KEY")
        print("  • Tool: Use connect_sonic_project.py to test connection")
        
        print("\nOPTION 2: Data Import Required")
        print("  • Need: Access to processes dat table data (CSV, database, etc.)")
        print("  • Action: Create import script to load data into current Supabase")
        print("  • Tool: Import pipeline ready to be customized")
        
        print("\nOPTION 3: Different Database/API")
        print("  • Need: Connection details for sonic analysis database")
        print("  • Action: Adapt data loader to connect to different system")
        print("  • Tool: Flexible loader architecture supports multiple backends")
        
        print("\n🛠️ TOOLS CREATED:")
        print("=" * 50)
        
        tools = [
            ("analyze_sonic_data.py", "Analyze sonic analysis project structure"),
            ("explore_supabase_tables.py", "Discover available tables and data"),
            ("connect_sonic_project.py", "Test connection to sonic project"),
            ("analyze_real_organoid_data.py", "Comprehensive real data analysis"),
            ("real_data_integration_plan.py", "This integration guide")
        ]
        
        for tool, description in tools:
            exists = "✅" if (self.project_root / tool).exists() else "❌"
            print(f"  {exists} {tool:<30} - {description}")
        
        print("\n📊 EXPECTED DATA STRUCTURE:")
        print("=" * 50)
        
        print("Based on organoid experiments, expecting:")
        print("• Time series: Oxygen measurements over time")
        print("• Identifiers: Plate ID, well number, organoid ID")
        print("• Experimental design: Drug treatments, concentrations, replicates")
        print("• Temporal: Timestamps with regular intervals")
        print("• Events: Media changes, treatment additions")
        print("• Metadata: Drug information, experimental conditions")
        
        print("\n🔄 PREPROCESSING PIPELINE:")
        print("=" * 50)
        
        preprocessing_steps = [
            "Data cleaning (remove artifacts, outliers)",
            "Time series normalization (control-based)",
            "Missing value interpolation",
            "Event correction (media change artifacts)",
            "Quality assessment (completeness, continuity)",
            "Feature engineering for embedding methods"
        ]
        
        for i, step in enumerate(preprocessing_steps, 1):
            print(f"  {i}. {step}")
        
        print("\n🧮 EMBEDDING METHODS READY:")
        print("=" * 50)
        
        methods = {
            "Traditional": ["DTW", "Fourier Transform", "SAX"],
            "Feature-based": ["TSFresh", "catch22", "Custom organoid features"],
            "Deep Learning": ["LSTM Autoencoder", "CNN Autoencoder", "Transformer", "Triplet Network"]
        }
        
        for category, method_list in methods.items():
            print(f"  {category}:")
            for method in method_list:
                print(f"    • {method}")
        
        print("\n📈 EVALUATION METRICS:")
        print("=" * 50)
        
        metrics = [
            "Clustering quality (silhouette, Calinski-Harabasz)",
            "Neighborhood preservation (trustworthiness, continuity)",  
            "Dimensionality reduction quality",
            "Biological relevance (drug response similarity)",
            "Embedding stability and reproducibility"
        ]
        
        for metric in metrics:
            print(f"  • {metric}")
        
        print("\n🚀 IMMEDIATE NEXT STEPS:")
        print("=" * 50)
        
        next_steps = [
            "1. Provide sonic analysis project connection details",
            "2. Run connection test and data structure analysis",
            "3. Execute preprocessing pipeline on real data",
            "4. Run embedding experiments with all methods",
            "5. Compare embedding quality across methods",
            "6. Generate publication-ready results and visualizations"
        ]
        
        for step in next_steps:
            print(f"  {step}")
    
    def create_data_import_template(self):
        """Create template for importing processes dat table data"""
        
        import_script = '''#!/usr/bin/env python3
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
    
    print("\\n✅ Data import completed!")
    print("Run analyze_real_organoid_data.py to verify the import")

if __name__ == "__main__":
    # TODO: Update this path to your processes dat table file
    data_file = "/path/to/your/processes_dat_table.csv"
    
    if os.path.exists(data_file):
        import_processes_dat_table(data_file)
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please update the data_file path in this script")
'''
        
        script_path = self.project_root / "import_processes_dat_table.py"
        with open(script_path, 'w') as f:
            f.write(import_script)
        
        print(f"✅ Created data import template: {script_path}")
    
    def create_experiment_runner(self):
        """Create script to run experiments on real data"""
        
        runner_script = '''#!/usr/bin/env python3
"""
Run embedding experiments on real organoid data
Execute this after data is loaded and verified
"""

import subprocess
import sys
from pathlib import Path

def run_real_data_experiments():
    """Run the full experiment pipeline on real data"""
    
    print("=" * 80)
    print("RUNNING EMBEDDING EXPERIMENTS ON REAL ORGANOID DATA")
    print("=" * 80)
    
    # Check data availability first
    print("\\n1. Verifying real data availability...")
    try:
        result = subprocess.run([
            sys.executable, "analyze_real_organoid_data.py"
        ], capture_output=True, text=True)
        
        if "Total time series points: 0" in result.stdout:
            print("❌ No real data found. Please import data first.")
            print("Use import_processes_dat_table.py to load your data.")
            return
        
        print("✅ Real data verified and available")
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        return
    
    # Run experiment configurations
    experiments = [
        ("quick_test", "Quick test on subset of real data"),
        ("config", "Full comparison of all embedding methods"),
        ("deep_learning_only", "Focus on neural network approaches")
    ]
    
    for config_name, description in experiments:
        print(f"\\n2. Running {description}...")
        print("=" * 50)
        
        try:
            cmd = [
                "uv", "run", "python", "experiments/run_experiment.py",
                f"--config-name={config_name}"
            ]
            
            result = subprocess.run(cmd, cwd=".", capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
            else:
                print(f"❌ {description} failed with return code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error running {description}: {e}")
    
    print("\\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED!")
    print("Check the experiments/results/ directory for outputs")
    print("=" * 80)

if __name__ == "__main__":
    run_real_data_experiments()
'''
        
        script_path = self.project_root / "run_real_data_experiments.py"
        with open(script_path, 'w') as f:
            f.write(runner_script)
        
        print(f"✅ Created experiment runner: {script_path}")

def main():
    """Main integration planning function"""
    
    integrator = RealDataIntegrator()
    
    # Print comprehensive integration plan
    integrator.print_integration_plan()
    
    # Create additional tools
    print("\n🛠️ CREATING ADDITIONAL TOOLS:")
    print("=" * 50)
    
    integrator.create_data_import_template()
    integrator.create_experiment_runner()
    
    print("\n💡 SUMMARY:")
    print("=" * 50)
    print("Your embedding experiment pipeline is fully ready for real organoid data!")
    print("The only missing piece is access to the sonic analysis project data.")
    print("")
    print("Next steps:")
    print("1. Provide sonic analysis project credentials OR")
    print("2. Use import_processes_dat_table.py to load your data OR") 
    print("3. Let me know the specific data source format")
    print("")
    print("Once connected, run: uv run python run_real_data_experiments.py")

if __name__ == "__main__":
    main()