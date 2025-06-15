#!/usr/bin/env python3
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
    print("\n1. Verifying real data availability...")
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
        print(f"\n2. Running {description}...")
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
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED!")
    print("Check the experiments/results/ directory for outputs")
    print("=" * 80)

if __name__ == "__main__":
    run_real_data_experiments()
