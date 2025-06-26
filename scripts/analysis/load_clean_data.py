#!/usr/bin/env python
"""
Load clean, filtered data for analysis.

This provides a simple interface to get quality-filtered data:
- 22 good plates with >300h duration and media changes
- 4,468 wells (1,000 controls, 3,468 treatments)  
- 123 unique drugs with DILI annotations
- All drugs mapped to metadata
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.utils.data_loader import DataLoader


def load_clean_wells():
    """Load the filtered, clean wells dataset."""
    wells_file = Path("results/data/filtered_wells.csv")
    
    if not wells_file.exists():
        print("Filtered wells file not found. Run scripts/database/get_filtered_dataset.py first.")
        return None
    
    return pd.read_csv(wells_file)


def load_clean_oxygen_data(plate_ids=None):
    """Load oxygen data for clean plates only."""
    with DataLoader() as loader:
        wells = load_clean_wells()
        if wells is None:
            return None
        
        if plate_ids is None:
            plate_ids = wells['plate_id'].unique().tolist()
        
        # Load oxygen data for these plates
        oxygen_data = loader.load_oxygen_data(plate_ids=plate_ids)
        
        # Filter to only clean wells
        clean_well_ids = wells['well_id'].unique()
        oxygen_data = oxygen_data[oxygen_data['well_id'].isin(clean_well_ids)]
        
        return oxygen_data


def get_treatment_wells():
    """Get only treatment wells (not controls)."""
    wells = load_clean_wells()
    if wells is None:
        return None
    
    return wells[~wells['is_control']]


def get_control_wells():
    """Get only control wells."""
    wells = load_clean_wells()
    if wells is None:
        return None
    
    return wells[wells['is_control']]


def get_dili_positive_wells():
    """Get wells with DILI-positive drugs."""
    wells = get_treatment_wells()
    if wells is None:
        return None
    
    return wells[wells['binary_dili'] == 1]


def get_dili_negative_wells():
    """Get wells with DILI-negative drugs."""
    wells = get_treatment_wells()
    if wells is None:
        return None
    
    return wells[wells['binary_dili'] == 0]


def main():
    """Example usage of the clean data functions."""
    print("=== CLEAN DATA SUMMARY ===\n")
    
    wells = load_clean_wells()
    if wells is None:
        return
    
    print(f"Total wells: {len(wells):,}")
    print(f"Treatment wells: {len(get_treatment_wells()):,}")
    print(f"Control wells: {len(get_control_wells()):,}")
    print(f"DILI positive wells: {len(get_dili_positive_wells()):,}")
    print(f"DILI negative wells: {len(get_dili_negative_wells()):,}")
    
    print(f"\nUnique drugs: {wells[~wells['is_control']]['drug'].nunique()}")
    print(f"Unique plates: {wells['plate_id'].nunique()}")
    
    # Show example of loading oxygen data
    print(f"\n=== EXAMPLE: Loading oxygen data for first plate ===")
    first_plate = wells['plate_id'].iloc[0]
    oxygen_data = load_clean_oxygen_data([first_plate])
    print(f"Oxygen measurements for plate {first_plate}: {len(oxygen_data):,} records")
    print(f"Time range: {oxygen_data['elapsed_hours'].min():.1f} - {oxygen_data['elapsed_hours'].max():.1f} hours")
    
    # Show drug distribution
    print(f"\n=== TOP DRUGS BY WELL COUNT ===")
    drug_counts = wells[~wells['is_control']]['drug'].value_counts().head(10)
    for drug, count in drug_counts.items():
        dili_status = wells[wells['drug'] == drug]['binary_dili'].iloc[0]
        status = "DILI+" if dili_status == 1 else "DILI-"
        print(f"  {drug}: {count} wells ({status})")


if __name__ == "__main__":
    main()