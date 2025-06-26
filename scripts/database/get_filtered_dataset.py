#!/usr/bin/env python
"""
Get the filtered dataset based on quality criteria.

This script returns wells that meet our quality criteria:
1. From plates with >300h duration and media changes
2. Exclude drugs with .number suffixes or (mg/ml) annotations
3. Include control wells (concentration = 0)
4. Only include drugs that exist in metadata
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.utils.data_loader import DataLoader
import re


def get_good_plates(loader):
    """Get plates that meet quality criteria."""
    plates = loader.load_plate_table()
    events = loader.load_media_events()
    
    # Calculate plate durations
    oxygen_data = loader.load_oxygen_data()
    plate_durations = oxygen_data.groupby('plate_id')['elapsed_hours'].max()
    plate_durations.name = 'duration_hours'
    
    # Get plates with media changes
    plates_with_media = events['plate_id'].unique()
    
    # Filter good plates
    good_plates = plates[
        (plates['id'].isin(plate_durations[plate_durations > 300].index)) &
        (plates['id'].isin(plates_with_media))
    ]['id'].tolist()
    
    return good_plates


def filter_wells(loader, good_plates):
    """Filter wells based on drug exclusion criteria."""
    wells = loader.load_well_metadata()
    drugs = loader.load_drug_metadata()
    
    # Filter to good plates
    wells = wells[wells['plate_id'].isin(good_plates)].copy()
    
    # Get valid drug names
    valid_drugs = set(drugs['drug'].unique())
    
    # Function to check if drug should be excluded
    def should_exclude_drug(drug_name):
        if pd.isna(drug_name) or drug_name == '':
            return False  # Empty names are controls
        
        # Check for .number suffixes
        if re.match(r'.*\.\d+$', drug_name):
            return True
        
        # Check for (mg/ml)
        if '(mg/ml)' in drug_name.lower():
            return True
        
        # Check if in valid drugs
        if drug_name not in valid_drugs:
            return True
            
        return False
    
    # Mark wells for exclusion
    wells['exclude'] = wells.apply(
        lambda row: should_exclude_drug(row['drug']) if row['concentration'] > 0 else False,
        axis=1
    )
    
    # Mark control wells
    wells['is_control'] = wells['concentration'] == 0
    
    # Get filtered wells
    filtered_wells = wells[~wells['exclude']].copy()
    
    return filtered_wells


def get_filtered_dataset():
    """Get the complete filtered dataset."""
    with DataLoader() as loader:
        # Get good plates
        good_plates = get_good_plates(loader)
        
        # Filter wells
        filtered_wells = filter_wells(loader, good_plates)
        
        # Add DILI information
        drugs = loader.load_drug_metadata()
        filtered_wells = filtered_wells.merge(
            drugs[['drug', 'binary_dili', 'dili_risk_score']],
            on='drug',
            how='left'
        )
        
        return filtered_wells, good_plates


def main():
    """Print summary of filtered dataset."""
    filtered_wells, good_plates = get_filtered_dataset()
    
    print("=== FILTERED DATASET SUMMARY ===\n")
    print(f"Good quality plates: {len(good_plates)}")
    print(f"Total wells: {len(filtered_wells):,}")
    print(f"Control wells: {filtered_wells['is_control'].sum():,}")
    print(f"Treatment wells: {(~filtered_wells['is_control']).sum():,}")
    print(f"Unique drugs: {filtered_wells[~filtered_wells['is_control']]['drug'].nunique()}")
    
    # DILI breakdown
    treatment_wells = filtered_wells[~filtered_wells['is_control']]
    print(f"\nDILI Status:")
    print(f"  Positive: {(treatment_wells['binary_dili'] == 1).sum():,} wells")
    print(f"  Negative: {(treatment_wells['binary_dili'] == 0).sum():,} wells")
    print(f"  Unknown: {treatment_wells['binary_dili'].isna().sum():,} wells")
    
    # Plate breakdown
    print(f"\nWells per plate:")
    plate_wells = filtered_wells.groupby('plate_id').size().sort_values(ascending=False)
    for plate_id, count in plate_wells.items():
        control_count = filtered_wells[
            (filtered_wells['plate_id'] == plate_id) & 
            (filtered_wells['is_control'])
        ].shape[0]
        print(f"  {plate_id}: {count} wells ({control_count} controls)")
    
    # Save filtered wells
    output_file = Path("results/data/filtered_wells.csv")
    filtered_wells.to_csv(output_file, index=False)
    print(f"\nFiltered wells saved to: {output_file}")
    
    return filtered_wells


if __name__ == "__main__":
    filtered_wells = main()