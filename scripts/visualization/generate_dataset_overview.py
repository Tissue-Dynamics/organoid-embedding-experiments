#!/usr/bin/env python3
"""
Generate Dataset Overview Data for HTML + D3.js Visualizations
Recreates the dataset_overview figures using the new HTML + D3.js approach
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json
from src.utils.data_loader import DataLoader
from scripts.database.get_filtered_dataset import get_filtered_dataset

def generate_dataset_composition_data():
    """Generate data for dataset composition visualization."""
    
    print("📊 Generating dataset composition data...")
    
    filtered_wells, good_plates = get_filtered_dataset()
    
    # Well type composition
    well_composition = {
        'Treatment': int((~filtered_wells['is_control']).sum()),
        'Control': int(filtered_wells['is_control'].sum())
    }
    
    # DILI distribution (treatment wells only)
    treatment_wells = filtered_wells[~filtered_wells['is_control']]
    dili_composition = {
        'DILI Positive': int((treatment_wells['binary_dili'] == 1).sum()),
        'DILI Negative': int((treatment_wells['binary_dili'] == 0).sum()),
        'Unknown DILI': int(treatment_wells['binary_dili'].isna().sum())
    }
    
    # Plate statistics
    plate_stats = filtered_wells.groupby('plate_id').agg({
        'well_id': 'count',
        'drug': lambda x: x[x.notna()].nunique()
    }).rename(columns={'well_id': 'total_wells', 'drug': 'unique_drugs'})
    
    wells_per_plate = {
        'mean': float(plate_stats['total_wells'].mean()),
        'median': float(plate_stats['total_wells'].median()),
        'std': float(plate_stats['total_wells'].std()),
        'distribution': plate_stats['total_wells'].tolist()
    }
    
    drugs_per_plate = {
        'mean': float(plate_stats['unique_drugs'].mean()),
        'median': float(plate_stats['unique_drugs'].median()),
        'std': float(plate_stats['unique_drugs'].std()),
        'distribution': plate_stats['unique_drugs'].tolist()
    }
    
    composition_data = {
        'well_composition': well_composition,
        'dili_composition': dili_composition,
        'wells_per_plate': wells_per_plate,
        'drugs_per_plate': drugs_per_plate,
        'summary': {
            'total_wells': len(filtered_wells),
            'total_plates': len(good_plates),
            'unique_drugs': treatment_wells['drug'].nunique(),
            'dili_positive_rate': float((treatment_wells['binary_dili'] == 1).sum() / len(treatment_wells) * 100)
        }
    }
    
    return composition_data

def generate_plate_summary_data():
    """Generate data for plate summary visualization."""
    
    print("📊 Generating plate summary data...")
    
    filtered_wells, good_plates = get_filtered_dataset()
    
    # Plate-level statistics
    plate_summary = filtered_wells.groupby('plate_id').agg({
        'well_id': 'count',
        'is_control': 'sum',
        'drug': lambda x: x[x.notna()].nunique(),
        'binary_dili': lambda x: (x == 1).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0
    }).rename(columns={
        'well_id': 'total_wells',
        'is_control': 'control_wells',
        'drug': 'unique_drugs',
        'binary_dili': 'dili_positive_rate'
    })
    
    plate_summary['treatment_wells'] = plate_summary['total_wells'] - plate_summary['control_wells']
    
    # Convert to list of dictionaries for D3.js
    plate_data = []
    for plate_id, row in plate_summary.iterrows():
        plate_data.append({
            'plate_id': plate_id,
            'total_wells': int(row['total_wells']),
            'control_wells': int(row['control_wells']),
            'treatment_wells': int(row['treatment_wells']),
            'unique_drugs': int(row['unique_drugs']),
            'dili_positive_rate': float(row['dili_positive_rate'])
        })
    
    # Summary statistics
    summary_stats = {
        'mean_wells_per_plate': float(plate_summary['total_wells'].mean()),
        'mean_controls_per_plate': float(plate_summary['control_wells'].mean()),
        'mean_drugs_per_plate': float(plate_summary['unique_drugs'].mean()),
        'mean_dili_rate': float(plate_summary['dili_positive_rate'].mean()),
        'plate_count': len(plate_data)
    }
    
    return {'plates': plate_data, 'summary': summary_stats}

def generate_dili_drug_analysis_data():
    """Generate data for DILI drug analysis visualization."""
    
    print("📊 Generating DILI drug analysis data...")
    
    filtered_wells, _ = get_filtered_dataset()
    treatment_wells = filtered_wells[~filtered_wells['is_control']]
    
    # Top drugs by well count
    drug_counts = treatment_wells.groupby('drug').agg({
        'well_id': 'count',
        'binary_dili': 'first',
        'dili_risk_score': 'first'
    }).rename(columns={'well_id': 'well_count'})
    
    # Get top 20 drugs
    top_drugs = drug_counts.nlargest(20, 'well_count')
    
    drug_data = []
    for drug, row in top_drugs.iterrows():
        dili_status = 'Unknown'
        if pd.notna(row['binary_dili']):
            dili_status = 'DILI Positive' if row['binary_dili'] == 1 else 'No DILI'
        
        drug_data.append({
            'drug': drug,
            'well_count': int(row['well_count']),
            'dili_status': dili_status,
            'dili_risk_score': float(row['dili_risk_score']) if pd.notna(row['dili_risk_score']) else None
        })
    
    # DILI risk score distribution
    valid_scores = treatment_wells['dili_risk_score'].dropna()
    score_distribution = {
        'scores': valid_scores.tolist(),
        'mean': float(valid_scores.mean()),
        'median': float(valid_scores.median()),
        'std': float(valid_scores.std())
    }
    
    # Overall DILI distribution
    dili_summary = {
        'total_drugs': treatment_wells['drug'].nunique(),
        'dili_positive': int((treatment_wells['binary_dili'] == 1).sum()),
        'dili_negative': int((treatment_wells['binary_dili'] == 0).sum()),
        'unknown_dili': int(treatment_wells['binary_dili'].isna().sum())
    }
    
    return {
        'top_drugs': drug_data,
        'score_distribution': score_distribution,
        'dili_summary': dili_summary
    }

def generate_concentration_analysis_data():
    """Generate data for concentration analysis visualization."""
    
    print("📊 Generating concentration analysis data...")
    
    filtered_wells, _ = get_filtered_dataset()
    treatment_wells = filtered_wells[~filtered_wells['is_control']]
    
    # Concentration distribution
    concentrations = treatment_wells['concentration'].values
    log_concentrations = np.log10(concentrations[concentrations > 0])
    
    concentration_dist = {
        'raw_concentrations': concentrations.tolist(),
        'log_concentrations': log_concentrations.tolist(),
        'min_conc': float(concentrations.min()),
        'max_conc': float(concentrations.max()),
        'median_conc': float(np.median(concentrations)),
        'unique_concentrations': len(np.unique(concentrations))
    }
    
    # Concentrations by DILI status
    conc_by_dili = []
    for dili_status in [0, 1]:
        subset = treatment_wells[treatment_wells['binary_dili'] == dili_status]
        if len(subset) > 0:
            conc_values = subset['concentration'].values
            log_values = np.log10(conc_values[conc_values > 0])
            conc_by_dili.append({
                'dili_status': 'DILI Positive' if dili_status == 1 else 'No DILI',
                'concentrations': conc_values.tolist(),
                'log_concentrations': log_values.tolist(),
                'count': len(subset)
            })
    
    # Concentration levels per drug
    drug_conc_levels = treatment_wells.groupby('drug')['concentration'].nunique().sort_values(ascending=False)
    
    conc_levels_data = []
    for drug, levels in drug_conc_levels.head(20).items():
        drug_wells = treatment_wells[treatment_wells['drug'] == drug]
        dili_status = 'Unknown'
        if pd.notna(drug_wells['binary_dili'].iloc[0]):
            dili_status = 'DILI Positive' if drug_wells['binary_dili'].iloc[0] == 1 else 'No DILI'
        
        conc_levels_data.append({
            'drug': drug,
            'concentration_levels': int(levels),
            'total_wells': len(drug_wells),
            'dili_status': dili_status
        })
    
    # Wells per concentration ranking
    conc_ranking = treatment_wells['concentration'].value_counts().sort_index()
    wells_per_conc = [
        {'concentration': float(conc), 'well_count': int(count)}
        for conc, count in conc_ranking.items()
    ]
    
    return {
        'concentration_distribution': concentration_dist,
        'concentrations_by_dili': conc_by_dili,
        'concentration_levels_per_drug': conc_levels_data,
        'wells_per_concentration': wells_per_conc
    }

def main():
    """Generate all dataset overview data and save to JSON files."""
    
    print("🔬 GENERATING DATASET OVERVIEW DATA")
    print("=" * 80)
    print("Creating data for HTML + D3.js visualizations")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path('scripts/visualization/data')
    output_dir.mkdir(exist_ok=True)
    
    # Generate all datasets
    datasets = {
        'dataset_composition': generate_dataset_composition_data(),
        'plate_summary': generate_plate_summary_data(),
        'dili_drug_analysis': generate_dili_drug_analysis_data(),
        'concentration_analysis': generate_concentration_analysis_data()
    }
    
    # Save each dataset
    for name, data in datasets.items():
        output_path = output_dir / f'{name}.json'
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {name} data to: {output_path}")
    
    # Print summary
    comp_data = datasets['dataset_composition']
    print(f"\n📊 DATASET OVERVIEW SUMMARY:")
    print(f"Total wells: {comp_data['summary']['total_wells']:,}")
    print(f"Treatment wells: {comp_data['well_composition']['Treatment']:,}")
    print(f"Control wells: {comp_data['well_composition']['Control']:,}")
    print(f"Unique drugs: {comp_data['summary']['unique_drugs']}")
    print(f"DILI positive rate: {comp_data['summary']['dili_positive_rate']:.1f}%")
    
    return datasets

if __name__ == "__main__":
    datasets = main()