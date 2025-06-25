#!/usr/bin/env python3
"""
Efficient Comprehensive DILI Analysis

PURPOSE:
    Fast, focused analysis with more drugs and better features while keeping processing time reasonable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import sys
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set DATABASE_URL
os.environ['DATABASE_URL'] = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"

from src.utils.data_loader import DataLoader

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "efficient_comprehensive_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("EFFICIENT COMPREHENSIVE DILI ANALYSIS")
print("=" * 80)

# ========== EXPANDED DILI DATASET ==========

def create_comprehensive_dili_dataset():
    """Create comprehensive DILI dataset - A is most dangerous, E is least dangerous"""
    
    dili_data = [
        # Highest DILI risk (A likelihood = most dangerous)
        {'drug': 'Amiodarone', 'binary_dili': True, 'likelihood': 'A', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Busulfan', 'binary_dili': True, 'likelihood': 'A', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Mercaptopurine', 'binary_dili': True, 'likelihood': 'A', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Floxuridine', 'binary_dili': True, 'likelihood': 'A', 'severity': 7, 'category': 'Most-DILI-Concern'},
        {'drug': 'Oxaliplatin', 'binary_dili': True, 'likelihood': 'A', 'severity': 4, 'category': 'High-DILI-Concern'},
        {'drug': 'Fluorouracil', 'binary_dili': True, 'likelihood': 'A', 'severity': 4, 'category': 'High-DILI-Concern'},
        {'drug': 'Methotrexate', 'binary_dili': True, 'likelihood': 'A', 'severity': 3, 'category': 'High-DILI-Concern'},
        {'drug': 'Thioguanine', 'binary_dili': True, 'likelihood': 'A', 'severity': 2, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Ambrisentan', 'binary_dili': False, 'likelihood': 'A', 'severity': 1, 'category': 'Low-DILI-Concern'},
        {'drug': 'Buspirone', 'binary_dili': False, 'likelihood': 'A', 'severity': 1, 'category': 'Low-DILI-Concern'},
        
        # High DILI risk (B likelihood)
        {'drug': 'Imatinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Lapatinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Sorafenib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Sunitinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Erlotinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Erlotinib hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Gefitinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 4, 'category': 'High-DILI-Concern'},
        {'drug': 'Tamoxifen', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Tamoxifen citrate', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Carmustine', 'binary_dili': True, 'likelihood': 'B', 'severity': 7, 'category': 'High-DILI-Concern'},
        {'drug': 'Regorafenib', 'binary_dili': True, 'likelihood': 'B', 'severity': 6, 'category': 'High-DILI-Concern'},
        {'drug': 'Dacarbazine', 'binary_dili': True, 'likelihood': 'B', 'severity': 6, 'category': 'High-DILI-Concern'},
        {'drug': 'Ceritinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Cyclophosphamide', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Melphalan', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Melphalan hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Temozolomide', 'binary_dili': True, 'likelihood': 'B', 'severity': 4, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Mitomycin', 'binary_dili': True, 'likelihood': 'B', 'severity': 4, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Doxorubicin', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Doxorubicin hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Irinotecan', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Irinotecan hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        
        # Moderate DILI risk (C likelihood)
        {'drug': 'Pazopanib', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Pazopanib hydrochloride', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Gemcitabine', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Gemcitabine hydrochloride', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Hydroxyurea', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Vincristine', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Vincristine sulfate', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Bortezomib', 'binary_dili': True, 'likelihood': 'C', 'severity': 7, 'category': 'High-DILI-Concern'},
        {'drug': 'Afatinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Cabozantinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Crizotinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 5, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Alectinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 4, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Binimetinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 4, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Ribociclib', 'binary_dili': True, 'likelihood': 'C', 'severity': 4, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Anastrozole', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Ruxolitinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Palbociclib', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Cisplatin', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Etoposide', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Docetaxel', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Moderate-DILI-Concern'},
        
        # Lower DILI risk (D likelihood)
        {'drug': 'Lenvatinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Nilotinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Dasatinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Ibrutinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Paclitaxel', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Carboplatin', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Chlorambucil', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Mitoxantrone', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Bleomycin sulfate', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Thiotepa', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Lomustine', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Procarbazine', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Procarbazine hydrochloride', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Less-DILI-Concern'},
        {'drug': 'Letrozole', 'binary_dili': True, 'likelihood': 'D', 'severity': 2, 'category': 'Less-DILI-Concern'},
        {'drug': 'Clofarabine', 'binary_dili': True, 'likelihood': 'D', 'severity': 8, 'category': 'Less-DILI-Concern'},
        
        # Lowest DILI risk (E likelihood = least dangerous)
        {'drug': 'Azacitidine', 'binary_dili': True, 'likelihood': 'E', 'severity': 8, 'category': 'Low-DILI-Concern'},
        {'drug': 'Fulvestrant', 'binary_dili': True, 'likelihood': 'E', 'severity': 7, 'category': 'Low-DILI-Concern'},
        {'drug': 'Everolimus', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Low-DILI-Concern'},
        {'drug': 'Ponatinib', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Low-DILI-Concern'},
        {'drug': 'Vemurafenib', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Low-DILI-Concern'},
        {'drug': 'Decitabine', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Low-DILI-Concern'},
        {'drug': 'Alpelisib', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Zoledronic acid', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Trametinib', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Cladribine', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Epirubicin', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Epirubicin hydrochloride', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Idarubicin', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Idarubicin hydrochloride', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Capecitabine', 'binary_dili': True, 'likelihood': 'E', 'severity': 2, 'category': 'Low-DILI-Concern'},
        {'drug': 'Axitinib', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Dabrafenib', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Dabrafenib mesylate', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Osimertinib', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Daunorubicin', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Daunorubicin hydrochloride', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Topotecan', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Topotecan hydrochloride', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Fludarabine', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Fludarabine phosphate', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
    ]
    
    return pd.DataFrame(dili_data)

# ========== FOCUSED FEATURE EXTRACTION ==========

def extract_focused_features(well_data, well_id, drug, concentration):
    """Extract focused, high-value features efficiently"""
    
    if len(well_data) < 100:
        return None
    
    # Ensure oxygen column exists
    if 'o2' in well_data.columns and 'oxygen' not in well_data.columns:
        well_data = well_data.rename(columns={'o2': 'oxygen'})
    
    oxygen = well_data['oxygen'].values
    times = well_data['elapsed_hours'].values
    
    if len(oxygen) < 100:
        return None
    
    features = {
        'well_id': well_id,
        'drug': drug,
        'concentration': concentration,
        'duration_hours': times[-1] - times[0]
    }
    
    # ========== 1. CORE STATISTICAL FEATURES ==========
    features.update({
        'oxygen_mean': np.mean(oxygen),
        'oxygen_std': np.std(oxygen),
        'oxygen_min': np.min(oxygen),
        'oxygen_max': np.max(oxygen),
        'oxygen_range': np.max(oxygen) - np.min(oxygen),
        'oxygen_cv': np.std(oxygen) / np.mean(oxygen) if np.mean(oxygen) != 0 else 0,
        'oxygen_skew': pd.Series(oxygen).skew(),
        'oxygen_kurt': pd.Series(oxygen).kurtosis(),
    })
    
    # ========== 2. PHASE-BASED FEATURES ==========
    
    phases = {
        'baseline': (0, 48),
        'immediate': (48, 96),
        'early': (96, 168),
        'late': (168, 288)
    }
    
    for phase_name, (start_h, end_h) in phases.items():
        phase_mask = (times >= start_h) & (times < end_h)
        if np.sum(phase_mask) >= 10:
            phase_values = oxygen[phase_mask]
            features.update({
                f'{phase_name}_mean': np.mean(phase_values),
                f'{phase_name}_std': np.std(phase_values),
                f'{phase_name}_slope': np.polyfit(times[phase_mask] - times[phase_mask][0], 
                                               phase_values, 1)[0] if len(phase_values) > 2 else 0
            })
    
    # ========== 3. DYNAMIC FEATURES ==========
    
    # Rolling statistics
    for window in [10, 50]:
        if len(oxygen) > window:
            rolling_mean = pd.Series(oxygen).rolling(window=window).mean()
            rolling_std = pd.Series(oxygen).rolling(window=window).std()
            
            features.update({
                f'rolling_{window}_mean_std': rolling_mean.std(),
                f'rolling_{window}_std_mean': rolling_std.mean(),
            })
    
    # ========== 4. EVENT FEATURES ==========
    
    # Simple event detection
    rolling_var = pd.Series(oxygen).rolling(window=10, center=True).var().fillna(0)
    baseline_var = np.median(rolling_var[:min(100, len(rolling_var))])
    
    if baseline_var > 0:
        events = times[rolling_var > 3 * baseline_var]
        features.update({
            'n_events': len(events),
            'event_rate': len(events) / (times[-1] - times[0]) * 24 if times[-1] > times[0] else 0
        })
    
    # ========== 5. TREND FEATURES ==========
    
    # Overall trend
    try:
        slope, intercept = np.polyfit(times, oxygen, 1)
        features.update({
            'overall_slope': slope,
            'trend_strength': abs(slope) * (times[-1] - times[0])
        })
    except:
        features.update({
            'overall_slope': 0,
            'trend_strength': 0
        })
    
    return features

# ========== MAIN PROCESSING ==========

def main():
    print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load DILI data
    dili_df = create_comprehensive_dili_dataset()
    print(f"\n📊 Loaded {len(dili_df)} drugs with DILI annotations")
    print(f"   DILI distribution:")
    print(f"     Binary positive: {dili_df['binary_dili'].sum()}")
    print(f"     Binary negative: {(dili_df['binary_dili'] == False).sum()}")
    print(f"     Likelihood A (most dangerous): {(dili_df['likelihood'] == 'A').sum()}")
    print(f"     Likelihood E (least dangerous): {(dili_df['likelihood'] == 'E').sum()}")
    
    # Get list of drugs we want to process
    target_drugs = dili_df['drug'].tolist()
    
    # Initialize data loader
    loader = DataLoader()
    wells = loader.load_well_metadata()
    
    # Find drugs that have experimental data
    available_drugs = set(wells['drug']) if 'drug' in wells.columns else set()
    matched_drugs = []
    
    for target_drug in target_drugs:
        # Try exact match first
        if target_drug in available_drugs:
            matched_drugs.append(target_drug)
        else:
            # Try partial match (remove suffixes)
            target_clean = target_drug.split(' ')[0].split('.')[0]
            for avail_drug in available_drugs:
                avail_clean = avail_drug.split(' ')[0].split('.')[0]
                if target_clean.lower() == avail_clean.lower():
                    matched_drugs.append(avail_drug)
                    break
    
    matched_drugs = list(set(matched_drugs))  # Remove duplicates
    print(f"   Found experimental data for {len(matched_drugs)} drugs")
    
    # Process drugs efficiently
    all_features = []
    
    print(f"\n🔄 Processing {len(matched_drugs)} drugs...")
    
    # Get wells for all matched drugs at once
    all_wells = wells[wells['drug'].isin(matched_drugs)]
    
    # Sample wells per drug (max 20 per drug for speed)
    sampled_wells = []
    for drug in matched_drugs:
        drug_wells = all_wells[all_wells['drug'] == drug]
        if len(drug_wells) > 0:
            sample_size = min(20, len(drug_wells))
            sampled_wells.append(drug_wells.sample(n=sample_size, random_state=42))
    
    if not sampled_wells:
        print("❌ No wells found!")
        return None
    
    sampled_wells_df = pd.concat(sampled_wells, ignore_index=True)
    print(f"   Processing {len(sampled_wells_df)} wells from {len(matched_drugs)} drugs")
    
    # Process by plate (limit to first 10 plates for efficiency)
    unique_plates = sampled_wells_df['plate_id'].unique()[:10]
    
    for i, plate_id in enumerate(unique_plates):
        print(f"   Plate {i+1}/{len(unique_plates)}: {plate_id}")
        
        try:
            # Load plate data
            plate_data = loader.load_oxygen_data(plate_ids=[str(plate_id)])
            
            if 'o2' in plate_data.columns and 'oxygen' not in plate_data.columns:
                plate_data = plate_data.rename(columns={'o2': 'oxygen'})
            
            # Get wells for this plate
            plate_wells = sampled_wells_df[sampled_wells_df['plate_id'] == plate_id]
            
            # Process each well
            for _, well_info in plate_wells.iterrows():
                well_id = well_info['well_id']
                well_data = plate_data[plate_data['well_id'] == well_id]
                
                if len(well_data) < 100:
                    continue
                
                well_data = well_data.sort_values('elapsed_hours')
                
                # Extract focused features
                features = extract_focused_features(
                    well_data,
                    well_id,
                    well_info['drug'],
                    well_info['concentration']
                )
                
                if features is not None:
                    all_features.append(features)
            
        except Exception as e:
            print(f"      Error processing plate {plate_id}: {e}")
            continue
    
    print(f"\n✓ Extracted features for {len(all_features)} wells")
    
    if len(all_features) == 0:
        print("❌ No features extracted!")
        return None
    
    # Convert to DataFrame
    well_features_df = pd.DataFrame(all_features)
    
    # ========== DRUG-LEVEL AGGREGATION ==========
    
    print("\n🔄 Aggregating features at drug level...")
    
    drug_features_list = []
    
    for drug in matched_drugs:
        drug_data = well_features_df[well_features_df['drug'] == drug]
        
        if len(drug_data) == 0:
            continue
        
        drug_features = {
            'drug': drug,
            'n_wells': len(drug_data),
        }
        
        # Aggregate numeric features
        numeric_cols = [col for col in drug_data.columns 
                       if col not in ['well_id', 'drug', 'concentration'] and 
                       drug_data[col].dtype in ['float64', 'int64']]
        
        for col in numeric_cols:
            values = drug_data[col].dropna()
            if len(values) > 0:
                drug_features[f'{col}_mean'] = values.mean()
                drug_features[f'{col}_std'] = values.std()
                drug_features[f'{col}_median'] = values.median()
        
        drug_features_list.append(drug_features)
    
    drug_features_df = pd.DataFrame(drug_features_list)
    print(f"   Created features for {len(drug_features_df)} drugs")
    
    # ========== MERGE WITH DILI DATA ==========
    
    print("\n🔗 Merging with DILI annotations...")
    
    # Create mapping dictionary for fuzzy matching
    dili_mapping = {}
    for _, dili_row in dili_df.iterrows():
        dili_drug = dili_row['drug']
        dili_mapping[dili_drug] = dili_row
        
        # Add simplified versions
        simplified = dili_drug.split(' ')[0].split('.')[0]
        if simplified not in dili_mapping:
            dili_mapping[simplified] = dili_row
    
    # Match drugs
    final_data = []
    for _, drug_row in drug_features_df.iterrows():
        exp_drug = drug_row['drug']
        
        # Try exact match first
        if exp_drug in dili_mapping:
            dili_info = dili_mapping[exp_drug]
        else:
            # Try simplified match
            exp_simplified = exp_drug.split(' ')[0].split('.')[0]
            if exp_simplified in dili_mapping:
                dili_info = dili_mapping[exp_simplified]
            else:
                continue
        
        # Combine drug features with DILI info
        combined_row = drug_row.copy()
        combined_row['dili_binary'] = dili_info['binary_dili']
        combined_row['dili_likelihood'] = dili_info['likelihood']
        combined_row['dili_severity'] = dili_info['severity']
        combined_row['dili_category'] = dili_info['category']
        
        final_data.append(combined_row)
    
    if not final_data:
        print("❌ No drugs matched between features and DILI data!")
        return None
    
    final_df = pd.DataFrame(final_data)
    print(f"   Final dataset: {len(final_df)} drugs with both features and DILI data")
    
    # Create numeric likelihood encoding (A=6 most dangerous, E=1 least dangerous)
    likelihood_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'E*': 1}
    final_df['dili_likelihood_numeric'] = final_df['dili_likelihood'].map(likelihood_mapping)
    
    # ========== QUICK CORRELATION ANALYSIS ==========
    
    print("\n📊 Quick correlation analysis...")
    
    # Get feature columns
    exclude_cols = ['drug', 'n_wells', 'dili_binary', 'dili_likelihood', 'dili_severity', 'dili_category', 'dili_likelihood_numeric']
    feature_cols = [col for col in final_df.columns if col not in exclude_cols]
    
    # Calculate correlations
    correlations = []
    for feat in feature_cols:
        valid_data = final_df[[feat, 'dili_likelihood_numeric']].dropna()
        if len(valid_data) >= 5:
            r, p = pearsonr(valid_data[feat], valid_data['dili_likelihood_numeric'])
            correlations.append({
                'feature': feat,
                'r': r,
                'p': p,
                'abs_r': abs(r)
            })
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('abs_r', ascending=False)
        
        print(f"\nTop 10 correlations with DILI likelihood:")
        for _, row in corr_df.head(10).iterrows():
            print(f"   {row['feature']}: r={row['r']:.3f} (p={row['p']:.3f})")
    
    # ========== SAVE RESULTS ==========
    
    print("\n💾 Saving results...")
    
    # Save final merged dataset
    final_df.to_parquet(results_dir / 'efficient_comprehensive_dili_dataset.parquet', index=False)
    
    # Save correlation results
    if correlations:
        corr_df.to_csv(results_dir / 'efficient_comprehensive_correlations.csv', index=False)
    
    # Summary statistics
    summary = {
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_wells_processed': len(well_features_df),
        'n_drugs_with_features': len(drug_features_df),
        'n_drugs_with_dili': len(final_df),
        'n_features_extracted': len(feature_cols),
        'dili_distribution': {
            'binary_positive': int(final_df['dili_binary'].sum()),
            'binary_negative': int((final_df['dili_binary'] == False).sum()),
            'likelihood_A_most_dangerous': int((final_df['dili_likelihood'] == 'A').sum()),
            'likelihood_E_least_dangerous': int((final_df['dili_likelihood'] == 'E').sum()),
            'severity_distribution': final_df['dili_severity'].value_counts().to_dict(),
            'likelihood_distribution': final_df['dili_likelihood'].value_counts().to_dict(),
        },
        'top_correlations': corr_df.head(10).to_dict('records') if correlations else [],
        'drugs_analyzed': sorted(final_df['drug'].tolist())
    }
    
    with open(results_dir / 'efficient_comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Efficient comprehensive analysis complete!")
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Wells processed: {summary['n_wells_processed']:,}")
    print(f"   Drugs analyzed: {summary['n_drugs_with_dili']}")
    print(f"   Features extracted: {summary['n_features_extracted']}")
    print(f"   DILI positive: {summary['dili_distribution']['binary_positive']}")
    print(f"   DILI negative: {summary['dili_distribution']['binary_negative']}")
    print(f"   Likelihood A (most dangerous): {summary['dili_distribution']['likelihood_A_most_dangerous']}")
    print(f"   Likelihood E (least dangerous): {summary['dili_distribution']['likelihood_E_least_dangerous']}")
    
    if correlations:
        best_corr = corr_df.iloc[0]
        print(f"\n🎯 Best predictor: {best_corr['feature']} (r={best_corr['r']:.3f}, p={best_corr['p']:.3f})")
    
    return final_df

if __name__ == "__main__":
    result_df = main()