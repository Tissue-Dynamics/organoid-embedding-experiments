#!/usr/bin/env python3
"""
Enhanced Comprehensive DILI Analysis

PURPOSE:
    Expanded analysis using the comprehensive DILI dataset with more sophisticated features.
    Builds on previous event-normalized analysis with additional drugs and feature types.
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
fig_dir = project_root / "results" / "figures" / "enhanced_comprehensive_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("ENHANCED COMPREHENSIVE DILI ANALYSIS")
print("=" * 80)

# ========== EXPANDED DILI DATASET ==========

def create_comprehensive_dili_dataset():
    """Create comprehensive DILI dataset with all available annotations"""
    
    dili_data = [
        # Highest DILI severity (8) - Most concerning
        {'drug': 'Amiodarone', 'binary_dili': True, 'likelihood': 'A', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Busulfan', 'binary_dili': True, 'likelihood': 'A', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Imatinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Lapatinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Pazopanib', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Pazopanib hydrochloride', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Sorafenib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Sunitinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Erlotinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Erlotinib hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Gemcitabine', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Gemcitabine hydrochloride', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Mercaptopurine', 'binary_dili': True, 'likelihood': 'A', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Hydroxyurea', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Tamoxifen', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Tamoxifen citrate', 'binary_dili': True, 'likelihood': 'B', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Vincristine', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Vincristine sulfate', 'binary_dili': True, 'likelihood': 'C', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Azacitidine', 'binary_dili': True, 'likelihood': 'E', 'severity': 8, 'category': 'Most-DILI-Concern'},
        {'drug': 'Clofarabine', 'binary_dili': True, 'likelihood': 'D', 'severity': 8, 'category': 'Most-DILI-Concern'},
        
        # High DILI severity (7) 
        {'drug': 'Bortezomib', 'binary_dili': True, 'likelihood': 'C', 'severity': 7, 'category': 'High-DILI-Concern'},
        {'drug': 'Carmustine', 'binary_dili': True, 'likelihood': 'B', 'severity': 7, 'category': 'High-DILI-Concern'},
        {'drug': 'Fulvestrant', 'binary_dili': True, 'likelihood': 'E', 'severity': 7, 'category': 'High-DILI-Concern'},
        {'drug': 'Floxuridine', 'binary_dili': True, 'likelihood': 'A', 'severity': 7, 'category': 'High-DILI-Concern'},
        
        # Moderate DILI severity (6)
        {'drug': 'Regorafenib', 'binary_dili': True, 'likelihood': 'B', 'severity': 6, 'category': 'Moderate-DILI-Concern'},
        {'drug': 'Dacarbazine', 'binary_dili': True, 'likelihood': 'B', 'severity': 6, 'category': 'Moderate-DILI-Concern'},
        
        # Moderate-low DILI severity (5)
        {'drug': 'Afatinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 5, 'category': 'DILI-Concern'},
        {'drug': 'Cabozantinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 5, 'category': 'DILI-Concern'},
        {'drug': 'Ceritinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'DILI-Concern'},
        {'drug': 'Crizotinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 5, 'category': 'DILI-Concern'},
        {'drug': 'Cyclophosphamide', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'DILI-Concern'},
        {'drug': 'Melphalan', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'DILI-Concern'},
        {'drug': 'Melphalan hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 5, 'category': 'DILI-Concern'},
        
        # Lower DILI severity (4)
        {'drug': 'Gefitinib', 'binary_dili': True, 'likelihood': 'B', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Alectinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Binimetinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Dasatinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Everolimus', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Ibrutinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Ponatinib', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Lenvatinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Nilotinib', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Vemurafenib', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Oxaliplatin', 'binary_dili': True, 'likelihood': 'A', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Fluorouracil', 'binary_dili': True, 'likelihood': 'A', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Temozolomide', 'binary_dili': True, 'likelihood': 'B', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Paclitaxel', 'binary_dili': True, 'likelihood': 'D', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Mitomycin', 'binary_dili': True, 'likelihood': 'B', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Decitabine', 'binary_dili': True, 'likelihood': 'E', 'severity': 4, 'category': 'Less-DILI-Concern'},
        {'drug': 'Ribociclib', 'binary_dili': True, 'likelihood': 'C', 'severity': 4, 'category': 'Less-DILI-Concern'},
        
        # Low DILI severity (3)
        {'drug': 'Anastrozole', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Ruxolitinib', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Palbociclib', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Alpelisib', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Zoledronic acid', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Trametinib', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Carboplatin', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Cisplatin', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Doxorubicin', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Doxorubicin hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Etoposide', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Irinotecan', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Irinotecan hydrochloride', 'binary_dili': True, 'likelihood': 'B', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Docetaxel', 'binary_dili': True, 'likelihood': 'C', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Chlorambucil', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Mitoxantrone', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Cladribine', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Bleomycin sulfate', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Epirubicin', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Epirubicin hydrochloride', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Idarubicin', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Idarubicin hydrochloride', 'binary_dili': True, 'likelihood': 'E', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Thiotepa', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Lomustine', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Procarbazine', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Procarbazine hydrochloride', 'binary_dili': True, 'likelihood': 'D', 'severity': 3, 'category': 'Low-DILI-Concern'},
        {'drug': 'Methotrexate', 'binary_dili': True, 'likelihood': 'A', 'severity': 3, 'category': 'Low-DILI-Concern'},
        
        # Very low DILI severity (2)
        {'drug': 'Axitinib', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Dabrafenib', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Dabrafenib mesylate', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Osimertinib', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Letrozole', 'binary_dili': True, 'likelihood': 'D', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Capecitabine', 'binary_dili': True, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Thioguanine', 'binary_dili': True, 'likelihood': 'A', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Daunorubicin', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Daunorubicin hydrochloride', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Topotecan', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Topotecan hydrochloride', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Fludarabine', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        {'drug': 'Fludarabine phosphate', 'binary_dili': False, 'likelihood': 'E', 'severity': 2, 'category': 'No-DILI-Concern'},
        
        # Minimal DILI severity (1)
        {'drug': 'Ambrisentan', 'binary_dili': False, 'likelihood': 'A', 'severity': 1, 'category': 'No-DILI-Concern'},
        {'drug': 'Buspirone', 'binary_dili': False, 'likelihood': 'A', 'severity': 1, 'category': 'No-DILI-Concern'},
    ]
    
    return pd.DataFrame(dili_data)

# ========== ENHANCED FEATURE EXTRACTION ==========

def extract_enhanced_features(well_data, well_id, drug, concentration):
    """Extract enhanced features from well time series"""
    
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
        'n_timepoints': len(oxygen),
        'duration_hours': times[-1] - times[0]
    }
    
    # ========== 1. ENHANCED STATISTICAL FEATURES ==========
    
    features.update({
        # Basic statistics
        'oxygen_mean': np.mean(oxygen),
        'oxygen_median': np.median(oxygen),
        'oxygen_std': np.std(oxygen),
        'oxygen_var': np.var(oxygen),
        'oxygen_min': np.min(oxygen),
        'oxygen_max': np.max(oxygen),
        'oxygen_range': np.max(oxygen) - np.min(oxygen),
        'oxygen_q25': np.percentile(oxygen, 25),
        'oxygen_q75': np.percentile(oxygen, 75),
        'oxygen_iqr': np.percentile(oxygen, 75) - np.percentile(oxygen, 25),
        'oxygen_skew': pd.Series(oxygen).skew(),
        'oxygen_kurt': pd.Series(oxygen).kurtosis(),
        'oxygen_cv': np.std(oxygen) / np.mean(oxygen) if np.mean(oxygen) != 0 else 0,
        
        # Additional robust statistics
        'oxygen_mad': np.median(np.abs(oxygen - np.median(oxygen))),  # Median absolute deviation
        'oxygen_rms': np.sqrt(np.mean(oxygen**2)),  # Root mean square
        'oxygen_trimmed_mean': pd.Series(oxygen).quantile([0.1, 0.9]).mean(),  # 10-90% trimmed mean
    })
    
    # ========== 2. TEMPORAL DYNAMICS FEATURES ==========
    
    # First and second derivatives
    dt = np.diff(times)
    if len(dt) > 0 and np.all(dt > 0):
        first_deriv = np.diff(oxygen) / dt
        features.update({
            'first_deriv_mean': np.mean(first_deriv),
            'first_deriv_std': np.std(first_deriv),
            'first_deriv_max': np.max(first_deriv),
            'first_deriv_min': np.min(first_deriv),
        })
        
        if len(first_deriv) > 1:
            second_deriv = np.diff(first_deriv) / dt[1:]
            features.update({
                'second_deriv_mean': np.mean(second_deriv),
                'second_deriv_std': np.std(second_deriv),
                'acceleration_max': np.max(second_deriv),
                'deceleration_max': np.min(second_deriv),
            })
    
    # ========== 3. PHASE-BASED ENHANCED FEATURES ==========
    
    # Define more granular phases
    phases = {
        'baseline': (0, 48),
        'immediate_response': (48, 72),
        'early_treatment': (72, 120),
        'mid_treatment': (120, 192),
        'late_treatment': (192, 288),
        'recovery': (288, 400)
    }
    
    phase_features = {}
    for phase_name, (start_h, end_h) in phases.items():
        phase_mask = (times >= start_h) & (times < end_h)
        if np.sum(phase_mask) >= 10:
            phase_values = oxygen[phase_mask]
            phase_times = times[phase_mask]
            
            # Basic statistics for phase
            phase_features.update({
                f'{phase_name}_mean': np.mean(phase_values),
                f'{phase_name}_std': np.std(phase_values),
                f'{phase_name}_min': np.min(phase_values),
                f'{phase_name}_max': np.max(phase_values),
                f'{phase_name}_range': np.max(phase_values) - np.min(phase_values),
                f'{phase_name}_cv': np.std(phase_values) / np.mean(phase_values) if np.mean(phase_values) != 0 else 0,
            })
            
            # Trend in phase
            if len(phase_values) > 2:
                try:
                    slope = np.polyfit(phase_times - phase_times[0], phase_values, 1)[0]
                    phase_features[f'{phase_name}_slope'] = slope
                except:
                    phase_features[f'{phase_name}_slope'] = 0
    
    features.update(phase_features)
    
    # ========== 4. IMPROVED EVENT DETECTION ==========
    
    # Multiple event detection approaches
    window_sizes = [5, 10, 20]
    event_metrics = {}
    
    for window in window_sizes:
        if len(oxygen) > window:
            # Variance-based detection
            rolling_var = pd.Series(oxygen).rolling(window=window, center=True).var().fillna(0)
            baseline_var = np.median(rolling_var[:min(100, len(rolling_var))])
            
            if baseline_var > 0:
                var_threshold = 3 * baseline_var
                var_events = times[rolling_var > var_threshold]
                
                # Standard deviation detection
                rolling_std = pd.Series(oxygen).rolling(window=window, center=True).std().fillna(0)
                baseline_std = np.median(rolling_std[:min(100, len(rolling_std))])
                
                if baseline_std > 0:
                    std_threshold = 2 * baseline_std
                    std_events = times[rolling_std > std_threshold]
                    
                    event_metrics.update({
                        f'var_events_w{window}': len(var_events),
                        f'std_events_w{window}': len(std_events),
                        f'event_density_w{window}': len(var_events) / (times[-1] - times[0]) * 24 if times[-1] > times[0] else 0
                    })
    
    features.update(event_metrics)
    
    # ========== 5. FREQUENCY DOMAIN FEATURES ==========
    
    # Simple frequency analysis using FFT
    try:
        if len(oxygen) >= 64:  # Minimum for meaningful FFT
            # Remove trend before FFT
            detrended = oxygen - np.polyval(np.polyfit(times, oxygen, 1), times)
            
            # Compute FFT
            fft_values = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended), d=np.mean(np.diff(times)))
            
            # Power spectral density
            psd = np.abs(fft_values)**2
            
            # Find dominant frequency
            positive_freqs = freqs[:len(freqs)//2]
            positive_psd = psd[:len(psd)//2]
            
            if len(positive_psd) > 0:
                dominant_freq_idx = np.argmax(positive_psd)
                features.update({
                    'dominant_frequency': positive_freqs[dominant_freq_idx] if dominant_freq_idx < len(positive_freqs) else 0,
                    'spectral_power_total': np.sum(positive_psd),
                    'spectral_power_low': np.sum(positive_psd[:len(positive_psd)//4]),  # Low frequency power
                    'spectral_power_high': np.sum(positive_psd[3*len(positive_psd)//4:]),  # High frequency power
                    'spectral_centroid': np.sum(positive_freqs * positive_psd) / np.sum(positive_psd) if np.sum(positive_psd) > 0 else 0
                })
    except:
        features.update({
            'dominant_frequency': 0,
            'spectral_power_total': 0,
            'spectral_power_low': 0,
            'spectral_power_high': 0,
            'spectral_centroid': 0
        })
    
    # ========== 6. PATTERN RECOGNITION FEATURES ==========
    
    # Autocorrelation at different lags
    lags = [1, 6, 12, 24, 48]  # 1, 6, 12, 24, 48 hour lags
    
    for lag in lags:
        if len(oxygen) > lag:
            # Find closest time indices for lag
            lag_indices = []
            for i in range(len(times)):
                target_time = times[i] + lag
                closest_idx = np.argmin(np.abs(times - target_time))
                if closest_idx < len(times) and np.abs(times[closest_idx] - target_time) < lag/2:
                    lag_indices.append((i, closest_idx))
            
            if len(lag_indices) > 10:
                original_vals = [oxygen[i] for i, j in lag_indices]
                lagged_vals = [oxygen[j] for i, j in lag_indices]
                
                if len(original_vals) > 1:
                    autocorr = np.corrcoef(original_vals, lagged_vals)[0, 1]
                    features[f'autocorr_lag_{lag}h'] = autocorr if not np.isnan(autocorr) else 0
    
    # ========== 7. STABILITY AND PERSISTENCE METRICS ==========
    
    # Calculate consecutive periods above/below mean
    mean_oxygen = np.mean(oxygen)
    above_mean = oxygen > mean_oxygen
    
    # Find runs of consecutive values above/below mean
    runs_above = []
    runs_below = []
    current_run = 0
    current_state = above_mean[0]
    
    for i in range(1, len(above_mean)):
        if above_mean[i] == current_state:
            current_run += 1
        else:
            if current_state:
                runs_above.append(current_run)
            else:
                runs_below.append(current_run)
            current_run = 1
            current_state = above_mean[i]
    
    # Add final run
    if current_state:
        runs_above.append(current_run)
    else:
        runs_below.append(current_run)
    
    features.update({
        'runs_above_mean_count': len(runs_above),
        'runs_below_mean_count': len(runs_below),
        'runs_above_mean_avg': np.mean(runs_above) if runs_above else 0,
        'runs_below_mean_avg': np.mean(runs_below) if runs_below else 0,
        'runs_above_mean_max': np.max(runs_above) if runs_above else 0,
        'runs_below_mean_max': np.max(runs_below) if runs_below else 0,
        'stability_index': len(runs_above) + len(runs_below)  # More runs = less stable
    })
    
    return features

# ========== MAIN PROCESSING ==========

def main():
    print(f"\n⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load DILI data
    dili_df = create_comprehensive_dili_dataset()
    print(f"\n📊 Loaded {len(dili_df)} drugs with DILI annotations")
    
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
    
    # Process drugs in batches
    all_features = []
    batch_size = 8
    
    for i in range(0, len(matched_drugs), batch_size):
        batch_drugs = matched_drugs[i:i+batch_size]
        print(f"\n🔄 Batch {i//batch_size + 1}/{(len(matched_drugs) + batch_size - 1)//batch_size}")
        print(f"   Drugs: {', '.join(batch_drugs[:3])}{'...' if len(batch_drugs) > 3 else ''}")
        
        # Get wells for these drugs
        batch_wells = wells[wells['drug'].isin(batch_drugs)]
        
        if len(batch_wells) == 0:
            continue
        
        # Sample wells per drug (max 25 per drug)
        sampled_wells = []
        for drug in batch_drugs:
            drug_wells = batch_wells[batch_wells['drug'] == drug]
            if len(drug_wells) > 0:
                sample_size = min(25, len(drug_wells))
                sampled_wells.append(drug_wells.sample(n=sample_size, random_state=42))
        
        if not sampled_wells:
            continue
        
        sampled_wells_df = pd.concat(sampled_wells, ignore_index=True)
        
        # Process by plate
        unique_plates = sampled_wells_df['plate_id'].unique()
        
        for plate_id in unique_plates:
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
                    
                    # Extract enhanced features
                    features = extract_enhanced_features(
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
        
        print(f"      Total features extracted: {len(all_features)}")
    
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
            'n_concentrations': drug_data['concentration'].nunique()
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
                drug_features[f'{col}_q25'] = values.quantile(0.25)
                drug_features[f'{col}_q75'] = values.quantile(0.75)
        
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
    
    # Create numeric likelihood encoding
    likelihood_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'E*': 6}
    final_df['dili_likelihood_numeric'] = final_df['dili_likelihood'].map(likelihood_mapping)
    
    # ========== SAVE RESULTS ==========
    
    print("\n💾 Saving comprehensive results...")
    
    # Save well-level features
    well_features_df.to_parquet(results_dir / 'enhanced_comprehensive_well_features.parquet', index=False)
    
    # Save drug-level features
    drug_features_df.to_parquet(results_dir / 'enhanced_comprehensive_drug_features.parquet', index=False)
    
    # Save final merged dataset
    final_df.to_parquet(results_dir / 'enhanced_comprehensive_dili_dataset.parquet', index=False)
    
    # Summary statistics
    summary = {
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_wells_processed': len(well_features_df),
        'n_drugs_with_features': len(drug_features_df),
        'n_drugs_with_dili': len(final_df),
        'n_features_per_well': len([c for c in well_features_df.columns 
                                   if c not in ['well_id', 'drug', 'concentration']]),
        'n_features_per_drug': len([c for c in drug_features_df.columns 
                                   if c not in ['drug', 'n_wells', 'n_concentrations']]),
        'dili_distribution': {
            'binary_positive': int(final_df['dili_binary'].sum()),
            'binary_negative': int((final_df['dili_binary'] == False).sum()),
            'severity_distribution': final_df['dili_severity'].value_counts().to_dict(),
            'likelihood_distribution': final_df['dili_likelihood'].value_counts().to_dict(),
            'category_distribution': final_df['dili_category'].value_counts().to_dict()
        },
        'drugs_processed': sorted(final_df['drug'].tolist())
    }
    
    with open(results_dir / 'enhanced_comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Enhanced comprehensive analysis complete!")
    print(f"\n📊 SUMMARY:")
    print(f"   Wells processed: {summary['n_wells_processed']:,}")
    print(f"   Drugs with features: {summary['n_drugs_with_features']}")
    print(f"   Drugs with DILI data: {summary['n_drugs_with_dili']}")
    print(f"   Features per well: {summary['n_features_per_well']}")
    print(f"   Features per drug: {summary['n_features_per_drug']}")
    print(f"   DILI positive: {summary['dili_distribution']['binary_positive']}")
    print(f"   DILI negative: {summary['dili_distribution']['binary_negative']}")
    
    return final_df

if __name__ == "__main__":
    result_df = main()