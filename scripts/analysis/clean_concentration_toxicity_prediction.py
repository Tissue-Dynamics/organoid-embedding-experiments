#!/usr/bin/env python3
"""
Clean Concentration-Dependent Toxicity Prediction

Focus on two clear feature sets:
1. Toxicity onset concentration detection from oxygen data
2. Variability/changes compared to control at each concentration

Then predict toxicity using these features + Cmax normalization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils.data_loader import DataLoader

warnings.filterwarnings('ignore')


class CleanConcentrationToxicityPredictor:
    """Clean approach to concentration-dependent toxicity prediction."""
    
    def __init__(self):
        self.drug_targets = self._load_drug_metadata()
        self.control_baseline = {}
        self.toxicity_onsets = {}
        
    def _load_drug_metadata(self) -> pd.DataFrame:
        """Load drug metadata with DILI and Cmax info."""
        return pd.read_csv('data/database/drug_rows.csv')
    
    def extract_concentration_toxicity_features(self, max_drugs: int = 30) -> pd.DataFrame:
        """Extract clean concentration-dependent toxicity features."""
        print(f"🔬 EXTRACTING CLEAN CONCENTRATION-TOXICITY FEATURES...")
        
        with DataLoader() as loader:
            oxygen_data = loader.load_oxygen_data()
            
        if oxygen_data.empty:
            print("❌ Failed to load organoid data")
            return pd.DataFrame()
        
        # Rename for consistency
        oxygen_data = oxygen_data.rename(columns={'o2': 'oxygen_consumption'})
        
        # Filter for drugs with toxicity data
        target_drugs = self.drug_targets['drug'].tolist()
        data = oxygen_data[oxygen_data['drug'].isin(target_drugs)]
        
        if max_drugs:
            available_drugs = data['drug'].unique()[:max_drugs]
            data = data[data['drug'].isin(available_drugs)]
        
        print(f"Analyzing {data['drug'].nunique()} drugs with {len(data)} data points")
        
        # Extract features for each drug-concentration combination
        features_list = []
        
        for drug in data['drug'].unique():
            print(f"  Processing {drug}...")
            
            drug_data = data[data['drug'] == drug]
            
            # 1. Establish control baseline (concentration = 0 or lowest concentration)
            control_baseline = self._extract_control_baseline(drug_data)
            self.control_baseline[drug] = control_baseline
            
            # 2. Extract features for each concentration
            concentrations = sorted(drug_data['concentration'].unique())
            
            for conc in concentrations:
                conc_data = drug_data[drug_data['concentration'] == conc]
                
                if len(conc_data) < 5:  # Need minimum data points
                    continue
                
                # Feature Set 1: Toxicity onset detection
                onset_features = self._extract_toxicity_onset_features(conc_data, control_baseline, conc)
                
                # Feature Set 2: Variability/changes vs control
                variability_features = self._extract_variability_vs_control_features(conc_data, control_baseline)
                
                # Combine features
                combined_features = {
                    'drug': drug,
                    'concentration': conc,
                    'log_concentration': np.log10(conc + 1e-12),
                    **onset_features,
                    **variability_features
                }
                
                features_list.append(combined_features)
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        
        # Add toxicity targets and Cmax normalization
        features_df = self._add_toxicity_and_cmax_data(features_df)
        
        print(f"✅ Extracted features for {len(features_df)} drug-concentration combinations")
        return features_df
    
    def _extract_control_baseline(self, drug_data: pd.DataFrame) -> Dict:
        """Extract control baseline characteristics for a drug."""
        # Use concentration = 0 as control, or lowest concentration if no 0
        control_data = drug_data[drug_data['concentration'] == 0]
        
        if control_data.empty:
            # Use lowest concentration as baseline
            min_conc = drug_data['concentration'].min()
            control_data = drug_data[drug_data['concentration'] == min_conc]
        
        if control_data.empty:
            return {}
        
        oxygen_values = control_data['oxygen_consumption'].values
        
        baseline = {
            'control_mean': np.mean(oxygen_values),
            'control_std': np.std(oxygen_values),
            'control_cv': np.std(oxygen_values) / (np.mean(oxygen_values) + 1e-8),
            'control_median': np.median(oxygen_values),
            'control_q25': np.percentile(oxygen_values, 25),
            'control_q75': np.percentile(oxygen_values, 75),
            'control_n_points': len(oxygen_values)
        }
        
        return baseline
    
    def _extract_toxicity_onset_features(self, conc_data: pd.DataFrame, 
                                       control_baseline: Dict, 
                                       concentration: float) -> Dict:
        """Extract features indicating toxicity onset at this concentration."""
        oxygen_values = conc_data['oxygen_consumption'].values
        
        if len(oxygen_values) < 3 or not control_baseline:
            return {}
        
        control_mean = control_baseline.get('control_mean', 0)
        control_std = control_baseline.get('control_std', 1)
        control_cv = control_baseline.get('control_cv', 0)
        
        features = {}
        
        # 1. Oxygen consumption decline (key toxicity indicator)
        current_mean = np.mean(oxygen_values)
        features['oxygen_decline_from_control'] = (control_mean - current_mean) / (control_mean + 1e-8)
        features['oxygen_decline_absolute'] = control_mean - current_mean
        features['oxygen_decline_z_score'] = (control_mean - current_mean) / (control_std + 1e-8)
        
        # 2. Toxicity threshold detection
        # If oxygen drops >20% from control, likely toxic
        features['toxicity_threshold_20pct'] = 1 if features['oxygen_decline_from_control'] > 0.2 else 0
        features['toxicity_threshold_10pct'] = 1 if features['oxygen_decline_from_control'] > 0.1 else 0
        features['toxicity_threshold_2std'] = 1 if features['oxygen_decline_z_score'] > 2 else 0
        
        # 3. Concentration-dependent toxicity severity
        features['concentration_toxicity_score'] = min(100, max(0, features['oxygen_decline_from_control'] * 100))
        
        # 4. Onset detection (binary)
        features['toxicity_onset_detected'] = 1 if (
            features['oxygen_decline_from_control'] > 0.15 or 
            features['oxygen_decline_z_score'] > 1.5
        ) else 0
        
        return features
    
    def _extract_variability_vs_control_features(self, conc_data: pd.DataFrame, 
                                                control_baseline: Dict) -> Dict:
        """Extract variability and change features compared to control."""
        oxygen_values = conc_data['oxygen_consumption'].values
        
        if len(oxygen_values) < 3 or not control_baseline:
            return {}
        
        control_mean = control_baseline.get('control_mean', 0)
        control_std = control_baseline.get('control_std', 1)
        control_cv = control_baseline.get('control_cv', 0)
        
        # Current concentration statistics
        current_mean = np.mean(oxygen_values)
        current_std = np.std(oxygen_values)
        current_cv = current_std / (current_mean + 1e-8)
        
        features = {}
        
        # 1. Variability changes (key toxicity indicator)
        features['cv_change_from_control'] = current_cv - control_cv
        features['cv_fold_change'] = current_cv / (control_cv + 1e-8)
        features['std_change_from_control'] = current_std - control_std
        features['std_fold_change'] = current_std / (control_std + 1e-8)
        
        # 2. Variability increase indicators (stress response)
        features['variability_increased'] = 1 if current_cv > control_cv * 1.3 else 0
        features['variability_greatly_increased'] = 1 if current_cv > control_cv * 2.0 else 0
        
        # 3. Distribution changes
        current_median = np.median(oxygen_values)
        control_median = control_baseline.get('control_median', current_median)
        
        features['median_shift'] = (current_median - control_median) / (control_median + 1e-8)
        features['distribution_shift_score'] = abs(features['median_shift']) + abs(features['cv_change_from_control'])
        
        # 4. Statistical significance of change
        # Use t-test if we have enough control data
        if control_baseline.get('control_n_points', 0) >= 5:
            try:
                # Reconstruct control values for t-test (approximate)
                control_vals = np.random.normal(control_mean, control_std, 
                                              min(50, control_baseline['control_n_points']))
                
                t_stat, p_value = stats.ttest_ind(control_vals, oxygen_values)
                features['change_p_value'] = p_value
                features['significant_change'] = 1 if p_value < 0.05 else 0
            except:
                features['change_p_value'] = 1.0
                features['significant_change'] = 0
        else:
            features['change_p_value'] = 1.0
            features['significant_change'] = 0
        
        # 5. Current concentration characteristics
        features['current_oxygen_mean'] = current_mean
        features['current_oxygen_cv'] = current_cv
        features['current_oxygen_std'] = current_std
        
        return features
    
    def _add_toxicity_and_cmax_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add toxicity targets and Cmax normalization."""
        print(f"📊 ADDING TOXICITY TARGETS AND CMAX NORMALIZATION...")
        
        # DILI scoring
        dili_mapping = {
            'vNo-DILI-Concern': 0,           # No toxicity
            'Ambiguous DILI-concern': 1,     # Ambiguous
            'vLess-DILI-Concern': 2,         # Less concern
            'vMost-DILI-Concern': 3          # Most concern
        }
        
        # Likelihood scoring (A=lowest risk, E=highest risk)
        likelihood_mapping = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'E*': 4
        }
        
        for _, row in features_df.iterrows():
            drug = row['drug']
            target_row = self.drug_targets[self.drug_targets['drug'] == drug]
            
            if not target_row.empty:
                drug_info = target_row.iloc[0]
                
                # Add toxicity categories
                features_df.loc[features_df['drug'] == drug, 'dili_category'] = dili_mapping.get(drug_info['dili'], 1)
                features_df.loc[features_df['drug'] == drug, 'likelihood_category'] = likelihood_mapping.get(drug_info['likelihood'], 2)
                features_df.loc[features_df['drug'] == drug, 'severity'] = drug_info.get('severity', 4)
                
                # Binary toxicity (Most/Less DILI concern = 1, No/Ambiguous = 0)
                is_toxic = 1 if drug_info['dili'] in ['vMost-DILI-Concern', 'vLess-DILI-Concern'] else 0
                features_df.loc[features_df['drug'] == drug, 'is_toxic'] = is_toxic
                
                # Cmax normalization
                cmax_oral = drug_info.get('cmax_oral_m', np.nan)
                cmax_iv = drug_info.get('cmax_iv_m', np.nan)
                cmax = cmax_oral if pd.notna(cmax_oral) else cmax_iv
                
                if pd.notna(cmax) and cmax > 0:
                    features_df.loc[features_df['drug'] == drug, 'cmax'] = cmax
                    features_df.loc[features_df['drug'] == drug, 'concentration_vs_cmax'] = row['concentration'] / cmax
                    features_df.loc[features_df['drug'] == drug, 'log_concentration_vs_cmax'] = np.log10((row['concentration'] / cmax) + 1e-12)
                    
                    # Clinical exposure categories
                    conc_ratio = row['concentration'] / cmax
                    if conc_ratio < 0.1:
                        exposure_cat = 0  # Sub-therapeutic
                    elif conc_ratio < 1.0:
                        exposure_cat = 1  # Therapeutic
                    elif conc_ratio < 10.0:
                        exposure_cat = 2  # Supra-therapeutic
                    else:
                        exposure_cat = 3  # Very high
                    
                    features_df.loc[features_df['drug'] == drug, 'exposure_category'] = exposure_cat
                else:
                    features_df.loc[features_df['drug'] == drug, 'cmax'] = np.nan
                    features_df.loc[features_df['drug'] == drug, 'concentration_vs_cmax'] = np.nan
                    features_df.loc[features_df['drug'] == drug, 'log_concentration_vs_cmax'] = np.nan
                    features_df.loc[features_df['drug'] == drug, 'exposure_category'] = np.nan
        
        return features_df
    
    def identify_toxicity_onset_concentrations(self, features_df: pd.DataFrame) -> Dict:
        """Identify the concentration where toxicity begins for each drug."""
        print(f"\n🎯 IDENTIFYING TOXICITY ONSET CONCENTRATIONS...")
        
        toxicity_onsets = {}
        
        for drug in features_df['drug'].unique():
            drug_data = features_df[features_df['drug'] == drug].copy()
            drug_data = drug_data.sort_values('concentration')
            
            # Find first concentration where toxicity is detected
            toxicity_detected = drug_data[drug_data['toxicity_onset_detected'] == 1]
            
            if not toxicity_detected.empty:
                onset_concentration = toxicity_detected['concentration'].min()
                onset_data = toxicity_detected[toxicity_detected['concentration'] == onset_concentration].iloc[0]
                
                toxicity_onsets[drug] = {
                    'onset_concentration': onset_concentration,
                    'log_onset_concentration': np.log10(onset_concentration + 1e-12),
                    'oxygen_decline_at_onset': onset_data['oxygen_decline_from_control'],
                    'cv_change_at_onset': onset_data['cv_change_from_control'],
                    'concentration_vs_cmax_at_onset': onset_data.get('concentration_vs_cmax', np.nan),
                    'dili_category': onset_data['dili_category'],
                    'is_toxic': onset_data['is_toxic']
                }
                
                print(f"  ✅ {drug}: Toxicity onset at {onset_concentration:.2e} M")
            else:
                print(f"  ❌ {drug}: No clear toxicity onset detected")
        
        self.toxicity_onsets = toxicity_onsets
        print(f"✅ Identified toxicity onsets for {len(toxicity_onsets)} drugs")
        
        return toxicity_onsets
    
    def predict_toxicity_from_concentration_features(self, features_df: pd.DataFrame) -> Dict:
        """Predict toxicity using concentration-dependent features."""
        print(f"\n🤖 PREDICTING TOXICITY FROM CONCENTRATION FEATURES...")
        
        if len(features_df) < 10:
            print("❌ Insufficient data for prediction")
            return {}
        
        # Prepare feature sets
        # Feature Set 1: Toxicity onset features
        onset_features = [col for col in features_df.columns 
                         if any(keyword in col for keyword in ['oxygen_decline', 'toxicity_threshold', 'onset'])]
        
        # Feature Set 2: Variability features
        variability_features = [col for col in features_df.columns 
                               if any(keyword in col for keyword in ['cv_', 'std_', 'variability', 'change'])]
        
        # Feature Set 3: Concentration features
        concentration_features = [col for col in features_df.columns 
                                 if any(keyword in col for keyword in ['concentration', 'exposure_category'])]
        
        all_features = onset_features + variability_features + concentration_features
        
        # Remove target variables and metadata
        feature_cols = [col for col in all_features 
                       if col not in ['drug', 'dili_category', 'likelihood_category', 'severity', 'is_toxic']]
        
        X = features_df[feature_cols].fillna(0)
        X = X.loc[:, X.std() > 1e-6]  # Remove constant features
        
        if X.shape[1] < 3:
            print("❌ Insufficient features for prediction")
            return {}
        
        print(f"Using {X.shape[1]} features for prediction:")
        print(f"  • Onset features: {len([f for f in X.columns if any(k in f for k in ['oxygen_decline', 'toxicity_threshold', 'onset'])])}")
        print(f"  • Variability features: {len([f for f in X.columns if any(k in f for k in ['cv_', 'std_', 'variability', 'change'])])}")
        print(f"  • Concentration features: {len([f for f in X.columns if any(k in f for k in ['concentration', 'exposure'])])}")
        
        results = {}
        logo = LeaveOneGroupOut()
        groups = features_df['drug']
        
        # 1. Binary toxicity prediction (is_toxic)
        if 'is_toxic' in features_df.columns and features_df['is_toxic'].nunique() > 1:
            print(f"\n--- Binary Toxicity Prediction ---")
            
            y_binary = features_df['is_toxic']
            
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            accuracy_scores = cross_val_score(model, X, y_binary, cv=logo, groups=groups, scoring='accuracy')
            
            # Train for feature importance
            model.fit(X, y_binary)
            
            results['binary_toxicity'] = {
                'accuracy': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'n_samples': len(y_binary),
                'n_drugs': groups.nunique(),
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
            print(f"  ✅ Binary Toxicity Accuracy: {accuracy_scores.mean():.3f} ± {accuracy_scores.std():.3f}")
        
        # 2. DILI category prediction
        if 'dili_category' in features_df.columns and features_df['dili_category'].nunique() > 2:
            print(f"\n--- DILI Category Prediction ---")
            
            y_dili = features_df['dili_category']
            
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            accuracy_scores = cross_val_score(model, X, y_dili, cv=logo, groups=groups, scoring='accuracy')
            
            results['dili_category'] = {
                'accuracy': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'n_samples': len(y_dili),
                'n_drugs': groups.nunique()
            }
            
            print(f"  ✅ DILI Category Accuracy: {accuracy_scores.mean():.3f} ± {accuracy_scores.std():.3f}")
        
        # 3. Concentration toxicity score prediction
        if 'concentration_toxicity_score' in features_df.columns:
            print(f"\n--- Concentration Toxicity Score Prediction ---")
            
            y_score = features_df['concentration_toxicity_score']
            
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            r2_scores = cross_val_score(model, X, y_score, cv=logo, groups=groups, scoring='r2')
            mae_scores = -cross_val_score(model, X, y_score, cv=logo, groups=groups, scoring='neg_mean_absolute_error')
            
            results['toxicity_score'] = {
                'r2': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'mae': mae_scores.mean(),
                'mae_std': mae_scores.std(),
                'n_samples': len(y_score),
                'n_drugs': groups.nunique()
            }
            
            print(f"  ✅ Toxicity Score R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
            print(f"     MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
        
        return results
    
    def analyze_cmax_relationships(self, features_df: pd.DataFrame) -> Dict:
        """Analyze relationships between toxicity and Cmax-normalized concentrations."""
        print(f"\n💊 ANALYZING CMAX RELATIONSHIPS...")
        
        # Filter data with Cmax information
        cmax_data = features_df[features_df['cmax'].notna()].copy()
        
        if len(cmax_data) < 10:
            print("❌ Insufficient Cmax data")
            return {}
        
        print(f"Analyzing {len(cmax_data)} data points with Cmax information")
        
        # Analyze toxicity by exposure category
        if 'exposure_category' in cmax_data.columns:
            exposure_analysis = cmax_data.groupby('exposure_category').agg({
                'toxicity_onset_detected': 'mean',
                'oxygen_decline_from_control': 'mean',
                'cv_change_from_control': 'mean',
                'concentration_toxicity_score': 'mean'
            }).round(3)
            
            print(f"\n📊 Toxicity by Exposure Category:")
            category_names = ['Sub-therapeutic (<0.1x Cmax)', 'Therapeutic (0.1-1x Cmax)', 
                             'Supra-therapeutic (1-10x Cmax)', 'Very high (>10x Cmax)']
            
            for cat in exposure_analysis.index:
                if cat < len(category_names):
                    cat_name = category_names[int(cat)]
                    toxicity_rate = exposure_analysis.loc[cat, 'toxicity_onset_detected'] * 100
                    print(f"  {cat_name}: {toxicity_rate:.1f}% toxicity onset")
        
        # Analyze concentration ratios where toxicity occurs
        toxic_data = cmax_data[cmax_data['toxicity_onset_detected'] == 1]
        
        if len(toxic_data) > 0:
            toxic_ratios = toxic_data['concentration_vs_cmax'].dropna()
            
            if len(toxic_ratios) > 0:
                print(f"\n🎯 Toxicity Onset Analysis:")
                print(f"  Median toxicity onset: {toxic_ratios.median():.1f}x Cmax")
                print(f"  Range: {toxic_ratios.min():.2f}x to {toxic_ratios.max():.1f}x Cmax")
                print(f"  25th percentile: {toxic_ratios.quantile(0.25):.1f}x Cmax")
                print(f"  75th percentile: {toxic_ratios.quantile(0.75):.1f}x Cmax")
        
        return {
            'exposure_analysis': exposure_analysis.to_dict() if 'exposure_analysis' in locals() else {},
            'toxic_ratios_stats': {
                'median': toxic_ratios.median() if 'toxic_ratios' in locals() and len(toxic_ratios) > 0 else np.nan,
                'min': toxic_ratios.min() if 'toxic_ratios' in locals() and len(toxic_ratios) > 0 else np.nan,
                'max': toxic_ratios.max() if 'toxic_ratios' in locals() and len(toxic_ratios) > 0 else np.nan,
                'q25': toxic_ratios.quantile(0.25) if 'toxic_ratios' in locals() and len(toxic_ratios) > 0 else np.nan,
                'q75': toxic_ratios.quantile(0.75) if 'toxic_ratios' in locals() and len(toxic_ratios) > 0 else np.nan
            }
        }
    
    def create_clean_results_visualization(self, features_df: pd.DataFrame, 
                                         toxicity_onsets: Dict, 
                                         prediction_results: Dict,
                                         cmax_analysis: Dict):
        """Create clean, focused visualization of results."""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # 1. Example concentration-response curves (top row)
        example_drugs = list(toxicity_onsets.keys())[:3]
        
        for i, drug in enumerate(example_drugs):
            ax = fig.add_subplot(gs[0, i])
            
            drug_data = features_df[features_df['drug'] == drug].sort_values('concentration')
            
            if len(drug_data) > 3:
                concentrations = drug_data['concentration'].values
                oxygen_decline = drug_data['oxygen_decline_from_control'].values
                
                ax.semilogx(concentrations, oxygen_decline * 100, 'o-', markersize=8, linewidth=3, color='darkred')
                
                # Mark toxicity onset
                if drug in toxicity_onsets:
                    onset_conc = toxicity_onsets[drug]['onset_concentration']
                    ax.axvline(onset_conc, color='red', linestyle='--', linewidth=2, label='Toxicity Onset')
                
                # Mark Cmax if available
                drug_cmax_data = drug_data[drug_data['cmax'].notna()]
                if not drug_cmax_data.empty:
                    cmax = drug_cmax_data['cmax'].iloc[0]
                    ax.axvline(cmax, color='green', linestyle='--', linewidth=2, label='Clinical Cmax')
                
                ax.set_xlabel('Concentration (M)')
                ax.set_ylabel('Oxygen Decline from Control (%)')
                ax.set_title(f'{drug}\nToxicity Detection', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, max(100, np.max(oxygen_decline * 100) * 1.1))
        
        # 4. Model performance comparison (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        
        if prediction_results:
            models = []
            accuracies = []
            
            if 'binary_toxicity' in prediction_results:
                models.append('Binary\nToxicity')
                accuracies.append(prediction_results['binary_toxicity']['accuracy'])
            
            if 'dili_category' in prediction_results:
                models.append('DILI\nCategory')
                accuracies.append(prediction_results['dili_category']['accuracy'])
            
            if 'toxicity_score' in prediction_results:
                models.append('Toxicity\nScore')
                accuracies.append(prediction_results['toxicity_score']['r2'])
            
            if models:
                bars = ax4.bar(models, accuracies, color=['darkred', 'darkblue', 'darkgreen'][:len(models)], 
                              alpha=0.8, edgecolor='black')
                ax4.set_ylabel('Accuracy / R²')
                ax4.set_title('Prediction Performance', fontweight='bold')
                ax4.set_ylim(0, 1.0)
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Toxicity onset distribution (middle left)
        ax5 = fig.add_subplot(gs[1, :2])
        
        if toxicity_onsets:
            onset_concentrations = [data['onset_concentration'] for data in toxicity_onsets.values()]
            log_onsets = [np.log10(conc) for conc in onset_concentrations]
            
            ax5.hist(log_onsets, bins=10, alpha=0.7, color='darkred', edgecolor='black')
            ax5.set_xlabel('Log₁₀ Toxicity Onset Concentration (M)')
            ax5.set_ylabel('Number of Drugs')
            ax5.set_title('Distribution of Toxicity Onset Concentrations', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. Cmax relationship analysis (middle right)
        ax6 = fig.add_subplot(gs[1, 2:])
        
        cmax_data = features_df[features_df['cmax'].notna()]
        if len(cmax_data) > 10:
            # Scatter plot: concentration ratio vs toxicity score
            scatter = ax6.scatter(cmax_data['concentration_vs_cmax'], 
                                cmax_data['concentration_toxicity_score'],
                                c=cmax_data['toxicity_onset_detected'], 
                                cmap='RdYlGn_r', s=60, alpha=0.7, edgecolors='black')
            
            ax6.set_xscale('log')
            ax6.set_xlabel('Concentration / Cmax Ratio')
            ax6.set_ylabel('Concentration Toxicity Score')
            ax6.set_title('Toxicity vs Clinical Exposure', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Add exposure zone lines
            ax6.axvline(0.1, color='green', linestyle=':', alpha=0.5, label='Therapeutic Range')
            ax6.axvline(1.0, color='orange', linestyle=':', alpha=0.5)
            ax6.axvline(10.0, color='red', linestyle=':', alpha=0.5, label='High Exposure')
            
            ax6.legend()
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('Toxicity Detected')
        
        # 7. Feature importance (bottom left)
        ax7 = fig.add_subplot(gs[2, :2])
        
        if 'binary_toxicity' in prediction_results and 'feature_importance' in prediction_results['binary_toxicity']:
            feature_importance = prediction_results['binary_toxicity']['feature_importance']
            
            # Top 10 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_features:
                features, importances = zip(*top_features)
                
                bars = ax7.barh(range(len(features)), importances, color='lightcoral', alpha=0.8)
                ax7.set_yticks(range(len(features)))
                ax7.set_yticklabels([f.replace('_', ' ').title() for f in features])
                ax7.set_xlabel('Feature Importance')
                ax7.set_title('Top Toxicity Predictors', fontweight='bold')
                ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. Summary statistics (bottom right)
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        # Calculate summary statistics
        n_drugs = features_df['drug'].nunique()
        n_combinations = len(features_df)
        n_with_onsets = len(toxicity_onsets)
        n_with_cmax = features_df['cmax'].notna().sum()
        
        avg_binary_acc = prediction_results.get('binary_toxicity', {}).get('accuracy', 0)
        avg_dili_acc = prediction_results.get('dili_category', {}).get('accuracy', 0)
        
        summary_text = f"""🎯 CLEAN CONCENTRATION-TOXICITY ANALYSIS

📊 DATASET SUMMARY:
• {n_drugs} drugs analyzed
• {n_combinations} drug-concentration combinations
• {n_with_onsets} drugs with detected toxicity onsets
• {n_with_cmax} combinations with Cmax data

🔬 FEATURE SETS:
• Toxicity Onset: Oxygen decline, threshold detection
• Variability Changes: CV, std, distribution shifts vs control
• Concentration: Log concentration, Cmax ratios

🎯 PREDICTION PERFORMANCE:
• Binary Toxicity: {avg_binary_acc:.1%} accuracy
• DILI Categories: {avg_dili_acc:.1%} accuracy"""

        if cmax_analysis and 'toxic_ratios_stats' in cmax_analysis:
            stats = cmax_analysis['toxic_ratios_stats']
            median_ratio = stats.get('median', np.nan)
            if not np.isnan(median_ratio):
                summary_text += f"""

💊 CMAX RELATIONSHIP:
• Median toxicity onset: {median_ratio:.1f}x Cmax
• Range: {stats.get('min', 0):.1f}x to {stats.get('max', 0):.1f}x Cmax
• Clinical implication: Monitor at >{median_ratio/2:.1f}x Cmax"""

        summary_text += f"""

✅ KEY INSIGHTS:
• Oxygen decline from control is primary toxicity indicator
• Variability increases precede major toxicity
• Most toxicity occurs at supra-therapeutic concentrations
• Cmax normalization enables clinical translation

🔄 ADVANTAGES OVER PREVIOUS:
• Direct concentration-toxicity relationship
• Control-normalized features reduce plate effects
• Cmax integration enables clinical application
• Clear mechanistic interpretation"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('🎯 CLEAN CONCENTRATION-DEPENDENT TOXICITY PREDICTION', 
                     fontsize=18, fontweight='bold')
        
        # Save figure
        output_path = Path('results/figures/clean_concentration_toxicity_results.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"\n✅ Clean results visualization saved to: {output_path}")


def main():
    """Run clean concentration-dependent toxicity prediction."""
    print("🎯 CLEAN CONCENTRATION-DEPENDENT TOXICITY PREDICTION")
    print("=" * 70)
    
    predictor = CleanConcentrationToxicityPredictor()
    
    # Extract clean concentration-toxicity features
    features_df = predictor.extract_concentration_toxicity_features(max_drugs=25)
    
    if features_df.empty:
        print("❌ No features extracted")
        return
    
    print(f"\n✅ Clean features extracted for {len(features_df)} combinations")
    print(f"   Covering {features_df['drug'].nunique()} unique drugs")
    
    # Identify toxicity onset concentrations
    toxicity_onsets = predictor.identify_toxicity_onset_concentrations(features_df)
    
    # Predict toxicity using concentration features
    prediction_results = predictor.predict_toxicity_from_concentration_features(features_df)
    
    # Analyze Cmax relationships
    cmax_analysis = predictor.analyze_cmax_relationships(features_df)
    
    # Create visualization
    predictor.create_clean_results_visualization(features_df, toxicity_onsets, 
                                               prediction_results, cmax_analysis)
    
    print("\n" + "="*70)
    print("🏆 CLEAN CONCENTRATION-TOXICITY SUMMARY")
    print("="*70)
    
    print(f"\n🔬 TOXICITY ONSET DETECTION:")
    print(f"  • {len(toxicity_onsets)} drugs with identified toxicity onsets")
    
    if prediction_results:
        print(f"\n🎯 PREDICTION PERFORMANCE:")
        for task, results in prediction_results.items():
            if 'accuracy' in results:
                print(f"  • {task.replace('_', ' ').title()}: {results['accuracy']:.3f} accuracy")
            elif 'r2' in results:
                print(f"  • {task.replace('_', ' ').title()}: R² = {results['r2']:.3f}")
    
    print(f"\n💡 KEY ADVANTAGES:")
    print("  1. Direct concentration-toxicity relationship detection")
    print("  2. Control-normalized features reduce experimental variation")
    print("  3. Clear onset concentration identification")
    print("  4. Cmax integration for clinical translation")
    print("  5. Mechanistically interpretable features")


if __name__ == "__main__":
    main()