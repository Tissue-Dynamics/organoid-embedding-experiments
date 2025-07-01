#!/usr/bin/env python3
"""
Integrated DILI-Concentration Toxicity Prediction System

Combines concentration-dependent toxicity modeling with DILI risk assessment to create
a comprehensive hepatotoxicity prediction system that answers:

1. At what concentration does liver toxicity occur?
2. What is the DILI severity at that concentration?
3. What is the safety margin vs clinical Cmax?
4. How does concentration relate to DILI likelihood categories?
5. What are the clinical risk implications?
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from scripts.analysis.concentration_dependent_toxicity_prediction import ConcentrationDependentToxicityPredictor

warnings.filterwarnings('ignore')


class IntegratedDILIConcentrationPredictor(ConcentrationDependentToxicityPredictor):
    """Integrate concentration-dependent toxicity with DILI risk assessment."""
    
    def __init__(self):
        super().__init__()
        self.dili_concentration_models = {}
        self.hepatotoxicity_thresholds = {}
        
    def extract_dili_specific_features(self, max_drugs: int = 30) -> pd.DataFrame:
        """Extract features specifically for DILI-concentration modeling."""
        print(f"🔬 EXTRACTING DILI-SPECIFIC CONCENTRATION FEATURES...")
        
        # Get base concentration features
        features_df = super().extract_concentration_dependent_features(max_drugs)
        
        if features_df.empty:
            return features_df
        
        # Add DILI-specific feature engineering
        print(f"📊 ADDING DILI-SPECIFIC BIOMARKERS...")
        
        # Hepatotoxicity-specific features
        for drug in features_df['drug'].unique():
            drug_data = features_df[features_df['drug'] == drug].copy()
            
            # Calculate DILI-specific dose-response features
            dili_features = self._extract_hepatotoxicity_biomarkers(drug_data)
            
            # Add to main dataframe
            for feature, value in dili_features.items():
                features_df.loc[features_df['drug'] == drug, feature] = value
        
        # Calculate concentration-DILI risk relationships
        features_df = self._add_dili_concentration_risk_scores(features_df)
        
        print(f"✅ Enhanced with DILI-specific features for {len(features_df)} combinations")
        return features_df
    
    def _extract_hepatotoxicity_biomarkers(self, drug_data: pd.DataFrame) -> Dict:
        """Extract liver-specific toxicity biomarkers from organoid response."""
        try:
            # Sort by concentration for dose-response analysis
            drug_data = drug_data.sort_values('concentration')
            
            concentrations = drug_data['concentration'].values
            
            # Hepatotoxicity biomarkers from organoid data
            biomarkers = {}
            
            # 1. Mitochondrial dysfunction indicators (key for DILI)
            if 'oxygen_mean' in drug_data.columns:
                oxygen_values = drug_data['oxygen_mean'].values
                
                # Calculate mitochondrial impairment progression
                baseline_oxygen = oxygen_values[0] if len(oxygen_values) > 0 else 0
                
                biomarkers.update({
                    'hepato_mitochondrial_decline_rate': self._calculate_dose_response_slope(concentrations, oxygen_values),
                    'hepato_oxygen_depletion_severity': (baseline_oxygen - np.min(oxygen_values)) / (baseline_oxygen + 1e-8),
                    'hepato_mitochondrial_recovery_capacity': np.std(oxygen_values) / (np.mean(oxygen_values) + 1e-8),
                })
            
            # 2. Metabolic stress indicators
            if 'oxygen_cv' in drug_data.columns:
                cv_values = drug_data['oxygen_cv'].values
                
                # Metabolic instability progression
                biomarkers.update({
                    'hepato_metabolic_instability_rate': self._calculate_dose_response_slope(concentrations, cv_values),
                    'hepato_metabolic_stress_threshold': self._find_threshold_concentration(drug_data, 'oxygen_cv', 1.5),
                    'hepato_metabolic_variability_max': np.max(cv_values),
                })
            
            # 3. Cellular stress response patterns
            if 'oxygen_std' in drug_data.columns:
                std_values = drug_data['oxygen_std'].values
                
                biomarkers.update({
                    'hepato_stress_response_amplitude': np.max(std_values) - np.min(std_values),
                    'hepato_stress_onset_concentration': self._find_threshold_concentration(drug_data, 'oxygen_std', 1.2),
                    'hepato_cellular_dysfunction_rate': self._calculate_dose_response_slope(concentrations, std_values),
                })
            
            # 4. DILI-specific concentration relationships
            if len(concentrations) >= 3:
                # Calculate liver-specific toxicity patterns
                biomarkers.update({
                    'hepato_concentration_span_orders': np.log10(concentrations.max() / (concentrations.min() + 1e-12)),
                    'hepato_dose_response_steepness': self._calculate_hill_steepness(drug_data),
                    'hepato_therapeutic_window': self._calculate_therapeutic_window(drug_data),
                })
            
            return biomarkers
        
        except Exception as e:
            print(f"    Error extracting hepatotoxicity biomarkers: {str(e)[:50]}")
            return {}
    
    def _add_dili_concentration_risk_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add DILI risk scores based on concentration-response relationships."""
        print(f"🎯 CALCULATING DILI CONCENTRATION RISK SCORES...")
        
        for drug in features_df['drug'].unique():
            drug_data = features_df[features_df['drug'] == drug].copy()
            
            if len(drug_data) < 3:
                continue
            
            # Get DILI severity information
            dili_score = drug_data['dili_score'].iloc[0] if 'dili_score' in drug_data.columns else 50
            likelihood_score = drug_data['likelihood_score'].iloc[0] if 'likelihood_score' in drug_data.columns else 50
            severity_score = drug_data['severity_score'].iloc[0] if 'severity_score' in drug_data.columns else 50
            
            # Calculate concentration-dependent DILI risk
            for idx, row in drug_data.iterrows():
                concentration = row['concentration']
                
                # DILI risk increases with concentration and intrinsic drug hepatotoxicity
                concentration_risk_factor = self._calculate_concentration_risk_factor(drug_data, concentration)
                
                # Integrated DILI-concentration risk scores
                integrated_dili_risk = (
                    0.4 * dili_score +  # Intrinsic drug DILI potential
                    0.3 * concentration_risk_factor +  # Concentration-dependent risk
                    0.2 * likelihood_score +  # Clinical likelihood
                    0.1 * severity_score  # Severity weighting
                )
                
                # Concentration-normalized risk (relative to Cmax)
                cmax_risk = 0
                if 'cmax' in drug_data.columns and pd.notna(row.get('cmax')):
                    cmax = row['cmax']
                    if cmax > 0:
                        concentration_multiple = concentration / cmax
                        cmax_risk = self._calculate_cmax_risk_score(concentration_multiple, dili_score)
                
                # Clinical hepatotoxicity probability
                hepatotox_probability = self._calculate_hepatotoxicity_probability(
                    concentration, dili_score, likelihood_score, severity_score
                )
                
                # Update dataframe
                features_df.loc[features_df.index == idx, 'integrated_dili_risk'] = integrated_dili_risk
                features_df.loc[features_df.index == idx, 'cmax_normalized_dili_risk'] = cmax_risk
                features_df.loc[features_df.index == idx, 'hepatotoxicity_probability'] = hepatotox_probability
                
                # DILI category-specific risk flags
                features_df.loc[features_df.index == idx, 'high_dili_risk_flag'] = (integrated_dili_risk >= 70)
                features_df.loc[features_df.index == idx, 'clinical_concern_flag'] = (cmax_risk >= 60)
        
        return features_df
    
    def _calculate_concentration_risk_factor(self, drug_data: pd.DataFrame, concentration: float) -> float:
        """Calculate concentration-dependent risk factor (0-100)."""
        try:
            concentrations = drug_data['concentration'].values
            
            # Risk increases with concentration percentile within drug's tested range
            concentration_percentile = np.percentile(concentrations, 
                                                   np.searchsorted(np.sort(concentrations), concentration) / len(concentrations) * 100)
            
            # Also consider absolute concentration magnitude
            log_concentration = np.log10(concentration + 1e-12)
            
            # Combine percentile and absolute magnitude
            risk_factor = min(100, concentration_percentile * 0.7 + min(50, abs(log_concentration) * 10))
            
            return risk_factor
        
        except Exception:
            return 50  # Default moderate risk
    
    def _calculate_cmax_risk_score(self, concentration_multiple: float, dili_score: float) -> float:
        """Calculate DILI risk based on concentration multiple of Cmax."""
        # Risk score based on exposure multiple and intrinsic DILI potential
        if concentration_multiple <= 0.1:
            exposure_risk = 10  # Sub-therapeutic
        elif concentration_multiple <= 1.0:
            exposure_risk = 30  # Therapeutic range
        elif concentration_multiple <= 3.0:
            exposure_risk = 50  # Moderate elevation
        elif concentration_multiple <= 10.0:
            exposure_risk = 70  # High exposure
        else:
            exposure_risk = 90  # Very high exposure
        
        # Combine with intrinsic drug DILI risk
        combined_risk = 0.6 * exposure_risk + 0.4 * dili_score
        return min(100, combined_risk)
    
    def _calculate_hepatotoxicity_probability(self, concentration: float, dili_score: float, 
                                           likelihood_score: float, severity_score: float) -> float:
        """Calculate probability of hepatotoxicity at given concentration."""
        # Sigmoid function combining multiple risk factors
        
        # Base probability from drug characteristics
        base_prob = (dili_score * 0.5 + likelihood_score * 0.3 + severity_score * 0.2) / 100
        
        # Concentration-dependent modifier
        log_conc = np.log10(concentration + 1e-12)
        conc_modifier = 1 / (1 + np.exp(-(log_conc + 5)))  # Sigmoid centered around 10^-5 M
        
        # Combined probability
        prob = base_prob * conc_modifier
        return min(1.0, prob)
    
    def model_dili_concentration_thresholds(self, features_df: pd.DataFrame) -> Dict:
        """Model DILI-specific concentration thresholds for clinical decision making."""
        print(f"\n🏥 MODELING DILI-SPECIFIC CONCENTRATION THRESHOLDS...")
        
        # Group by drug to calculate thresholds
        dili_thresholds = {}
        
        for drug in features_df['drug'].unique():
            drug_data = features_df[features_df['drug'] == drug].copy()
            drug_data = drug_data.sort_values('concentration')
            
            if len(drug_data) < 4:
                continue
            
            print(f"  Analyzing {drug}: DILI threshold identification")
            
            # DILI-specific thresholds
            thresholds = {
                'drug': drug,
                'dili_category': drug_data['dili_score'].iloc[0],
                'likelihood_category': drug_data['likelihood_score'].iloc[0],
                'severity_category': drug_data['severity_score'].iloc[0],
            }
            
            # 1. Hepatotoxicity onset concentration (where liver effects begin)
            hepatotox_onset = self._find_hepatotoxicity_onset(drug_data)
            if hepatotox_onset:
                thresholds['hepatotoxicity_onset_concentration'] = hepatotox_onset
            
            # 2. DILI concern threshold (clinical significance)
            dili_concern_threshold = self._find_dili_concern_threshold(drug_data)
            if dili_concern_threshold:
                thresholds['dili_concern_concentration'] = dili_concern_threshold
            
            # 3. Clinical safety margins
            cmax = drug_data['cmax'].iloc[0] if 'cmax' in drug_data.columns and pd.notna(drug_data['cmax'].iloc[0]) else None
            if cmax and hepatotox_onset:
                thresholds['hepatotox_safety_margin'] = hepatotox_onset / cmax
                thresholds['clinical_cmax'] = cmax
                
                # Clinical risk categories
                safety_margin = thresholds['hepatotox_safety_margin']
                if safety_margin < 3:
                    thresholds['clinical_risk_category'] = 'High Risk'
                elif safety_margin < 10:
                    thresholds['clinical_risk_category'] = 'Moderate Risk'
                else:
                    thresholds['clinical_risk_category'] = 'Low Risk'
            
            # 4. Therapeutic window analysis
            therapeutic_window = self._calculate_dili_therapeutic_window(drug_data)
            thresholds.update(therapeutic_window)
            
            dili_thresholds[drug] = thresholds
        
        self.hepatotoxicity_thresholds = dili_thresholds
        print(f"✅ Identified DILI thresholds for {len(dili_thresholds)} drugs")
        
        return dili_thresholds
    
    def _find_hepatotoxicity_onset(self, drug_data: pd.DataFrame) -> Optional[float]:
        """Find concentration where hepatotoxicity effects begin."""
        try:
            # Use multiple biomarkers to identify onset
            biomarkers = ['oxygen_cv', 'hepato_metabolic_instability_rate', 'hepato_stress_response_amplitude']
            
            onset_concentrations = []
            
            for biomarker in biomarkers:
                if biomarker in drug_data.columns:
                    threshold_conc = self._find_threshold_concentration(drug_data, biomarker, 1.3)
                    if threshold_conc:
                        onset_concentrations.append(threshold_conc)
            
            # Return median onset concentration if multiple biomarkers agree
            if onset_concentrations:
                return np.median(onset_concentrations)
            
            return None
        
        except Exception:
            return None
    
    def _find_dili_concern_threshold(self, drug_data: pd.DataFrame) -> Optional[float]:
        """Find concentration where DILI becomes clinically concerning."""
        try:
            # DILI concern based on integrated risk score
            if 'integrated_dili_risk' in drug_data.columns:
                # Find where integrated DILI risk exceeds 60 (concerning level)
                concerning_data = drug_data[drug_data['integrated_dili_risk'] >= 60]
                if not concerning_data.empty:
                    return concerning_data['concentration'].min()
            
            # Fallback: use CV threshold
            return self._find_threshold_concentration(drug_data, 'oxygen_cv', 2.0)
        
        except Exception:
            return None
    
    def _calculate_dili_therapeutic_window(self, drug_data: pd.DataFrame) -> Dict:
        """Calculate DILI-specific therapeutic window metrics."""
        try:
            window_metrics = {}
            
            # Find effective concentration (lowest with minimal toxicity)
            effective_conc = drug_data['concentration'].min()
            
            # Find hepatotoxicity onset
            hepatotox_onset = self._find_hepatotoxicity_onset(drug_data)
            
            if hepatotox_onset and effective_conc:
                # Therapeutic window = ratio of toxic to effective concentration
                therapeutic_window = hepatotox_onset / effective_conc
                window_metrics['therapeutic_window_ratio'] = therapeutic_window
                
                # Window classification
                if therapeutic_window < 3:
                    window_metrics['therapeutic_window_class'] = 'Narrow'
                elif therapeutic_window < 10:
                    window_metrics['therapeutic_window_class'] = 'Moderate'
                else:
                    window_metrics['therapeutic_window_class'] = 'Wide'
                
                # DILI-specific safety indices
                dili_score = drug_data['dili_score'].iloc[0]
                safety_index = therapeutic_window * (100 - dili_score) / 100
                window_metrics['dili_safety_index'] = safety_index
            
            return window_metrics
        
        except Exception:
            return {}
    
    def predict_clinical_dili_risk(self, features_df: pd.DataFrame) -> Dict:
        """Train models to predict clinical DILI risk at specific concentrations."""
        print(f"\n🎯 TRAINING CLINICAL DILI RISK PREDICTION MODELS...")
        
        if len(features_df) < 10:
            print("❌ Insufficient data for DILI risk modeling")
            return {}
        
        # Prepare features for modeling
        feature_cols = [col for col in features_df.columns 
                       if col.startswith(('hepato_', 'oxygen_', 'concentration', 'log_')) 
                       and col not in ['hepatotoxicity_probability']]
        
        X = features_df[feature_cols].fillna(0)
        X = X.loc[:, X.std() > 1e-6]  # Remove constant features
        
        if X.shape[1] < 3:
            print("❌ Insufficient features for DILI risk modeling")
            return {}
        
        results = {}
        
        # Predict multiple DILI-related outcomes
        targets = {
            'integrated_dili_risk': 'Integrated DILI Risk Score (0-100)',
            'hepatotoxicity_probability': 'Hepatotoxicity Probability (0-1)',
            'cmax_normalized_dili_risk': 'Cmax-Normalized DILI Risk (0-100)'
        }
        
        logo = LeaveOneGroupOut()
        groups = features_df['drug']
        
        for target, description in targets.items():
            if target not in features_df.columns:
                continue
            
            y = features_df[target].fillna(features_df[target].median())
            
            if y.nunique() < 3:
                continue
            
            print(f"\n--- {description} ---")
            
            try:
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Try multiple models
                models = [
                    ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                    ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42))
                ]
                
                best_r2 = -np.inf
                best_model_info = None
                
                for model_name, model in models:
                    r2_scores = cross_val_score(model, X_scaled, y, cv=logo, groups=groups, scoring='r2')
                    mae_scores = -cross_val_score(model, X_scaled, y, cv=logo, groups=groups, scoring='neg_mean_absolute_error')
                    
                    if r2_scores.mean() > best_r2:
                        best_r2 = r2_scores.mean()
                        
                        # Train on all data for feature importance
                        model.fit(X_scaled, y)
                        
                        best_model_info = {
                            'model_name': model_name,
                            'r2': r2_scores.mean(),
                            'r2_std': r2_scores.std(),
                            'mae': mae_scores.mean(),
                            'mae_std': mae_scores.std(),
                            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                            'description': description
                        }
                
                if best_model_info:
                    results[target] = best_model_info
                    print(f"  ✅ Best model: {best_model_info['model_name']}")
                    print(f"     R² = {best_model_info['r2']:.3f} ± {best_model_info['r2_std']:.3f}")
                    print(f"     MAE = {best_model_info['mae']:.3f} ± {best_model_info['mae_std']:.3f}")
            
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}")
        
        self.dili_concentration_models = results
        return results
    
    def create_clinical_dili_dashboard(self, features_df: pd.DataFrame, 
                                     dili_thresholds: Dict, 
                                     prediction_results: Dict):
        """Create comprehensive clinical DILI risk dashboard."""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
        
        # 1. DILI Risk vs Concentration (top row)
        example_drugs = list(dili_thresholds.keys())[:3]
        
        for i, drug in enumerate(example_drugs):
            ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
            
            drug_data = features_df[features_df['drug'] == drug].sort_values('concentration')
            
            if len(drug_data) > 3:
                concentrations = drug_data['concentration'].values
                dili_risk = drug_data['integrated_dili_risk'].values if 'integrated_dili_risk' in drug_data.columns else None
                
                if dili_risk is not None:
                    ax.semilogx(concentrations, dili_risk, 'o-', markersize=8, linewidth=3, color='darkred')
                    
                    # Add DILI thresholds
                    thresholds = dili_thresholds[drug]
                    
                    if 'hepatotoxicity_onset_concentration' in thresholds:
                        onset = thresholds['hepatotoxicity_onset_concentration']
                        ax.axvline(onset, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Hepatotox Onset')
                    
                    if 'dili_concern_concentration' in thresholds:
                        concern = thresholds['dili_concern_concentration']
                        ax.axvline(concern, color='red', linestyle='--', linewidth=2, alpha=0.8, label='DILI Concern')
                    
                    if 'clinical_cmax' in thresholds:
                        cmax = thresholds['clinical_cmax']
                        ax.axvline(cmax, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Clinical Cmax')
                    
                    # Risk zones
                    ax.axhspan(0, 30, alpha=0.1, color='green', label='Low Risk')
                    ax.axhspan(30, 60, alpha=0.1, color='yellow', label='Moderate Risk')
                    ax.axhspan(60, 100, alpha=0.1, color='red', label='High Risk')
                    
                    ax.set_xlabel('Concentration (M)')
                    ax.set_ylabel('Integrated DILI Risk Score')
                    ax.set_title(f'{drug}\nDILI Risk Profile', fontweight='bold', fontsize=12)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 100)
        
        # 2. Safety Margin Analysis (second row)
        ax4 = fig.add_subplot(gs[1, :3])
        
        safety_margins = []
        dili_categories = []
        drug_names = []
        
        for drug, thresholds in dili_thresholds.items():
            if 'hepatotox_safety_margin' in thresholds:
                safety_margins.append(thresholds['hepatotox_safety_margin'])
                dili_categories.append(thresholds['dili_category'])
                drug_names.append(drug)
        
        if safety_margins:
            scatter = ax4.scatter(dili_categories, safety_margins, 
                                c=dili_categories, cmap='RdYlGn_r', 
                                s=100, alpha=0.7, edgecolors='black')
            
            # Add safety margin thresholds
            ax4.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
            ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk Threshold')
            
            ax4.set_xlabel('DILI Score (Drug Intrinsic Risk)')
            ax4.set_ylabel('Safety Margin (Hepatotox Onset / Cmax)')
            ax4.set_title('DILI Safety Margins vs Drug Risk', fontweight='bold')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('DILI Score')
        
        # 3. Clinical Risk Categories (second row right)
        ax5 = fig.add_subplot(gs[1, 3:])
        
        risk_categories = {}
        for drug, thresholds in dili_thresholds.items():
            risk_cat = thresholds.get('clinical_risk_category', 'Unknown')
            if risk_cat not in risk_categories:
                risk_categories[risk_cat] = 0
            risk_categories[risk_cat] += 1
        
        if risk_categories:
            categories = list(risk_categories.keys())
            counts = list(risk_categories.values())
            colors = {'High Risk': 'red', 'Moderate Risk': 'orange', 'Low Risk': 'green', 'Unknown': 'gray'}
            bar_colors = [colors.get(cat, 'gray') for cat in categories]
            
            bars = ax5.bar(categories, counts, color=bar_colors, alpha=0.8, edgecolor='black')
            ax5.set_ylabel('Number of Drugs')
            ax5.set_title('Clinical DILI Risk Distribution', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Therapeutic Window Analysis (third row)
        ax6 = fig.add_subplot(gs[2, :3])
        
        window_data = []
        for drug, thresholds in dili_thresholds.items():
            if 'therapeutic_window_ratio' in thresholds:
                window_data.append({
                    'drug': drug,
                    'window_ratio': thresholds['therapeutic_window_ratio'],
                    'window_class': thresholds.get('therapeutic_window_class', 'Unknown'),
                    'dili_score': thresholds['dili_category'],
                    'safety_index': thresholds.get('dili_safety_index', 0)
                })
        
        if window_data:
            window_df = pd.DataFrame(window_data)
            
            scatter = ax6.scatter(window_df['dili_score'], window_df['window_ratio'],
                                c=window_df['safety_index'], cmap='RdYlGn',
                                s=100, alpha=0.7, edgecolors='black')
            
            ax6.set_xlabel('DILI Score (Drug Intrinsic Risk)')
            ax6.set_ylabel('Therapeutic Window Ratio')
            ax6.set_title('Therapeutic Window vs DILI Risk', fontweight='bold')
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3)
            
            # Add window class boundaries
            ax6.axhline(y=3, color='red', linestyle=':', alpha=0.5, label='Narrow Window')
            ax6.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='Moderate Window')
            ax6.legend()
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('DILI Safety Index')
        
        # 5. Model Performance (third row right)
        ax7 = fig.add_subplot(gs[2, 3:])
        
        if prediction_results:
            models = list(prediction_results.keys())
            r2_scores = [prediction_results[m]['r2'] for m in models]
            
            bars = ax7.bar(range(len(models)), r2_scores, 
                          color=['darkred', 'darkblue', 'darkgreen'][:len(models)], 
                          alpha=0.8, edgecolor='black')
            ax7.set_xticks(range(len(models)))
            ax7.set_xticklabels([m.replace('_', '\n').title() for m in models], fontsize=10)
            ax7.set_ylabel('R² Score')
            ax7.set_title('DILI Risk Prediction Performance', fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='y')
            ax7.set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, r2_scores):
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Feature Importance for DILI prediction (fourth row)
        ax8 = fig.add_subplot(gs[3, :3])
        
        if prediction_results and 'integrated_dili_risk' in prediction_results:
            feature_importance = prediction_results['integrated_dili_risk']['feature_importance']
            
            # Top 10 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_features:
                features, importances = zip(*top_features)
                
                bars = ax8.barh(range(len(features)), importances, color='lightcoral', alpha=0.8)
                ax8.set_yticks(range(len(features)))
                ax8.set_yticklabels([f.replace('_', ' ').title() for f in features])
                ax8.set_xlabel('Feature Importance')
                ax8.set_title('Top DILI Risk Predictors', fontweight='bold')
                ax8.grid(True, alpha=0.3, axis='x')
        
        # 7. Clinical Decision Matrix (fourth row right)
        ax9 = fig.add_subplot(gs[3, 3:])
        ax9.axis('off')
        
        # Create clinical decision matrix
        decision_text = """🏥 CLINICAL DILI DECISION MATRIX

🚨 HIGH RISK (Immediate Action Required):
• DILI Score >70 AND Safety Margin <3x
• Hepatotoxicity onset <Clinical Cmax
• Integrated DILI Risk >80

⚠️  MODERATE RISK (Enhanced Monitoring):
• DILI Score 40-70 OR Safety Margin 3-10x
• Hepatotoxicity onset 1-10x Clinical Cmax
• Integrated DILI Risk 40-80

✅ LOW RISK (Standard Monitoring):
• DILI Score <40 AND Safety Margin >10x
• Hepatotoxicity onset >10x Clinical Cmax
• Integrated DILI Risk <40

📊 RISK FACTORS HIERARCHY:
1. Safety Margin (Hepatotox/Cmax ratio)
2. Intrinsic Drug DILI Score
3. Concentration-Response Steepness
4. Therapeutic Window Ratio
5. Metabolic Stress Biomarkers

💊 CLINICAL APPLICATIONS:
• Starting dose selection (10-50% of threshold)
• Therapeutic drug monitoring thresholds
• Dose escalation safety margins
• Patient-specific risk stratification
• Regulatory decision support"""
        
        ax9.text(0.05, 0.95, decision_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('🏥 INTEGRATED DILI-CONCENTRATION CLINICAL DASHBOARD', 
                     fontsize=20, fontweight='bold')
        
        # Save figure
        output_path = Path('results/figures/integrated_dili_concentration_dashboard.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"\n✅ Clinical DILI dashboard saved to: {output_path}")


def main():
    """Run integrated DILI-concentration toxicity prediction analysis."""
    print("🏥 INTEGRATED DILI-CONCENTRATION TOXICITY PREDICTION")
    print("=" * 80)
    
    predictor = IntegratedDILIConcentrationPredictor()
    
    # Extract DILI-specific concentration features
    features_df = predictor.extract_dili_specific_features(max_drugs=25)
    
    if features_df.empty:
        print("❌ No features extracted")
        return
    
    print(f"\n✅ DILI-concentration features extracted for {len(features_df)} combinations")
    print(f"   Covering {features_df['drug'].nunique()} unique drugs")
    
    # Model DILI-specific concentration thresholds
    dili_thresholds = predictor.model_dili_concentration_thresholds(features_df)
    
    # Predict clinical DILI risk
    prediction_results = predictor.predict_clinical_dili_risk(features_df)
    
    # Create comprehensive clinical dashboard
    predictor.create_clinical_dili_dashboard(features_df, dili_thresholds, prediction_results)
    
    print("\n" + "="*80)
    print("🏆 INTEGRATED DILI-CONCENTRATION SUMMARY")
    print("="*80)
    
    print(f"\n🔬 DILI-SPECIFIC ANALYSIS:")
    print(f"  • {len(dili_thresholds)} drugs with DILI-concentration profiles")
    
    # Safety margin distribution
    safety_margins = [t.get('hepatotox_safety_margin', 0) for t in dili_thresholds.values() if 'hepatotox_safety_margin' in t]
    if safety_margins:
        high_risk = sum(1 for sm in safety_margins if sm < 3)
        moderate_risk = sum(1 for sm in safety_margins if 3 <= sm < 10)
        low_risk = sum(1 for sm in safety_margins if sm >= 10)
        
        print(f"  • Safety Margins: {high_risk} High Risk, {moderate_risk} Moderate Risk, {low_risk} Low Risk")
    
    if prediction_results:
        print(f"\n🎯 CLINICAL DILI RISK PREDICTION:")
        for target, results in prediction_results.items():
            print(f"  • {results['description']}")
            print(f"    R² = {results['r2']:.3f} ± {results['r2_std']:.3f}")
    
    print(f"\n💊 CLINICAL INTEGRATION BENEFITS:")
    print("  1. Concentration-specific DILI risk assessment")
    print("  2. Safety margin calculation vs clinical Cmax")
    print("  3. Hepatotoxicity onset prediction")
    print("  4. Therapeutic window optimization")
    print("  5. Patient-specific dose guidance")
    print("  6. Regulatory decision support with quantitative thresholds")


if __name__ == "__main__":
    main()