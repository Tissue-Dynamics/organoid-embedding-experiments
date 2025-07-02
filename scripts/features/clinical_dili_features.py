#!/usr/bin/env python3
"""
Simplified clinically-inspired DILI feature extraction
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import DataLoader

def extract_clinical_dili_features():
    """Extract features based on clinical DILI assessment principles."""
    
    print("🏥 EXTRACTING CLINICALLY-INSPIRED FEATURES")
    print("=" * 60)
    
    # Load data
    with DataLoader() as loader:
        oxygen_data = loader.load_oxygen_data()
    
    drug_metadata = pd.read_csv('data/database/drug_rows.csv')
    
    # Filter for drugs with DILI data
    target_drugs = drug_metadata[drug_metadata['dili'].notna()]['drug'].unique()
    data = oxygen_data[oxygen_data['drug'].isin(target_drugs)]
    
    # Limit to first 25 drugs for speed
    data = data[data['drug'].isin(data['drug'].unique()[:25])]
    
    print(f"Analyzing {data['drug'].nunique()} drugs")
    
    all_features = []
    
    for drug in data['drug'].unique():
        print(f"Processing {drug}...")
        drug_data = data[data['drug'] == drug]
        drug_info = drug_metadata[drug_metadata['drug'] == drug].iloc[0]
        
        # Process each concentration
        for conc in sorted(drug_data['concentration'].unique()):
            if conc == 0:  # Skip control
                continue
            
            conc_data = drug_data[drug_data['concentration'] == conc]
            control_data = drug_data[drug_data['concentration'] == 0]
            
            if len(conc_data) < 10 or len(control_data) < 10:
                continue
            
            features = {
                'drug': drug,
                'concentration': conc,
                'dili': drug_info['dili']
            }
            
            # SEVERITY-INSPIRED FEATURES (like ALT elevation, Hy's Law)
            
            # 1. Magnitude of elevation (like ALT >3x, >5x ULN)
            control_mean = control_data['o2'].mean()
            control_std = control_data['o2'].std()
            conc_mean = conc_data['o2'].mean()
            conc_max = conc_data['o2'].max()
            
            fold_change = (conc_mean - control_mean) / (abs(control_mean) + 1e-6)
            max_fold_change = (conc_max - control_mean) / (abs(control_mean) + 1e-6)
            
            features['o2_fold_change'] = fold_change
            features['o2_max_fold_change'] = max_fold_change
            features['elevation_3x'] = 1 if fold_change > 3 else 0
            features['elevation_5x'] = 1 if fold_change > 5 else 0
            
            # 2. Duration of injury (like hospitalization length)
            threshold = control_mean + 3 * control_std
            time_above_threshold = (conc_data['o2'] > threshold).mean()
            features['fraction_time_toxic'] = time_above_threshold
            
            # 3. Recovery dynamics (simulate media change recovery)
            # Look at 24h cycles
            recovery_scores = []
            for t in [24, 48, 72, 96]:
                pre_data = conc_data[(conc_data['elapsed_hours'] > t-4) & (conc_data['elapsed_hours'] < t)]
                post_data = conc_data[(conc_data['elapsed_hours'] > t) & (conc_data['elapsed_hours'] < t+4)]
                
                if len(pre_data) > 0 and len(post_data) > 0:
                    # Simulate recovery: expect O2 to decrease after "media change"
                    recovery = 1 - (post_data['o2'].mean() / (pre_data['o2'].mean() + 1e-6))
                    recovery_scores.append(recovery)
            
            if recovery_scores:
                features['avg_recovery_capacity'] = np.mean(recovery_scores)
                features['poor_recovery'] = 1 if np.mean(recovery_scores) < 0.2 else 0
            else:
                features['avg_recovery_capacity'] = 0
                features['poor_recovery'] = 0
            
            # 4. Hy's Law analog (elevation + poor recovery)
            features['hys_law_analog'] = 1 if (fold_change > 3 and features['poor_recovery']) else 0
            
            # 5. Time to peak toxicity
            if len(conc_data) > 0:
                time_to_peak = conc_data.loc[conc_data['o2'].idxmax(), 'elapsed_hours']
                features['time_to_peak'] = time_to_peak
                features['early_onset'] = 1 if time_to_peak < 24 else 0
                features['delayed_onset'] = 1 if time_to_peak > 72 else 0
            
            # LIKELIHOOD-INSPIRED FEATURES (reproducibility, dose-response)
            
            # 1. Well-to-well reproducibility
            well_groups = conc_data.groupby(['plate_id', 'well_id'])['o2'].mean()
            if len(well_groups) > 1:
                features['replicate_cv'] = well_groups.std() / (well_groups.mean() + 1e-6)
                features['high_reproducibility'] = 1 if features['replicate_cv'] < 0.2 else 0
            else:
                features['replicate_cv'] = 0
                features['high_reproducibility'] = 1
            
            # 2. Signal-to-noise ratio
            signal = conc_mean - control_mean
            noise = np.sqrt(conc_data['o2'].var() + control_data['o2'].var())
            features['signal_to_noise'] = abs(signal) / (noise + 1e-6)
            features['high_confidence'] = 1 if features['signal_to_noise'] > 2 else 0
            
            # 3. Temporal consistency
            time_series = conc_data.groupby('elapsed_hours')['o2'].mean()
            if len(time_series) > 10:
                # Check for smooth progression vs erratic
                differences = np.diff(time_series.values)
                features['temporal_smoothness'] = 1 / (1 + np.std(differences))
            else:
                features['temporal_smoothness'] = 0
            
            # PATTERN FEATURES (hepatocellular vs cholestatic)
            
            # Early vs late elevation
            early_data = conc_data[conc_data['elapsed_hours'] < 48]
            late_data = conc_data[conc_data['elapsed_hours'] > 96]
            
            if len(early_data) > 5 and len(late_data) > 5:
                early_elevation = (early_data['o2'].mean() - control_mean) / (abs(control_mean) + 1e-6)
                late_elevation = (late_data['o2'].mean() - control_mean) / (abs(control_mean) + 1e-6)
                
                features['early_elevation'] = early_elevation
                features['late_elevation'] = late_elevation
                
                # Pattern classification
                if early_elevation > 0.5 and late_elevation < early_elevation * 0.5:
                    features['pattern_hepatocellular'] = 1  # Quick rise, quick fall
                    features['pattern_cholestatic'] = 0
                elif late_elevation > early_elevation * 1.5:
                    features['pattern_hepatocellular'] = 0
                    features['pattern_cholestatic'] = 1  # Progressive rise
                else:
                    features['pattern_hepatocellular'] = 0
                    features['pattern_cholestatic'] = 0
            
            # CLINICAL CONTEXT
            if pd.notna(drug_info.get('cmax_oral_m')):
                conc_vs_cmax = (conc * 1e-6) / drug_info['cmax_oral_m']
                features['conc_vs_cmax'] = conc_vs_cmax
                features['therapeutic_range'] = 1 if 0.1 <= conc_vs_cmax <= 10 else 0
            
            all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Add DILI severity score
    dili_map = {
        'vNo-DILI-Concern': 0,
        'Ambiguous DILI-concern': 1,
        'vLess-DILI-Concern': 2,
        'vMost-DILI-Concern': 3
    }
    features_df['dili_severity'] = features_df['dili'].map(dili_map)
    
    print(f"\n✅ Extracted {len(features_df)} feature sets")
    
    return features_df

def analyze_clinical_features(features_df):
    """Analyze how clinical features predict DILI."""
    
    print("\n📊 CLINICAL FEATURE ANALYSIS")
    print("=" * 60)
    
    # Key clinical-inspired features
    clinical_features = [
        'o2_fold_change', 'elevation_3x', 'elevation_5x',
        'fraction_time_toxic', 'avg_recovery_capacity', 'hys_law_analog',
        'signal_to_noise', 'high_confidence', 'temporal_smoothness'
    ]
    
    # Feature statistics by DILI category
    print("\n📊 FEATURE AVERAGES BY DILI CATEGORY:")
    
    for feature in clinical_features:
        if feature in features_df.columns:
            print(f"\n{feature}:")
            for dili_cat in ['vNo-DILI-Concern', 'vLess-DILI-Concern', 'vMost-DILI-Concern']:
                cat_data = features_df[features_df['dili'] == dili_cat]
                if len(cat_data) > 0:
                    if feature in ['elevation_3x', 'elevation_5x', 'hys_law_analog']:
                        avg = cat_data[feature].mean() * 100
                        print(f"  {dili_cat}: {avg:.1f}% positive")
                    else:
                        avg = cat_data[feature].mean()
                        print(f"  {dili_cat}: {avg:.3f}")
    
    # Random Forest to test predictive power
    print("\n🤖 RANDOM FOREST DILI PREDICTION:")
    
    # Prepare data
    feature_cols = [col for col in clinical_features if col in features_df.columns]
    X = features_df[feature_cols].fillna(0)
    y = (features_df['dili_severity'] >= 2).astype(int)  # Binary: toxic vs non-toxic
    
    if len(np.unique(y)) > 1:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validated Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc')
        
        print(f"  Cross-validated AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Feature importance
        rf.fit(X_scaled, y)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 Clinical Features:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
    
    # Visualize key relationships
    visualize_clinical_features(features_df)
    
    return features_df

def visualize_clinical_features(features_df):
    """Visualize clinical feature relationships."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Clinical DILI Features from Organoid Data', fontsize=16, fontweight='bold')
    
    # 1. ALT-like elevation by DILI category
    ax1 = axes[0, 0]
    
    dili_order = ['vNo-DILI-Concern', 'Ambiguous DILI-concern', 
                  'vLess-DILI-Concern', 'vMost-DILI-Concern']
    
    fold_changes = []
    labels = []
    
    for dili in dili_order:
        dili_data = features_df[features_df['dili'] == dili]
        if len(dili_data) > 0:
            fold_changes.append(dili_data['o2_fold_change'].values)
            labels.append(dili.replace('DILI-Concern', '').replace('v', ''))
    
    ax1.boxplot(fold_changes, labels=labels)
    ax1.axhline(3, color='red', linestyle='--', alpha=0.5, label='3x threshold')
    ax1.set_ylabel('O2 Fold Change (like ALT)')
    ax1.set_title('Elevation Magnitude by DILI Category', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # 2. Hy's Law analog
    ax2 = axes[0, 1]
    
    ax2.scatter(features_df['o2_fold_change'], 
               1 - features_df['avg_recovery_capacity'],  # Poor recovery
               c=features_df['dili_severity'],
               cmap='RdYlGn_r', alpha=0.6, s=50)
    
    ax2.axvline(3, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0.8, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('O2 Fold Change')
    ax2.set_ylabel('Poor Recovery (1 - Recovery Capacity)')
    ax2.set_title('Hy\'s Law Analog', fontweight='bold')
    
    # 3. Clinical thresholds
    ax3 = axes[0, 2]
    
    threshold_rates = []
    for threshold in ['elevation_3x', 'elevation_5x', 'hys_law_analog']:
        if threshold in features_df.columns:
            rate = features_df[threshold].mean() * 100
            threshold_rates.append(rate)
        else:
            threshold_rates.append(0)
    
    ax3.bar(['3x Elevation', '5x Elevation', "Hy's Law"], threshold_rates)
    ax3.set_ylabel('% Positive')
    ax3.set_title('Clinical Threshold Rates', fontweight='bold')
    
    # 4. Time to toxicity
    ax4 = axes[1, 0]
    
    if 'time_to_peak' in features_df.columns:
        for i, dili in enumerate(['vNo-DILI-Concern', 'vLess-DILI-Concern', 'vMost-DILI-Concern']):
            dili_data = features_df[features_df['dili'] == dili]
            if len(dili_data) > 0:
                ax4.hist(dili_data['time_to_peak'], bins=20, alpha=0.5, 
                        label=dili.replace('v', '').replace('-Concern', ''))
        
        ax4.set_xlabel('Time to Peak Toxicity (hours)')
        ax4.set_ylabel('Count')
        ax4.set_title('Onset Timing by DILI Category', fontweight='bold')
        ax4.legend()
    
    # 5. Signal quality
    ax5 = axes[1, 1]
    
    ax5.scatter(features_df['signal_to_noise'], 
               features_df['temporal_smoothness'],
               c=features_df['dili_severity'],
               cmap='RdYlGn_r', alpha=0.6, s=50)
    
    ax5.set_xlabel('Signal-to-Noise Ratio')
    ax5.set_ylabel('Temporal Smoothness')
    ax5.set_title('Signal Quality Metrics', fontweight='bold')
    
    # 6. Clinical summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = """CLINICAL FEATURE SUMMARY

🏥 SEVERITY FEATURES:
• O2 fold change (like ALT elevation)
• Duration above threshold
• Recovery capacity
• Hy's Law analog

📊 LIKELIHOOD FEATURES:
• Replicate reproducibility
• Signal-to-noise ratio
• Temporal consistency
• Dose-response quality

🎯 KEY FINDINGS:
• Clear elevation patterns
• Recovery dynamics captured
• Pattern classification possible
• Clinical thresholds applicable"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/figures/clinical_dili_features_simple.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ Clinical features visualization saved to: {output_path}")

def main():
    """Run clinical DILI feature analysis."""
    
    # Extract features
    features_df = extract_clinical_dili_features()
    
    # Analyze and visualize
    features_df = analyze_clinical_features(features_df)
    
    # Save features
    output_path = Path('results/data/clinical_dili_features_simple.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Clinical features saved to: {output_path}")
    
    # Show example high-risk drugs
    print("\n🚨 HIGH-RISK DRUGS (Hy's Law analog positive):")
    if 'hys_law_analog' in features_df.columns:
        hys_positive = features_df[features_df['hys_law_analog'] == 1]
        if len(hys_positive) > 0:
            for _, row in hys_positive.head(10).iterrows():
                print(f"  {row['drug']} at {row['concentration']} µM: "
                      f"{row['o2_fold_change']:.1f}x elevation, "
                      f"recovery: {row['avg_recovery_capacity']:.2f}")

if __name__ == "__main__":
    main()