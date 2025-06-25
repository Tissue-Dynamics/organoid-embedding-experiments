#!/usr/bin/env python3
"""
Comprehensive Hill Curve Fitting for ALL Feature Types

PURPOSE:
    Apply Hill curve fitting to ALL extracted feature types to enable
    complete cross-drug comparison through pharmacological normalization.
    This implements Step 4.3 of the Advanced Feature Engineering Plan.

METHODOLOGY:
    - Applies 4-parameter Hill fitting to all feature types:
      * Event-aware features (consumption ratios, temporal features)
      * Hierarchical SAX features (coarse, medium, fine levels)
      * Multiscale catch22 features (24h, 48h, 96h windows)
      * Quality-aware features
    - Creates unified Hill parameter matrix for downstream analysis
    - Generates comprehensive quality assessment across all feature types

INPUTS:
    - results/data/event_aware_features_drugs.parquet
    - results/data/hierarchical_sax_features_drugs.parquet  
    - results/data/multiscale_catch22_drugs.parquet
    - results/data/quality_aware_features.parquet

OUTPUTS:
    - results/data/comprehensive_hill_parameters.parquet
      All Hill parameters for all feature types
    - results/data/comprehensive_feature_matrix.parquet
      Drug x Hill parameter matrix for embeddings
    - results/figures/comprehensive_dose_response/
      Visualizations comparing Hill fits across feature types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
from tqdm import tqdm
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "comprehensive_dose_response"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE HILL CURVE FITTING FOR ALL FEATURE TYPES")
print("=" * 80)

# Hill equation (same as dose_response_normalization.py)
def hill_equation(concentration, E0, Emax, EC50, n):
    """4-parameter Hill equation"""
    c = np.array(concentration)
    c = np.maximum(c, 1e-10)
    return E0 + (Emax - E0) * (c**n) / (EC50**n + c**n)

def fit_hill_curve(concentrations, responses, bounds=None):
    """Fit Hill curve with robust error handling"""
    # Remove NaN values
    mask = ~(np.isnan(concentrations) | np.isnan(responses))
    conc = np.array(concentrations)[mask]
    resp = np.array(responses)[mask]
    
    if len(conc) < 4:
        return None, None, {'R2': 0, 'RMSE': np.inf, 'success': False, 'error': 'insufficient_data'}
    
    # Initial parameter guesses
    E0_init = resp[conc == conc.min()].mean() if len(resp[conc == conc.min()]) > 0 else resp[0]
    Emax_init = resp[conc == conc.max()].mean() if len(resp[conc == conc.max()]) > 0 else resp[-1]
    EC50_init = np.median(conc)
    n_init = 1.0
    
    p0 = [E0_init, Emax_init, EC50_init, n_init]
    
    # Set bounds
    if bounds is None:
        bounds = (
            [resp.min() * 0.8, resp.min() * 0.8, conc.min() * 0.1, 0.1],
            [resp.max() * 1.2, resp.max() * 1.2, conc.max() * 10, 10]
        )
    
    try:
        # 4-parameter fit
        popt, pcov = curve_fit(hill_equation, conc, resp, p0=p0, bounds=bounds, maxfev=5000)
        
        # Calculate quality metrics
        y_pred = hill_equation(conc, *popt)
        residuals = resp - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((resp - np.mean(resp))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        params = {
            'E0': popt[0],
            'Emax': popt[1], 
            'EC50': popt[2],
            'n': popt[3],
            'log_EC50': np.log10(popt[2]) if popt[2] > 0 else np.nan
        }
        
        quality = {
            'R2': r2,
            'RMSE': rmse,
            'n_points': len(conc),
            'success': True,
            'model': '4-parameter'
        }
        
        return params, popt, quality
        
    except Exception as e:
        # Try 3-parameter fit
        try:
            def hill_3param(c, E0, Emax, EC50):
                return hill_equation(c, E0, Emax, EC50, 1)
            
            popt_3p, _ = curve_fit(hill_3param, conc, resp, 
                                  p0=[E0_init, Emax_init, EC50_init],
                                  bounds=(bounds[0][:3], bounds[1][:3]))
            
            params = {
                'E0': popt_3p[0],
                'Emax': popt_3p[1],
                'EC50': popt_3p[2],
                'n': 1.0,
                'log_EC50': np.log10(popt_3p[2]) if popt_3p[2] > 0 else np.nan
            }
            
            y_pred = hill_3param(conc, *popt_3p)
            r2 = 1 - np.sum((resp - y_pred)**2) / np.sum((resp - resp.mean())**2)
            
            quality = {
                'R2': r2,
                'RMSE': np.sqrt(np.mean((resp - y_pred)**2)),
                'n_points': len(conc),
                'success': True,
                'model': '3-parameter'
            }
            
            return params, np.array([*popt_3p, 1.0]), quality
            
        except:
            return None, None, {'R2': 0, 'RMSE': np.inf, 'success': False, 'error': str(e)}

def process_feature_dataframe(df, feature_type, desc):
    """Process a feature dataframe and extract Hill parameters"""
    print(f"\n🔄 Processing {feature_type} features...")
    print(f"   Data shape: {df.shape}")
    print(f"   Drugs: {df['drug'].nunique()}")
    
    # Get feature columns (exclude metadata)
    meta_cols = ['drug', 'concentration', 'n_wells', 'n_replicates']
    feature_cols = [col for col in df.columns if col not in meta_cols]
    print(f"   Features to fit: {len(feature_cols)}")
    
    all_results = []
    fitting_stats = {'total': 0, 'successful': 0, 'high_quality': 0}
    
    drugs = df['drug'].unique()
    
    for drug in tqdm(drugs, desc=f"Fitting {desc}"):
        drug_data = df[df['drug'] == drug].copy()
        concentrations = sorted(drug_data['concentration'].unique())
        
        if len(concentrations) < 4:
            continue
            
        for feature in feature_cols:
            fitting_stats['total'] += 1
            
            # Extract dose-response data
            dose_response = []
            for conc in concentrations:
                conc_vals = drug_data[drug_data['concentration'] == conc][feature].dropna()
                if len(conc_vals) > 0:
                    # Check if values are numeric
                    try:
                        numeric_vals = pd.to_numeric(conc_vals, errors='coerce').dropna()
                        if len(numeric_vals) > 0:
                            dose_response.append({
                                'concentration': conc,
                                'response': numeric_vals.mean(),
                                'std': numeric_vals.std(),
                                'n': len(numeric_vals)
                            })
                    except:
                        continue
            
            if len(dose_response) < 4:
                continue
                
            dr_df = pd.DataFrame(dose_response)
            
            # Fit Hill curve
            params, popt, quality = fit_hill_curve(
                dr_df['concentration'].values,
                dr_df['response'].values
            )
            
            if params is not None and quality['success']:
                fitting_stats['successful'] += 1
                
                if quality['R2'] > 0.7:
                    fitting_stats['high_quality'] += 1
                
                result = {
                    'drug': drug,
                    'feature': feature,
                    'feature_type': feature_type,
                    'n_concentrations': len(concentrations),
                    **params,
                    **quality
                }
                
                all_results.append(result)
    
    print(f"   Results: {fitting_stats['successful']}/{fitting_stats['total']} successful ({fitting_stats['successful']/fitting_stats['total']*100:.1f}%)")
    print(f"   High quality (R² > 0.7): {fitting_stats['high_quality']}")
    
    return pd.DataFrame(all_results)

# Load all feature datasets
print("\n📊 Loading feature datasets...")

# 1. Event-aware features
try:
    event_features_df = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")
    print(f"   Event-aware features: {event_features_df.shape}")
except FileNotFoundError:
    print("   Event-aware features: Not found")
    event_features_df = pd.DataFrame()

# 2. Hierarchical SAX features  
try:
    sax_features_df = pd.read_parquet(results_dir / "hierarchical_sax_features_drugs.parquet")
    print(f"   SAX features: {sax_features_df.shape}")
except FileNotFoundError:
    print("   SAX features: Not found")
    sax_features_df = pd.DataFrame()

# 3. Multiscale catch22 features
try:
    catch22_features_df = pd.read_parquet(results_dir / "multiscale_catch22_drugs.parquet")
    print(f"   catch22 features: {catch22_features_df.shape}")
except FileNotFoundError:
    print("   catch22 features: Not found")
    catch22_features_df = pd.DataFrame()

# 4. Quality-aware features
try:
    quality_features_df = pd.read_parquet(results_dir / "quality_aware_features.parquet")
    print(f"   Quality features: {quality_features_df.shape}")
except FileNotFoundError:
    print("   Quality features: Not found")
    quality_features_df = pd.DataFrame()

# Process each feature type
all_hill_results = []

# 1. Event-aware features - check if it has concentration data
if len(event_features_df) > 0:
    if 'concentration' in event_features_df.columns:
        event_results = process_feature_dataframe(event_features_df, "event_aware", "event-aware features")
        all_hill_results.append(event_results)
    else:
        print(f"   Event-aware features: Skipped (no concentration column - already aggregated)")

# 2. SAX features  
if len(sax_features_df) > 0:
    if 'concentration' in sax_features_df.columns:
        sax_results = process_feature_dataframe(sax_features_df, "hierarchical_sax", "SAX features")
        all_hill_results.append(sax_results)
    else:
        print(f"   SAX features: Skipped (no concentration column - already aggregated)")

# 3. catch22 features
if len(catch22_features_df) > 0:
    if 'concentration' in catch22_features_df.columns:
        catch22_results = process_feature_dataframe(catch22_features_df, "multiscale_catch22", "catch22 features")
        all_hill_results.append(catch22_results)
    else:
        print(f"   catch22 features: Skipped (no concentration column - already aggregated)")

# 4. Quality features
if len(quality_features_df) > 0:
    if 'concentration' in quality_features_df.columns:
        quality_results = process_feature_dataframe(quality_features_df, "quality_aware", "quality features")
        all_hill_results.append(quality_results)
    else:
        print(f"   Quality features: Skipped (no concentration column - already aggregated)")

# Combine all results
if all_hill_results:
    comprehensive_hill_df = pd.concat(all_hill_results, ignore_index=True)
    
    print(f"\n📊 COMPREHENSIVE HILL FITTING SUMMARY:")
    print(f"   Total fits: {len(comprehensive_hill_df)}")
    print(f"   Feature types: {comprehensive_hill_df['feature_type'].nunique()}")
    print(f"   Unique drugs: {comprehensive_hill_df['drug'].nunique()}")
    print(f"   High quality fits (R² > 0.7): {(comprehensive_hill_df['R2'] > 0.7).sum()}")
    
    # Summary by feature type
    print(f"\n📈 RESULTS BY FEATURE TYPE:")
    type_summary = comprehensive_hill_df.groupby('feature_type').agg({
        'R2': ['count', 'mean', lambda x: (x > 0.7).sum()],
        'log_EC50': ['mean', 'std']
    }).round(3)
    
    for feat_type in comprehensive_hill_df['feature_type'].unique():
        type_data = comprehensive_hill_df[comprehensive_hill_df['feature_type'] == feat_type]
        print(f"   {feat_type}:")
        print(f"      Fits: {len(type_data)}")
        print(f"      Mean R²: {type_data['R2'].mean():.3f}")
        print(f"      High quality: {(type_data['R2'] > 0.7).sum()}")
        print(f"      Mean log(EC50): {type_data['log_EC50'].mean():.2f}")
    
    # Save comprehensive results
    print(f"\n💾 Saving comprehensive Hill parameters...")
    comprehensive_hill_df.to_parquet(results_dir / 'comprehensive_hill_parameters.parquet', index=False)
    print(f"   Saved: {results_dir / 'comprehensive_hill_parameters.parquet'}")
    
    # Create comprehensive feature matrix
    print(f"\n🔄 Creating comprehensive feature matrix...")
    
    # Pivot to create drug x parameter matrix
    feature_matrix_data = []
    
    drugs = comprehensive_hill_df['drug'].unique()
    for drug in drugs:
        drug_data = comprehensive_hill_df[comprehensive_hill_df['drug'] == drug]
        
        # Only include good fits
        good_fits = drug_data[drug_data['R2'] > 0.5]
        
        if len(good_fits) == 0:
            continue
            
        drug_features = {'drug': drug}
        
        # Add Hill parameters for each feature
        for _, fit in good_fits.iterrows():
            feature_name = f"{fit['feature_type']}_{fit['feature']}"
            drug_features[f'{feature_name}_EC50'] = fit['EC50']
            drug_features[f'{feature_name}_log_EC50'] = fit['log_EC50']
            drug_features[f'{feature_name}_Emax'] = fit['Emax']
            drug_features[f'{feature_name}_E0'] = fit['E0']
            drug_features[f'{feature_name}_n'] = fit['n']
            drug_features[f'{feature_name}_R2'] = fit['R2']
        
        feature_matrix_data.append(drug_features)
    
    comprehensive_feature_matrix = pd.DataFrame(feature_matrix_data)
    comprehensive_feature_matrix.to_parquet(results_dir / 'comprehensive_feature_matrix.parquet', index=False)
    
    print(f"   Feature matrix: {comprehensive_feature_matrix.shape}")
    print(f"   Saved: {results_dir / 'comprehensive_feature_matrix.parquet'}")
    
    # Generate visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    
    # 1. Feature type comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hill Fitting Quality by Feature Type', fontsize=16, fontweight='bold')
    
    # R² distribution by feature type
    ax = axes[0, 0]
    feature_types = comprehensive_hill_df['feature_type'].unique()
    r2_data = [comprehensive_hill_df[comprehensive_hill_df['feature_type'] == ft]['R2'].values 
               for ft in feature_types]
    
    bp = ax.boxplot(r2_data, labels=[ft.replace('_', ' ').title() for ft in feature_types], 
                    patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('R²')
    ax.set_title('A. Fit Quality by Feature Type')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # EC50 distribution by feature type
    ax = axes[0, 1]
    ec50_data = [comprehensive_hill_df[comprehensive_hill_df['feature_type'] == ft]['log_EC50'].dropna().values 
                 for ft in feature_types]
    
    bp = ax.boxplot(ec50_data, labels=[ft.replace('_', ' ').title() for ft in feature_types], 
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('log10(EC50)')
    ax.set_title('B. Potency Distribution by Feature Type')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Success rate by feature type
    ax = axes[1, 0]
    success_rates = []
    for ft in feature_types:
        ft_data = comprehensive_hill_df[comprehensive_hill_df['feature_type'] == ft]
        total_possible = len(ft_data)
        high_quality = (ft_data['R2'] > 0.7).sum()
        success_rates.append(high_quality / total_possible if total_possible > 0 else 0)
    
    bars = ax.bar(range(len(feature_types)), success_rates, 
                  color=colors[:len(feature_types)], alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(feature_types)))
    ax.set_xticklabels([ft.replace('_', ' ').title() for ft in feature_types], rotation=45, ha='right')
    ax.set_ylabel('High Quality Fit Rate (R² > 0.7)')
    ax.set_title('C. Success Rate by Feature Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add numbers on bars
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')
    
    # Feature count by type
    ax = axes[1, 1]
    feature_counts = comprehensive_hill_df['feature_type'].value_counts()
    pie = ax.pie(feature_counts.values, labels=[ft.replace('_', ' ').title() for ft in feature_counts.index], 
                 autopct='%1.1f%%', colors=colors[:len(feature_counts)])
    ax.set_title('D. Distribution of Fitted Features')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'feature_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cross-feature EC50 correlation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create EC50 matrix for correlation
    ec50_pivot = comprehensive_hill_df[comprehensive_hill_df['R2'] > 0.7].pivot_table(
        index='drug', columns='feature_type', values='log_EC50', aggfunc='mean'
    )
    
    if ec50_pivot.shape[1] > 1:
        corr_matrix = ec50_pivot.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Cross-Feature Type EC50 Correlations\n(High Quality Fits Only)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'cross_feature_ec50_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   Visualizations saved to: {fig_dir}/")
    
    print(f"\n✅ Comprehensive Hill curve fitting complete!")
    print(f"   Total features with Hill parameters: {len(comprehensive_hill_df)}")
    print(f"   Drugs in feature matrix: {len(comprehensive_feature_matrix)}")
    print(f"   Next step: Use comprehensive feature matrix for drug embeddings")

else:
    print("\n❌ No feature datasets found for Hill fitting!")
    print("   Make sure the following files exist:")
    print("   - event_aware_features_drugs.parquet")
    print("   - hierarchical_sax_features_drugs.parquet") 
    print("   - multiscale_catch22_drugs.parquet")
    print("   - quality_aware_features.parquet")