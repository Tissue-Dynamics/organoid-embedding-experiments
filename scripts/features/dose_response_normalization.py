#!/usr/bin/env python3
"""
Dose-Response Normalization using Hill Curve Fitting

PURPOSE:
    Fits 4-parameter Hill curves to dose-response data for all event-aware features,
    enabling cross-drug comparison through standardized pharmacological parameters.
    This normalization is critical for comparing drugs with different potencies.

METHODOLOGY:
    - Fits Hill equation: f(c) = E0 + (Emax-E0) * c^n / (EC50^n + c^n)
    - Extracts key parameters: EC50 (potency), Emax (efficacy), n (Hill slope)
    - Handles fitting failures with fallback to 3-parameter models
    - Calculates fit quality metrics (R², RMSE) for each curve

INPUTS:
    - results/data/event_aware_features_wells.parquet
      Contains well-level features extracted from oxygen consumption data

OUTPUTS:
    - results/data/dose_response_hill_parameters.parquet
      All successful Hill curve fits with parameters and quality metrics
    - results/data/dose_response_drug_summary.parquet
      Drug-level summary with median parameters and top features
    - results/data/dose_response_feature_matrix.parquet
      Drug x Hill parameter matrix for downstream analysis
    - results/figures/dose_response/example_dose_response_curves.png
      Visualization of example Hill curve fits
    - results/figures/dose_response/hill_parameter_distributions.png
      Distribution plots of Hill parameters across all fits

REQUIREMENTS:
    - numpy, pandas, scipy, matplotlib, seaborn, joblib, tqdm
    - Event-aware features must be extracted first
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
import warnings
import joblib
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys

warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "dose_response"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DOSE-RESPONSE NORMALIZATION WITH HILL CURVES")
print("=" * 80)

# Hill equation function
def hill_equation(concentration, E0, Emax, EC50, n):
    """
    4-parameter Hill equation
    E0: baseline effect (no drug)
    Emax: maximum effect
    EC50: concentration at half-maximal effect
    n: Hill coefficient (slope)
    """
    c = np.array(concentration)
    # Add small epsilon to avoid log(0)
    c = np.maximum(c, 1e-10)
    return E0 + (Emax - E0) * (c**n) / (EC50**n + c**n)

def fit_hill_curve(concentrations, responses, bounds=None):
    """
    Fit Hill curve to dose-response data
    Returns parameters and fit quality metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(concentrations) | np.isnan(responses))
    conc = np.array(concentrations)[mask]
    resp = np.array(responses)[mask]
    
    if len(conc) < 4:  # Need at least 4 points for 4-parameter fit
        return None, None, {'R2': 0, 'RMSE': np.inf, 'success': False}
    
    # Log-transform concentrations for better fitting
    log_conc = np.log10(np.maximum(conc, 1e-10))
    
    # Initial parameter guesses
    E0_init = resp[conc == conc.min()].mean() if len(resp[conc == conc.min()]) > 0 else resp[0]
    Emax_init = resp[conc == conc.max()].mean() if len(resp[conc == conc.max()]) > 0 else resp[-1]
    EC50_init = np.median(conc)
    n_init = 1.0
    
    p0 = [E0_init, Emax_init, EC50_init, n_init]
    
    # Set bounds if not provided
    if bounds is None:
        bounds = (
            [resp.min() * 0.8, resp.min() * 0.8, conc.min() * 0.1, 0.1],  # Lower bounds
            [resp.max() * 1.2, resp.max() * 1.2, conc.max() * 10, 10]      # Upper bounds
        )
    
    try:
        # Fit the curve
        popt, pcov = curve_fit(hill_equation, conc, resp, p0=p0, bounds=bounds, maxfev=5000)
        
        # Calculate fit quality metrics
        y_pred = hill_equation(conc, *popt)
        residuals = resp - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((resp - np.mean(resp))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Extract parameters
        params = {
            'E0': popt[0],
            'Emax': popt[1],
            'EC50': popt[2],
            'n': popt[3],
            'log_EC50': np.log10(popt[2]) if popt[2] > 0 else np.nan
        }
        
        # Calculate parameter uncertainties
        if pcov is not None and not np.isinf(pcov).any():
            param_std = np.sqrt(np.diag(pcov))
            params.update({
                'E0_std': param_std[0],
                'Emax_std': param_std[1],
                'EC50_std': param_std[2],
                'n_std': param_std[3]
            })
        
        quality = {
            'R2': r2,
            'RMSE': rmse,
            'n_points': len(conc),
            'success': True,
            'convergence': True
        }
        
        return params, popt, quality
        
    except Exception as e:
        # Fitting failed - try simpler models
        try:
            # Try 3-parameter Hill (fix n=1)
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
                'convergence': False,
                'model': '3-parameter'
            }
            
            return params, np.array([*popt_3p, 1.0]), quality
            
        except:
            # Complete failure - return None
            return None, None, {'R2': 0, 'RMSE': np.inf, 'success': False, 'error': str(e)}

# Load event-aware features
print("\n📊 Loading feature data...")
features_df = pd.read_parquet(results_dir / "event_aware_features_wells.parquet")
print(f"   Loaded {len(features_df)} well-level features")
print(f"   Drugs: {features_df['drug'].nunique()}")
print(f"   Features: {len([c for c in features_df.columns if c not in ['well_id', 'drug', 'concentration']])}")

# Get feature columns
feature_cols = [col for col in features_df.columns 
               if col not in ['well_id', 'drug', 'concentration', 'n_events']]

print(f"\n🎯 Features to fit: {len(feature_cols)}")
for i, feat in enumerate(feature_cols[:5]):
    print(f"   {i+1}. {feat}")
if len(feature_cols) > 5:
    print(f"   ... and {len(feature_cols) - 5} more")

# Process each drug
drugs = features_df['drug'].unique()
print(f"\n🔄 Processing {len(drugs)} drugs...")

all_hill_params = []
fitting_summary = {
    'total_fits': 0,
    'successful_fits': 0,
    'high_quality_fits': 0,  # R2 > 0.7
    'failed_drugs': [],
    'best_fits': []
}

# Process each drug
for drug_idx, drug in enumerate(tqdm(drugs, desc="Fitting Hill curves")):
    drug_data = features_df[features_df['drug'] == drug].copy()
    
    # Get unique concentrations
    concentrations = sorted(drug_data['concentration'].unique())
    
    if len(concentrations) < 4:
        fitting_summary['failed_drugs'].append({
            'drug': drug,
            'reason': f'Insufficient concentrations ({len(concentrations)})'
        })
        continue
    
    # Process each feature
    for feature in feature_cols:
        fitting_summary['total_fits'] += 1
        
        # Aggregate feature values by concentration (mean of replicates)
        dose_response_data = []
        for conc in concentrations:
            conc_data = drug_data[drug_data['concentration'] == conc][feature].dropna()
            if len(conc_data) > 0:
                dose_response_data.append({
                    'concentration': conc,
                    'response_mean': conc_data.mean(),
                    'response_std': conc_data.std(),
                    'response_sem': conc_data.std() / np.sqrt(len(conc_data)),
                    'n_replicates': len(conc_data)
                })
        
        if len(dose_response_data) < 4:
            continue
            
        dr_df = pd.DataFrame(dose_response_data)
        
        # Fit Hill curve
        params, popt, quality = fit_hill_curve(
            dr_df['concentration'].values,
            dr_df['response_mean'].values
        )
        
        if params is not None and quality['success']:
            fitting_summary['successful_fits'] += 1
            
            if quality['R2'] > 0.7:
                fitting_summary['high_quality_fits'] += 1
            
            # Store results
            result = {
                'drug': drug,
                'feature': feature,
                'n_concentrations': len(concentrations),
                'concentration_range': f"{min(concentrations):.2e}-{max(concentrations):.2e}",
                **params,
                **quality
            }
            
            all_hill_params.append(result)
            
            # Track best fits
            if quality['R2'] > 0.9:
                fitting_summary['best_fits'].append({
                    'drug': drug,
                    'feature': feature,
                    'R2': quality['R2'],
                    'EC50': params['EC50']
                })

# Convert to DataFrame
hill_params_df = pd.DataFrame(all_hill_params)

print(f"\n📊 FITTING SUMMARY:")
print(f"   Total fits attempted: {fitting_summary['total_fits']}")
print(f"   Successful fits: {fitting_summary['successful_fits']} ({fitting_summary['successful_fits']/fitting_summary['total_fits']*100:.1f}%)")
print(f"   High quality (R² > 0.7): {fitting_summary['high_quality_fits']} ({fitting_summary['high_quality_fits']/fitting_summary['total_fits']*100:.1f}%)")
print(f"   Failed drugs: {len(fitting_summary['failed_drugs'])}")

# Show best fits
if fitting_summary['best_fits']:
    print(f"\n🏆 BEST FITS (R² > 0.9):")
    best_fits_sorted = sorted(fitting_summary['best_fits'], key=lambda x: x['R2'], reverse=True)[:5]
    for fit in best_fits_sorted:
        print(f"   {fit['drug']} - {fit['feature']}: R² = {fit['R2']:.3f}, EC50 = {fit['EC50']:.2e}")

# Analyze parameter distributions
print(f"\n📈 PARAMETER DISTRIBUTIONS:")

if len(hill_params_df) > 0:
    # EC50 distribution
    valid_ec50 = hill_params_df['log_EC50'].dropna()
    if len(valid_ec50) > 0:
        print(f"   log(EC50): {valid_ec50.mean():.2f} ± {valid_ec50.std():.2f}")
        print(f"   EC50 range: {10**valid_ec50.min():.2e} - {10**valid_ec50.max():.2e}")
    
    # Hill coefficient distribution
    valid_n = hill_params_df['n'].dropna()
    if len(valid_n) > 0:
        print(f"   Hill coefficient: {valid_n.mean():.2f} ± {valid_n.std():.2f}")
    
    # R² distribution
    print(f"   R² distribution:")
    print(f"      > 0.9: {(hill_params_df['R2'] > 0.9).sum()}")
    print(f"      > 0.7: {(hill_params_df['R2'] > 0.7).sum()}")
    print(f"      > 0.5: {(hill_params_df['R2'] > 0.5).sum()}")

# ========== VISUALIZATION FUNCTIONS ==========

def visualize_hill_curve_anatomy(save_path):
    """Educational visualization showing Hill curve parameters"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Understanding Hill Curve Parameters', fontsize=16, fontweight='bold')
    
    # Left panel: Standard Hill curve with annotations
    ax1.set_title('A. Hill Curve Anatomy', fontsize=12, loc='left')
    
    # Generate example curve
    conc = np.logspace(-3, 2, 1000)
    E0, Emax, EC50, n = 10, 90, 1, 2
    response = hill_equation(conc, E0, Emax, EC50, n)
    
    ax1.plot(conc, response, 'b-', linewidth=3, label='Hill Curve')
    
    # Annotate E0
    ax1.axhline(E0, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0.0005, E0+3, 'E0 (Baseline)', fontsize=10, ha='left')
    
    # Annotate Emax
    ax1.axhline(Emax, color='gray', linestyle='--', alpha=0.5)
    ax1.text(50, Emax-5, 'Emax (Maximum effect)', fontsize=10, ha='right')
    
    # Annotate EC50
    ax1.axvline(EC50, color='red', linestyle='--', alpha=0.7)
    ax1.scatter([EC50], [hill_equation(EC50, E0, Emax, EC50, n)], 
               color='red', s=100, zorder=5)
    ax1.text(EC50*1.5, 30, 'EC50\n(Half-maximal\nconcentration)', 
            fontsize=10, ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show dynamic range
    ax1.annotate('', xy=(0.01, E0), xytext=(0.01, Emax),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(0.005, (E0+Emax)/2, 'Dynamic\nRange', fontsize=10, 
            ha='right', va='center', color='green')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Concentration', fontsize=11)
    ax1.set_ylabel('Response', fontsize=11)
    ax1.set_xlim(0.001, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right panel: Effect of Hill coefficient
    ax2.set_title('B. Effect of Hill Coefficient (n)', fontsize=12, loc='left')
    
    for n_val, color in zip([0.5, 1, 2, 4], ['blue', 'green', 'orange', 'red']):
        response = hill_equation(conc, E0, Emax, EC50, n_val)
        ax2.plot(conc, response, color=color, linewidth=2, label=f'n = {n_val}')
    
    ax2.axvline(EC50, color='gray', linestyle='--', alpha=0.5)
    ax2.text(EC50*1.2, 20, 'EC50', fontsize=10)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Concentration', fontsize=11)
    ax2.set_ylabel('Response', fontsize=11)
    ax2.set_xlim(0.001, 100)
    ax2.legend(title='Hill Coefficient')
    ax2.grid(True, alpha=0.3)
    
    # Add text explanation
    ax2.text(0.98, 0.02, 
            'Higher n = Steeper curve\n(cooperative binding)\n\nLower n = Shallower curve\n(non-cooperative)',
            transform=ax2.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_fitting_process(features_df, drug, feature, save_path):
    """Show the fitting process for a specific drug-feature combination"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
    fig.suptitle(f'Hill Curve Fitting Process: {drug} - {feature}', fontsize=16, fontweight='bold')
    
    # Get data for this drug-feature combination
    drug_data = features_df[features_df['drug'] == drug].copy()
    
    # Main plot: Raw data and fitted curve
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_title('A. Dose-Response Data and Hill Curve Fit', fontsize=12, loc='left')
    
    # Plot individual replicates
    concentrations = sorted(drug_data['concentration'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(concentrations)))
    
    for i, conc in enumerate(concentrations):
        conc_data = drug_data[drug_data['concentration'] == conc][feature].dropna()
        if len(conc_data) > 0:
            # Plot individual points
            ax_main.scatter([conc] * len(conc_data), conc_data, 
                          color=colors[i], alpha=0.6, s=50, 
                          label=f'{conc:.1e} µM' if i < 5 else '')
            
            # Plot mean with error bar
            mean_val = conc_data.mean()
            sem_val = conc_data.std() / np.sqrt(len(conc_data))
            ax_main.errorbar(conc, mean_val, yerr=sem_val, 
                           fmt='o', color='black', markersize=8, 
                           capsize=5, capthick=2, elinewidth=2)
    
    # Fit and plot Hill curve
    dose_response_data = []
    for conc in concentrations:
        conc_data = drug_data[drug_data['concentration'] == conc][feature].dropna()
        if len(conc_data) > 0:
            dose_response_data.append({
                'concentration': conc,
                'response_mean': conc_data.mean(),
                'response_std': conc_data.std(),
                'n_replicates': len(conc_data)
            })
    
    if len(dose_response_data) >= 4:
        dr_df = pd.DataFrame(dose_response_data)
        params, popt, quality = fit_hill_curve(
            dr_df['concentration'].values,
            dr_df['response_mean'].values
        )
        
        if params is not None:
            # Plot fitted curve
            conc_range = np.logspace(np.log10(concentrations[0]*0.1), 
                                   np.log10(concentrations[-1]*10), 200)
            fitted_values = hill_equation(conc_range, params['E0'], params['Emax'], 
                                        params['EC50'], params['n'])
            ax_main.plot(conc_range, fitted_values, 'r-', linewidth=3, 
                       label=f'Hill fit (R² = {quality["R2"]:.3f})')
            
            # Mark EC50
            ax_main.axvline(params['EC50'], color='green', linestyle='--', 
                          alpha=0.7, linewidth=2)
            ax_main.text(params['EC50']*1.2, ax_main.get_ylim()[1]*0.9, 
                       f'EC50 = {params["EC50"]:.2e}', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax_main.set_xscale('log')
    ax_main.set_xlabel('Concentration (µM)', fontsize=11)
    ax_main.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
    ax_main.legend(loc='best', ncol=2)
    ax_main.grid(True, alpha=0.3)
    
    # Bottom left: Residuals plot
    ax_res = fig.add_subplot(gs[1, 0])
    ax_res.set_title('B. Fitting Residuals', fontsize=11, loc='left')
    
    if 'params' in locals() and params is not None:
        fitted_means = hill_equation(dr_df['concentration'].values, 
                                   params['E0'], params['Emax'], 
                                   params['EC50'], params['n'])
        residuals = dr_df['response_mean'].values - fitted_means
        
        ax_res.scatter(dr_df['concentration'].values, residuals, 
                      color='blue', alpha=0.7, s=60)
        ax_res.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax_res.set_xscale('log')
        ax_res.set_xlabel('Concentration (µM)', fontsize=10)
        ax_res.set_ylabel('Residual', fontsize=10)
        ax_res.grid(True, alpha=0.3)
    
    # Bottom middle: Parameter values
    ax_params = fig.add_subplot(gs[1, 1])
    ax_params.axis('off')
    
    if 'params' in locals() and params is not None:
        param_text = (
            f"Hill Parameters:\n\n"
            f"E0 (baseline) = {params['E0']:.2f}\n"
            f"Emax (maximum) = {params['Emax']:.2f}\n"
            f"EC50 = {params['EC50']:.2e} µM\n"
            f"n (Hill coef.) = {params['n']:.2f}\n\n"
            f"Fit Quality:\n"
            f"R² = {quality['R2']:.3f}\n"
            f"RMSE = {quality['RMSE']:.3f}"
        )
        ax_params.text(0.1, 0.5, param_text, fontsize=11, 
                      transform=ax_params.transAxes, va='center',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Bottom right: Replicate consistency
    ax_rep = fig.add_subplot(gs[1, 2])
    ax_rep.set_title('C. Replicate Consistency', fontsize=11, loc='left')
    
    # Calculate CV for each concentration
    cv_data = []
    for conc in concentrations:
        conc_data = drug_data[drug_data['concentration'] == conc][feature].dropna()
        if len(conc_data) > 1:
            cv = conc_data.std() / conc_data.mean() if conc_data.mean() != 0 else 0
            cv_data.append({'concentration': conc, 'cv': cv, 'n': len(conc_data)})
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        bars = ax_rep.bar(range(len(cv_df)), cv_df['cv'], 
                         color='orange', alpha=0.7, edgecolor='black')
        
        # Add sample size on top of bars
        for i, (bar, n) in enumerate(zip(bars, cv_df['n'])):
            ax_rep.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'n={n}', ha='center', va='bottom', fontsize=9)
        
        ax_rep.set_xticks(range(len(cv_df)))
        ax_rep.set_xticklabels([f'{c:.0e}' for c in cv_df['concentration']], rotation=45)
        ax_rep.set_xlabel('Concentration (µM)', fontsize=10)
        ax_rep.set_ylabel('Coefficient of Variation', fontsize=10)
        ax_rep.axhline(0.2, color='red', linestyle='--', alpha=0.5, 
                      label='CV = 0.2 threshold')
        ax_rep.legend()
        ax_rep.grid(True, alpha=0.3, axis='y')
    
    # Add comparison with linear fit
    ax_linear = fig.add_subplot(gs[2, :])
    ax_linear.set_title('D. Comparison with Linear Fit', fontsize=11, loc='left')
    
    if 'dr_df' in locals():
        # Plot data points
        ax_linear.scatter(dr_df['concentration'], dr_df['response_mean'], 
                         color='black', s=60, alpha=0.7, label='Data')
        
        # Linear fit in log space
        try:
            log_conc = np.log10(dr_df['concentration'])
            slope, intercept = np.polyfit(log_conc, dr_df['response_mean'], 1)
            linear_fit = slope * np.log10(conc_range) + intercept
            ax_linear.plot(conc_range, linear_fit, 'g--', linewidth=2, 
                          label=f'Linear fit (log-space)')
        except:
            # Skip linear fit if it fails
            pass
        
        # Hill fit
        if 'params' in locals() and params is not None:
            ax_linear.plot(conc_range, fitted_values, 'r-', linewidth=2, 
                         label=f'Hill fit (R² = {quality["R2"]:.3f})')
        
        ax_linear.set_xscale('log')
        ax_linear.set_xlabel('Concentration (µM)', fontsize=11)
        ax_linear.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax_linear.legend()
        ax_linear.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_quality_heatmap(hill_params_df, save_path):
    """Create heatmap of fit quality across drugs and features"""
    # Pivot to create drug x feature R² matrix
    quality_matrix = hill_params_df.pivot_table(
        index='drug', columns='feature', values='R2', aggfunc='mean'
    )
    
    # Sort by mean R² across features
    drug_order = quality_matrix.mean(axis=1).sort_values(ascending=False).index[:30]  # Top 30 drugs
    feature_order = quality_matrix.mean(axis=0).sort_values(ascending=False).index[:20]  # Top 20 features
    
    quality_subset = quality_matrix.loc[drug_order, feature_order]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Dose-Response Fit Quality Heatmap', fontsize=16, fontweight='bold')
    
    # Create heatmap
    sns.heatmap(quality_subset, cmap='RdYlGn', vmin=0, vmax=1, 
               annot=False, fmt='.2f', cbar_kws={'label': 'R²'},
               xticklabels=[f.replace('_mean', '').replace('_', ' ')[:15] 
                           for f in feature_order],
               yticklabels=[d[:20] for d in drug_order])
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Drugs', fontsize=12)
    ax.set_title('Fit Quality (R²) for Top Drugs and Features', fontsize=14, pad=20)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add quality indicators
    ax.text(1.15, 0.7, 'Good fit\n(R² > 0.7)', transform=ax.transAxes, 
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(1.15, 0.5, 'Moderate fit\n(0.5 < R² < 0.7)', transform=ax.transAxes, 
           fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.text(1.15, 0.3, 'Poor fit\n(R² < 0.5)', transform=ax.transAxes, 
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_ec50_comparison(hill_params_df, save_path):
    """Compare EC50 values across features to show normalization effect"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EC50-based Cross-Drug Comparison', fontsize=16, fontweight='bold')
    
    # Select features with good fits
    feature_quality = hill_params_df.groupby('feature')['R2'].mean().sort_values(ascending=False)
    top_features = feature_quality[feature_quality > 0.6].head(4).index
    
    for idx, feature in enumerate(top_features[:4]):
        ax = axes[idx // 2, idx % 2]
        
        # Get EC50 values for this feature
        feature_data = hill_params_df[
            (hill_params_df['feature'] == feature) & 
            (hill_params_df['R2'] > 0.5) &
            (hill_params_df['log_EC50'].notna())
        ].copy()
        
        if len(feature_data) > 5:
            # Sort by EC50
            feature_data = feature_data.sort_values('log_EC50')
            
            # Plot EC50 values
            y_pos = np.arange(len(feature_data))
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_data)))
            
            bars = ax.barh(y_pos, feature_data['log_EC50'], 
                          color=colors, alpha=0.7, edgecolor='black')
            
            # Add drug names
            ax.set_yticks(y_pos[::2])  # Show every other label to avoid crowding
            ax.set_yticklabels([d[:15] + '...' if len(d) > 15 else d 
                              for d in feature_data['drug'].iloc[::2]], fontsize=8)
            
            ax.set_xlabel('log10(EC50)', fontsize=11)
            ax.set_title(f'{feature.replace("_", " ").title()[:30]}\n(n={len(feature_data)} drugs)',
                        fontsize=11)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add potency categories
            ax.axvline(-6, color='green', linestyle='--', alpha=0.5)
            ax.axvline(-3, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            
            # Add legend
            if idx == 0:
                ax.text(0.02, 0.98, 'Potent\n(nM)', transform=ax.transAxes, 
                       fontsize=9, va='top', color='green',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.text(0.5, 0.98, 'Moderate\n(µM)', transform=ax.transAxes, 
                       fontsize=9, va='top', ha='center', color='orange',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.text(0.98, 0.98, 'Weak\n(mM)', transform=ax.transAxes, 
                       fontsize=9, va='top', ha='right', color='red',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_parameter_correlations(hill_params_df, save_path):
    """Analyze correlations between Hill parameters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hill Parameter Relationships', fontsize=16, fontweight='bold')
    
    # Filter high-quality fits
    hq_fits = hill_params_df[hill_params_df['R2'] > 0.7].copy()
    
    # 1. EC50 vs Emax
    ax = axes[0, 0]
    if len(hq_fits) > 10:
        scatter = ax.scatter(hq_fits['log_EC50'], hq_fits['Emax'], 
                           c=hq_fits['R2'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('log10(EC50)', fontsize=11)
        ax.set_ylabel('Emax', fontsize=11)
        ax.set_title('A. Potency vs Efficacy', fontsize=12, loc='left')
        plt.colorbar(scatter, ax=ax, label='R²')
        ax.grid(True, alpha=0.3)
    
    # 2. EC50 vs Hill coefficient
    ax = axes[0, 1]
    if len(hq_fits) > 10:
        ax.scatter(hq_fits['log_EC50'], hq_fits['n'], 
                  c=hq_fits['R2'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('log10(EC50)', fontsize=11)
        ax.set_ylabel('Hill Coefficient (n)', fontsize=11)
        ax.set_title('B. Potency vs Cooperativity', fontsize=12, loc='left')
        ax.set_ylim(0, 5)
        ax.grid(True, alpha=0.3)
    
    # 3. Dynamic range vs Hill coefficient
    ax = axes[0, 2]
    if len(hq_fits) > 10:
        hq_fits['dynamic_range'] = np.abs(hq_fits['Emax'] - hq_fits['E0'])
        ax.scatter(hq_fits['n'], hq_fits['dynamic_range'], 
                  alpha=0.6, s=50, color='purple')
        ax.set_xlabel('Hill Coefficient (n)', fontsize=11)
        ax.set_ylabel('Dynamic Range (|Emax - E0|)', fontsize=11)
        ax.set_title('C. Cooperativity vs Response Range', fontsize=12, loc='left')
        ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.3)
    
    # 4. Feature-grouped parameter distributions
    ax = axes[1, 0]
    # Group features by type
    consumption_features = [f for f in hq_fits['feature'].unique() if 'consumption' in f]
    cv_features = [f for f in hq_fits['feature'].unique() if 'cv' in f]
    other_features = [f for f in hq_fits['feature'].unique() 
                     if f not in consumption_features and f not in cv_features]
    
    feature_groups = {
        'Consumption': consumption_features[:5],
        'Variability (CV)': cv_features[:5],
        'Other': other_features[:5]
    }
    
    ec50_by_group = []
    labels = []
    for group, features in feature_groups.items():
        group_data = hq_fits[hq_fits['feature'].isin(features)]['log_EC50'].dropna()
        if len(group_data) > 0:
            ec50_by_group.append(group_data)
            labels.append(f'{group}\n(n={len(group_data)})')
    
    if ec50_by_group:
        bp = ax.boxplot(ec50_by_group, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax.set_ylabel('log10(EC50)', fontsize=11)
        ax.set_title('D. EC50 by Feature Type', fontsize=12, loc='left')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 5. R² vs parameter uncertainty
    ax = axes[1, 1]
    if 'EC50_std' in hq_fits.columns:
        # Calculate relative uncertainty
        hq_fits['EC50_rel_uncertainty'] = hq_fits['EC50_std'] / hq_fits['EC50']
        valid_data = hq_fits[hq_fits['EC50_rel_uncertainty'] < 10]  # Remove outliers
        
        if len(valid_data) > 10:
            ax.scatter(valid_data['R2'], valid_data['EC50_rel_uncertainty'], 
                      alpha=0.6, s=50, color='coral')
            ax.set_xlabel('R²', fontsize=11)
            ax.set_ylabel('EC50 Relative Uncertainty', fontsize=11)
            ax.set_title('E. Fit Quality vs Parameter Confidence', fontsize=12, loc='left')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = (
        f"Summary Statistics (High-Quality Fits):\n\n"
        f"Total fits: {len(hq_fits)}\n"
        f"Unique drugs: {hq_fits['drug'].nunique()}\n"
        f"Unique features: {hq_fits['feature'].nunique()}\n\n"
        f"EC50 range: {10**hq_fits['log_EC50'].min():.2e} - {10**hq_fits['log_EC50'].max():.2e}\n"
        f"Median EC50: {10**hq_fits['log_EC50'].median():.2e}\n\n"
        f"Hill coefficient:\n"
        f"  Mean: {hq_fits['n'].mean():.2f}\n"
        f"  Std: {hq_fits['n'].std():.2f}\n"
        f"  Range: {hq_fits['n'].min():.2f} - {hq_fits['n'].max():.2f}"
    )
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, transform=ax.transAxes, 
           va='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_drug_clustering(hill_features_df, save_path):
    """Cluster drugs based on their Hill parameters across features"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Drug Clustering based on Hill Parameters', fontsize=16, fontweight='bold')
    
    # Select EC50 features
    ec50_cols = [col for col in hill_features_df.columns 
                if '_log_EC50' in col and not col.endswith('_std')]
    
    if len(ec50_cols) > 3:
        # Prepare data
        X = hill_features_df[ec50_cols].fillna(hill_features_df[ec50_cols].median())
        drug_names = hill_features_df['drug'].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        ax = axes[0]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=100)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
        ax.set_title('PCA of EC50 Values', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add drug labels for outliers
        for i, drug in enumerate(drug_names):
            if (np.abs(X_pca[i, 0]) > np.percentile(np.abs(X_pca[:, 0]), 90) or 
                np.abs(X_pca[i, 1]) > np.percentile(np.abs(X_pca[:, 1]), 90)):
                ax.annotate(drug[:15], (X_pca[i, 0], X_pca[i, 1]), 
                          fontsize=8, alpha=0.7)
        
        # Feature contribution biplot
        ax = axes[1]
        ax.set_title('Feature Contributions to PCs', fontsize=12)
        
        # Plot feature vectors
        for i, feature in enumerate(ec50_cols[:10]):  # Top 10 features
            ax.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
                    head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
            ax.text(pca.components_[0, i]*3.2, pca.components_[1, i]*3.2,
                   feature.replace('_log_EC50', '').replace('_', ' ')[:15],
                   fontsize=8, ha='center')
        
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('PC1 Loading', fontsize=11)
        ax.set_ylabel('PC2 Loading', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        
        # EC50 range heatmap
        ax = axes[2]
        # Calculate EC50 range for each drug
        ec50_ranges = []
        for _, row in hill_features_df.iterrows():
            ec50_vals = row[ec50_cols].dropna()
            if len(ec50_vals) > 2:
                ec50_ranges.append({
                    'drug': row['drug'],
                    'min_log_ec50': ec50_vals.min(),
                    'max_log_ec50': ec50_vals.max(),
                    'range': ec50_vals.max() - ec50_vals.min(),
                    'n_features': len(ec50_vals)
                })
        
        if ec50_ranges:
            range_df = pd.DataFrame(ec50_ranges).sort_values('range', ascending=False).head(20)
            
            # Create range plot
            y_pos = np.arange(len(range_df))
            ax.barh(y_pos, range_df['range'], color='lightcoral', alpha=0.7, edgecolor='black')
            
            # Add min/max markers
            ax.scatter(range_df['min_log_ec50'], y_pos, marker='|', s=100, color='blue', label='Min')
            ax.scatter(range_df['max_log_ec50'], y_pos, marker='|', s=100, color='red', label='Max')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([d[:20] for d in range_df['drug']], fontsize=9)
            ax.set_xlabel('log10(EC50)', fontsize=11)
            ax.set_title('EC50 Range Across Features\n(Top 20 Drugs)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========== GENERATE ALL VISUALIZATIONS ==========

print(f"\n📊 Creating comprehensive visualizations...")

# 1. Educational visualization of Hill curve
print("   Creating Hill curve anatomy visualization...")
visualize_hill_curve_anatomy(fig_dir / 'hill_curve_anatomy.png')

# 2. Example fitting process
if len(hill_params_df) > 0:
    # Select a good example
    good_examples = hill_params_df[hill_params_df['R2'] > 0.8]
    if len(good_examples) > 0:
        example = good_examples.iloc[0]
        print(f"   Creating fitting process visualization for {example['drug']} - {example['feature']}...")
        visualize_fitting_process(features_df, example['drug'], example['feature'], 
                                fig_dir / 'fitting_process_example.png')

# 3. Quality heatmap
print("   Creating fit quality heatmap...")
visualize_quality_heatmap(hill_params_df, fig_dir / 'fit_quality_heatmap.png')

# 4. EC50 comparison
print("   Creating EC50 comparison visualization...")
visualize_ec50_comparison(hill_params_df, fig_dir / 'ec50_comparison.png')

# 5. Parameter correlations
print("   Creating parameter correlation analysis...")
visualize_parameter_correlations(hill_params_df, fig_dir / 'parameter_correlations.png')

# Drug clustering will be done after hill_features_df is created

# Keep the original simple visualizations for backward compatibility
print("   Creating standard summary visualizations...")

# Figure 1: Example dose-response curves
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Example Dose-Response Curves with Hill Fits', fontsize=16, fontweight='bold')

# Select 6 example fits (3 good, 3 poor)
if len(hill_params_df[hill_params_df['R2'] > 0.8]) > 0:
    good_fits = hill_params_df[hill_params_df['R2'] > 0.8].sample(
        min(3, len(hill_params_df[hill_params_df['R2'] > 0.8])))
else:
    good_fits = pd.DataFrame()

if len(hill_params_df[hill_params_df['R2'] < 0.5]) > 0:
    poor_fits = hill_params_df[hill_params_df['R2'] < 0.5].sample(
        min(3, len(hill_params_df[hill_params_df['R2'] < 0.5])))
else:
    poor_fits = pd.DataFrame()

example_fits = pd.concat([good_fits, poor_fits])

for idx, (_, fit) in enumerate(example_fits.iterrows()):
    if idx >= 6:
        break
    ax = axes[idx // 3, idx % 3]
    
    # Get original data
    drug_data = features_df[features_df['drug'] == fit['drug']]
    feature_data = drug_data[['concentration', fit['feature']]].dropna()
    
    # Plot raw data
    ax.scatter(feature_data['concentration'], feature_data[fit['feature']], 
              alpha=0.6, s=50, label='Raw data')
    
    # Plot fitted curve
    if not pd.isna(fit['EC50']):
        conc_range = np.logspace(np.log10(feature_data['concentration'].min()),
                                np.log10(feature_data['concentration'].max()), 100)
        fitted_values = hill_equation(conc_range, fit['E0'], fit['Emax'], fit['EC50'], fit['n'])
        ax.plot(conc_range, fitted_values, 'r-', linewidth=2, label=f'Hill fit (R²={fit["R2"]:.3f})')
        
        # Mark EC50
        ax.axvline(fit['EC50'], color='green', linestyle='--', alpha=0.5, label=f'EC50={fit["EC50"]:.2e}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Concentration')
    ax.set_ylabel(fit['feature'].replace('_', ' ').title()[:20] + '...')
    ax.set_title(f'{fit["drug"][:15]}...\nR² = {fit["R2"]:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'example_dose_response_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Parameter distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hill Parameter Distributions', fontsize=16, fontweight='bold')

# log(EC50) distribution
ax = axes[0, 0]
valid_log_ec50 = hill_params_df['log_EC50'].dropna()
if len(valid_log_ec50) > 0:
    ax.hist(valid_log_ec50, bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(valid_log_ec50.median(), color='red', linestyle='--', 
              label=f'Median: {valid_log_ec50.median():.2f}')
    ax.set_xlabel('log10(EC50)')
    ax.set_ylabel('Frequency')
    ax.set_title('EC50 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hill coefficient distribution
ax = axes[0, 1]
valid_n = hill_params_df['n'].dropna()
if len(valid_n) > 0:
    ax.hist(valid_n[valid_n < 10], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(1, color='red', linestyle='--', label='n = 1 (standard)')
    ax.set_xlabel('Hill Coefficient (n)')
    ax.set_ylabel('Frequency')
    ax.set_title('Hill Slope Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# R² distribution
ax = axes[1, 0]
ax.hist(hill_params_df['R2'], bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(0.7, color='red', linestyle='--', label='R² = 0.7 threshold')
ax.set_xlabel('R²')
ax.set_ylabel('Frequency')
ax.set_title('Fit Quality Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Feature-wise success rate
ax = axes[1, 1]
feature_success = hill_params_df.groupby('feature').agg({
    'R2': ['mean', 'count'],
    'success': 'sum'
}).sort_values(('R2', 'mean'), ascending=False)

if len(feature_success) > 0:
    top_features = feature_success.head(10)
    feature_names = [f.replace('_mean', '').replace('_', ' ').title()[:15] + '...' 
                    for f in top_features.index]
    
    ax.barh(range(len(top_features)), top_features[('R2', 'mean')], 
           color='lightblue', edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Mean R²')
    ax.set_title('Top 10 Features by Fit Quality')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(fig_dir / 'hill_parameter_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All visualizations saved to: {fig_dir}/")

# Save results
print(f"\n💾 Saving results...")
hill_params_df.to_parquet(results_dir / 'dose_response_hill_parameters.parquet', index=False)

# Create drug-level summary with key parameters
drug_summary = []
for drug in drugs:
    drug_params = hill_params_df[hill_params_df['drug'] == drug]
    
    if len(drug_params) > 0:
        summary = {
            'drug': drug,
            'n_successful_fits': len(drug_params),
            'n_high_quality_fits': (drug_params['R2'] > 0.7).sum(),
            'mean_R2': drug_params['R2'].mean(),
            'median_log_EC50': drug_params['log_EC50'].median(),
            'median_EC50': 10**drug_params['log_EC50'].median() if not drug_params['log_EC50'].isna().all() else np.nan,
            'median_Emax': drug_params['Emax'].median(),
            'median_n': drug_params['n'].median()
        }
        
        # Add top features
        top_3_features = drug_params.nlargest(3, 'R2')[['feature', 'R2', 'EC50']].to_dict('records')
        for i, feat in enumerate(top_3_features):
            summary[f'top_feature_{i+1}'] = feat['feature']
            summary[f'top_feature_{i+1}_R2'] = feat['R2']
            summary[f'top_feature_{i+1}_EC50'] = feat['EC50']
        
        drug_summary.append(summary)

drug_summary_df = pd.DataFrame(drug_summary)
drug_summary_df.to_parquet(results_dir / 'dose_response_drug_summary.parquet', index=False)

print(f"   Hill parameters: {results_dir / 'dose_response_hill_parameters.parquet'}")
print(f"   Drug summary: {results_dir / 'dose_response_drug_summary.parquet'}")

print(f"\n✅ Dose-response normalization complete!")
print(f"   Successfully fit {len(hill_params_df)} dose-response curves")
print(f"   High-quality fits (R² > 0.7): {(hill_params_df['R2'] > 0.7).sum()}")
print(f"   Drugs with parameters: {drug_summary_df['drug'].nunique()}")

# Create feature matrix with Hill parameters
print(f"\n🔄 Creating Hill parameter feature matrix...")

# Pivot to create drug x parameter matrix
hill_features = []

for drug in drugs:
    drug_params = hill_params_df[hill_params_df['drug'] == drug]
    
    if len(drug_params) == 0:
        continue
    
    drug_features = {'drug': drug}
    
    # For each feature, add its Hill parameters
    for _, param_row in drug_params.iterrows():
        feature = param_row['feature']
        
        # Only include high-quality fits
        if param_row['R2'] > 0.5:
            drug_features[f'{feature}_EC50'] = param_row['EC50']
            drug_features[f'{feature}_log_EC50'] = param_row['log_EC50']
            drug_features[f'{feature}_Emax'] = param_row['Emax']
            drug_features[f'{feature}_E0'] = param_row['E0']
            drug_features[f'{feature}_n'] = param_row['n']
            drug_features[f'{feature}_R2'] = param_row['R2']
    
    hill_features.append(drug_features)

hill_features_df = pd.DataFrame(hill_features)
hill_features_df.to_parquet(results_dir / 'dose_response_feature_matrix.parquet', index=False)

print(f"   Created feature matrix: {len(hill_features_df)} drugs x {len(hill_features_df.columns)-1} parameters")
print(f"   Saved to: {results_dir / 'dose_response_feature_matrix.parquet'}")

# 6. Drug clustering (now that hill_features_df is created)
if len(hill_features_df) > 10:
    print(f"\n📊 Creating drug clustering visualization...")
    visualize_drug_clustering(hill_features_df, fig_dir / 'drug_clustering_hill_params.png')

print(f"\n🎯 Next steps:")
print(f"   1. Use EC50 values for cross-drug comparison")
print(f"   2. Correlate Hill parameters with DILI")
print(f"   3. Create embeddings from normalized parameters")
print(f"   4. Compare potency rankings within drug classes")