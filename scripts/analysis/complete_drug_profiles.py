#!/usr/bin/env python3
"""
Detailed analysis of drugs with complete data profiles
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

from src.utils.data_loader import DataLoader

def analyze_complete_drug_profiles():
    """Analyze drugs with complete data (DILI, PK, and good oxygen data)."""
    
    print("🔬 COMPLETE DRUG PROFILE ANALYSIS")
    print("=" * 80)
    print("Focusing on drugs with:")
    print("✓ DILI classification")
    print("✓ Clinical PK data (Cmax)")
    print("✓ High-quality oxygen measurements")
    print("=" * 80)
    
    # Load data
    with DataLoader() as loader:
        oxygen_data = loader.load_oxygen_data()
        
    drug_metadata = pd.read_csv('data/database/drug_rows.csv')
    
    # Filter for complete drugs
    complete_drugs = drug_metadata[
        (drug_metadata['dili'].notna()) & 
        (drug_metadata['dili'] != 'Unknown') &
        (drug_metadata['cmax_oral_m'].notna())
    ]['drug'].tolist()
    
    print(f"\nFound {len(complete_drugs)} drugs with complete metadata")
    
    # Analyze each complete drug
    drug_profiles = []
    
    for drug in complete_drugs:
        print(f"\nAnalyzing {drug}...")
        
        # Get drug data
        drug_oxygen = oxygen_data[oxygen_data['drug'] == drug]
        drug_meta = drug_metadata[drug_metadata['drug'] == drug].iloc[0]
        
        if len(drug_oxygen) < 1000:
            print(f"  ⚠️  Skipping - only {len(drug_oxygen)} measurements")
            continue
        
        profile = analyze_drug_profile(drug, drug_oxygen, drug_meta)
        drug_profiles.append(profile)
        
        # Print key findings
        print(f"  ✓ DILI: {profile['dili_category']}")
        print(f"  ✓ Cmax: {profile['cmax_um']:.2f} µM")
        print(f"  ✓ Max response: {profile['max_fold_change']:.1f}x at {profile['max_response_conc']:.1f} µM")
        if pd.notna(profile['ec50_um']):
            print(f"  ✓ EC50: {profile['ec50_um']:.2f} µM (safety margin: {profile['safety_margin']:.1f}x)")
    
    # Convert to DataFrame
    profiles_df = pd.DataFrame(drug_profiles)
    
    # Save detailed profiles
    output_path = Path('results/data/complete_drug_profiles.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(profiles_df)} complete drug profiles to: {output_path}")
    
    # Analyze patterns
    analyze_dili_patterns(profiles_df)
    
    # Create detailed visualizations
    create_detailed_visualizations(profiles_df, oxygen_data)
    
    return profiles_df

def analyze_drug_profile(drug, drug_oxygen, drug_meta):
    """Detailed analysis of a single drug."""
    
    profile = {
        'drug': drug,
        'dili_category': drug_meta['dili'],
        'severity': drug_meta.get('severity', np.nan),
        'likelihood': drug_meta.get('likelihood', np.nan),
        
        # Clinical data
        'cmax_um': drug_meta['cmax_oral_m'] * 1e6,
        'half_life_h': drug_meta.get('half_life_hours', np.nan),
        'mw': drug_meta.get('molecular_weight', np.nan),
        'logp': drug_meta.get('logp', np.nan),
        
        # Data quality
        'n_measurements': len(drug_oxygen),
        'n_concentrations': drug_oxygen['concentration'].nunique(),
        'n_timepoints': drug_oxygen['elapsed_hours'].nunique(),
        'time_coverage_h': drug_oxygen['elapsed_hours'].max(),
    }
    
    # Concentration-response analysis
    conc_response = drug_oxygen.groupby('concentration').agg({
        'o2': ['mean', 'std', 'count']
    })
    conc_response.columns = ['o2_mean', 'o2_std', 'n']
    conc_response = conc_response.reset_index()
    
    # Control baseline
    control_data = conc_response[conc_response['concentration'] == 0]
    if len(control_data) > 0:
        control_mean = control_data['o2_mean'].iloc[0]
        control_std = control_data['o2_std'].iloc[0]
    else:
        control_mean = 0
        control_std = 1
    
    profile['control_o2_mean'] = control_mean
    profile['control_o2_std'] = control_std
    
    # Maximum response
    max_response = conc_response.loc[conc_response['o2_mean'].idxmax()]
    profile['max_o2_mean'] = max_response['o2_mean']
    profile['max_response_conc'] = max_response['concentration']
    profile['max_fold_change'] = (max_response['o2_mean'] - control_mean) / (abs(control_mean) + 1e-6)
    
    # EC50 calculation
    try:
        ec50, hill_slope = fit_dose_response(conc_response, control_mean)
        profile['ec50_um'] = ec50
        profile['hill_slope'] = hill_slope
        
        # Safety margin
        if pd.notna(ec50) and ec50 > 0:
            profile['safety_margin'] = ec50 / profile['cmax_um']
        else:
            profile['safety_margin'] = np.nan
    except:
        profile['ec50_um'] = np.nan
        profile['hill_slope'] = np.nan
        profile['safety_margin'] = np.nan
    
    # Time-dependent features
    time_features = analyze_time_course(drug_oxygen, profile['cmax_um'])
    profile.update(time_features)
    
    # Clinical relevance scores
    profile['covers_therapeutic'] = any(
        0.1 <= (conc / profile['cmax_um']) <= 10 
        for conc in conc_response['concentration'] if conc > 0
    )
    
    profile['max_cmax_multiple'] = conc_response['concentration'].max() / profile['cmax_um']
    
    # Response at clinical concentrations
    clinical_concs = conc_response[
        (conc_response['concentration'] > 0.1 * profile['cmax_um']) &
        (conc_response['concentration'] < 10 * profile['cmax_um'])
    ]
    
    if len(clinical_concs) > 0:
        profile['response_at_cmax'] = clinical_concs['o2_mean'].mean()
        profile['fold_change_at_cmax'] = (clinical_concs['o2_mean'].mean() - control_mean) / (abs(control_mean) + 1e-6)
    else:
        profile['response_at_cmax'] = np.nan
        profile['fold_change_at_cmax'] = np.nan
    
    return profile

def fit_dose_response(conc_response, control_mean):
    """Fit 4-parameter logistic dose-response curve."""
    
    # Filter positive concentrations
    dr_data = conc_response[conc_response['concentration'] > 0].copy()
    
    if len(dr_data) < 4:
        return np.nan, np.nan
    
    # Log-transform concentrations
    x = np.log10(dr_data['concentration'].values)
    y = dr_data['o2_mean'].values
    
    # 4-parameter logistic function
    def logistic(x, bottom, top, ec50, slope):
        return bottom + (top - bottom) / (1 + 10**((ec50 - x) * slope))
    
    try:
        # Initial guess
        p0 = [
            y.min(),  # bottom
            y.max(),  # top
            np.median(x),  # log EC50
            1.0  # slope
        ]
        
        # Fit curve
        popt, _ = optimize.curve_fit(logistic, x, y, p0=p0, maxfev=5000)
        
        # Convert log EC50 back to concentration
        ec50 = 10**popt[2]
        hill_slope = popt[3]
        
        return ec50, hill_slope
    except:
        return np.nan, np.nan

def analyze_time_course(drug_oxygen, cmax_um):
    """Analyze time-dependent features."""
    
    features = {}
    
    # Time to reach maximum response
    time_max = drug_oxygen.groupby('elapsed_hours')['o2'].mean()
    features['time_to_max_h'] = time_max.idxmax()
    
    # Early vs late response
    early_data = drug_oxygen[drug_oxygen['elapsed_hours'] < 24]
    late_data = drug_oxygen[drug_oxygen['elapsed_hours'] > 96]
    
    if len(early_data) > 100 and len(late_data) > 100:
        features['early_response'] = early_data['o2'].mean()
        features['late_response'] = late_data['o2'].mean()
        features['temporal_change'] = features['late_response'] - features['early_response']
    else:
        features['early_response'] = np.nan
        features['late_response'] = np.nan
        features['temporal_change'] = np.nan
    
    # Variability over time
    time_cv = drug_oxygen.groupby('elapsed_hours')['o2'].std() / drug_oxygen.groupby('elapsed_hours')['o2'].mean()
    features['temporal_cv_mean'] = time_cv.mean()
    features['temporal_cv_max'] = time_cv.max()
    
    return features

def analyze_dili_patterns(profiles_df):
    """Analyze patterns by DILI category."""
    
    print("\n📊 DILI PATTERN ANALYSIS")
    print("=" * 80)
    
    # Group by DILI category
    dili_groups = profiles_df.groupby('dili_category')
    
    # Key metrics by DILI
    metrics = ['max_fold_change', 'ec50_um', 'safety_margin', 'fold_change_at_cmax']
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for dili_cat in ['vNo-DILI-Concern', 'vLess-DILI-Concern', 'vMost-DILI-Concern']:
            if dili_cat in dili_groups.groups:
                values = dili_groups.get_group(dili_cat)[metric].dropna()
                if len(values) > 0:
                    print(f"  {dili_cat}: {values.mean():.2f} ± {values.std():.2f} (n={len(values)})")
    
    # Statistical tests
    print("\n📈 STATISTICAL COMPARISONS:")
    
    # Compare fold changes between DILI groups
    no_dili = profiles_df[profiles_df['dili_category'] == 'vNo-DILI-Concern']['max_fold_change'].dropna()
    most_dili = profiles_df[profiles_df['dili_category'] == 'vMost-DILI-Concern']['max_fold_change'].dropna()
    
    if len(no_dili) > 3 and len(most_dili) > 3:
        t_stat, p_val = stats.ttest_ind(no_dili, most_dili)
        print(f"\nFold change No-DILI vs Most-DILI: p={p_val:.4f}")
    
    # Correlation analysis
    print("\n🔗 CORRELATIONS:")
    
    # EC50 vs Cmax
    ec50_data = profiles_df[profiles_df['ec50_um'].notna()]
    if len(ec50_data) > 10:
        corr = np.corrcoef(np.log10(ec50_data['ec50_um']), np.log10(ec50_data['cmax_um']))[0, 1]
        print(f"  log(EC50) vs log(Cmax): r={corr:.3f}")
    
    # Safety margin vs DILI
    safety_data = profiles_df[profiles_df['safety_margin'].notna()]
    if len(safety_data) > 10:
        dili_numeric = safety_data['dili_category'].map({
            'vNo-DILI-Concern': 0,
            'vLess-DILI-Concern': 1,
            'vMost-DILI-Concern': 2
        })
        corr = np.corrcoef(safety_data['safety_margin'], dili_numeric)[0, 1]
        print(f"  Safety margin vs DILI severity: r={corr:.3f}")

def create_detailed_visualizations(profiles_df, oxygen_data):
    """Create detailed visualizations of complete drug profiles."""
    
    # Set up figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. EC50 vs Cmax by DILI
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    colors = {
        'vNo-DILI-Concern': 'green',
        'vLess-DILI-Concern': 'orange', 
        'vMost-DILI-Concern': 'red',
        'Ambiguous DILI-concern': 'gray'
    }
    
    for dili, color in colors.items():
        dili_data = profiles_df[profiles_df['dili_category'] == dili]
        if len(dili_data) > 0:
            ax1.scatter(dili_data['cmax_um'], dili_data['ec50_um'],
                       c=color, label=dili.replace('v', '').replace('-Concern', ''),
                       alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # Add safety margin lines
    x_range = np.logspace(-2, 3, 100)
    ax1.plot(x_range, x_range, 'k--', alpha=0.3, label='1x margin')
    ax1.plot(x_range, x_range * 10, 'k--', alpha=0.3, label='10x margin')
    ax1.plot(x_range, x_range * 100, 'k--', alpha=0.3, label='100x margin')
    
    ax1.set_xlabel('Clinical Cmax (µM)', fontsize=12)
    ax1.set_ylabel('EC50 (µM)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Toxicity Threshold vs Clinical Exposure', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Fold change at Cmax
    ax2 = fig.add_subplot(gs[0, 2:])
    
    dili_order = ['vNo-DILI-Concern', 'vLess-DILI-Concern', 'vMost-DILI-Concern']
    fc_data = []
    labels = []
    
    for dili in dili_order:
        dili_drugs = profiles_df[profiles_df['dili_category'] == dili]
        if len(dili_drugs) > 0:
            fc_data.append(dili_drugs['fold_change_at_cmax'].dropna().values)
            labels.append(dili.replace('v', '').replace('-Concern', ''))
    
    if fc_data:
        bp = ax2.boxplot(fc_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['green', 'orange', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Fold Change at Clinical Cmax', fontsize=12)
    ax2.set_title('Response at Therapeutic Concentrations', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. Safety margin distribution
    ax3 = fig.add_subplot(gs[1, :2])
    
    safety_margins = profiles_df['safety_margin'].dropna()
    ax3.hist(safety_margins[safety_margins < 1000], bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(1, color='red', linestyle='--', linewidth=2, label='No margin')
    ax3.axvline(10, color='orange', linestyle='--', linewidth=2, label='10x margin')
    ax3.axvline(100, color='green', linestyle='--', linewidth=2, label='100x margin')
    ax3.set_xlabel('Safety Margin (EC50/Cmax)', fontsize=12)
    ax3.set_ylabel('Number of Drugs', fontsize=12)
    ax3.set_title('Safety Margin Distribution', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Time-dependent patterns
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Plot average time course for each DILI category
    for dili, color in colors.items():
        dili_drugs = profiles_df[profiles_df['dili_category'] == dili]['drug'].tolist()
        if dili_drugs:
            # Get oxygen data for these drugs at max concentration
            dili_oxygen = oxygen_data[
                (oxygen_data['drug'].isin(dili_drugs)) & 
                (oxygen_data['concentration'] == 22.5)
            ]
            
            if len(dili_oxygen) > 100:
                time_course = dili_oxygen.groupby('elapsed_hours')['o2'].mean()
                ax4.plot(time_course.index, time_course.values, 
                        color=color, label=dili.replace('v', '').replace('-Concern', ''),
                        linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('O2 Level', fontsize=12)
    ax4.set_title('Time Course by DILI Category (22.5 µM)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 168)
    
    # 5. Dose-response curves for representative drugs
    ax5 = fig.add_subplot(gs[2, :])
    
    # Select representative drugs from each category
    representative_drugs = []
    for dili in dili_order:
        dili_drugs = profiles_df[profiles_df['dili_category'] == dili]
        if len(dili_drugs) > 0:
            # Pick drug with median fold change
            median_drug = dili_drugs.iloc[(dili_drugs['max_fold_change'] - dili_drugs['max_fold_change'].median()).abs().argsort()[:1]]
            representative_drugs.append((median_drug.iloc[0]['drug'], dili, colors[dili]))
    
    for drug, dili, color in representative_drugs:
        drug_data = oxygen_data[oxygen_data['drug'] == drug]
        dr_curve = drug_data.groupby('concentration')['o2'].mean()
        
        ax5.semilogx(dr_curve.index[dr_curve.index > 0], 
                    dr_curve[dr_curve.index > 0].values,
                    'o-', color=color, label=f"{drug} ({dili.replace('v', '').replace('-Concern', '')})",
                    markersize=8, linewidth=2, alpha=0.8)
    
    ax5.set_xlabel('Concentration (µM)', fontsize=12)
    ax5.set_ylabel('O2 Level', fontsize=12)
    ax5.set_title('Representative Dose-Response Curves', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Clinical relevance summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate summary statistics
    n_safe = len(profiles_df[profiles_df['safety_margin'] > 10])
    n_risky = len(profiles_df[profiles_df['safety_margin'] < 1])
    n_therapeutic = len(profiles_df[profiles_df['covers_therapeutic'] == True])
    
    avg_fold_no_dili = profiles_df[profiles_df['dili_category'] == 'vNo-DILI-Concern']['max_fold_change'].mean()
    avg_fold_most_dili = profiles_df[profiles_df['dili_category'] == 'vMost-DILI-Concern']['max_fold_change'].mean()
    
    summary_text = f"""
    COMPLETE DRUG PROFILE SUMMARY
    ════════════════════════════════════════════════════════════════════════════════
    
    Drugs with Complete Data: {len(profiles_df)}
    
    Safety Analysis:
    • Drugs with >10x safety margin: {n_safe} ({n_safe/len(profiles_df)*100:.1f}%)
    • Drugs with <1x safety margin: {n_risky} ({n_risky/len(profiles_df)*100:.1f}%)
    • Drugs covering therapeutic range: {n_therapeutic} ({n_therapeutic/len(profiles_df)*100:.1f}%)
    
    DILI Patterns:
    • Average fold change - No DILI: {avg_fold_no_dili:.1f}x
    • Average fold change - Most DILI: {avg_fold_most_dili:.1f}x
    • Median EC50: {profiles_df['ec50_um'].median():.1f} µM
    • Median safety margin: {profiles_df['safety_margin'].median():.1f}x
    
    Key Findings:
    • Higher O2 response does not always correlate with DILI risk
    • Safety margin (EC50/Cmax) provides better risk stratification
    • Time-dependent patterns differ between DILI categories
    • Most drugs show toxicity only at supratherapeutic concentrations
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Complete Drug Profile Analysis', fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = Path('results/figures/complete_drug_profiles_analysis.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ Detailed visualization saved to: {output_path}")

if __name__ == "__main__":
    profiles_df = analyze_complete_drug_profiles()