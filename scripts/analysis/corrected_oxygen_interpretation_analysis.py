#!/usr/bin/env python3
"""
CORRECTED Oxygen Interpretation Analysis

CRITICAL CORRECTION: The 'o2' column represents OXYGEN PRESENCE/CONCENTRATION in medium.
- Lower O2 values = Higher oxygen consumption (cells consuming more oxygen)
- Higher O2 values = Lower oxygen consumption (cells consuming less oxygen)

This changes EVERYTHING about toxicity interpretation!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy import stats

from src.utils.data_loader import DataLoader

def analyze_corrected_oxygen_interpretation():
    """Re-analyze oxygen data with correct interpretation."""
    print("🔬 CORRECTED OXYGEN INTERPRETATION ANALYSIS")
    print("=" * 70)
    print("🚨 CRITICAL CORRECTION:")
    print("  • O2 column = OXYGEN PRESENCE in medium (not consumption rate)")
    print("  • Lower O2 = Higher oxygen consumption (more metabolically active)")
    print("  • Higher O2 = Lower oxygen consumption (less metabolically active)")
    print("=" * 70)
    
    # Load data
    with DataLoader() as loader:
        oxygen_data = loader.load_oxygen_data()
    
    if oxygen_data.empty:
        print("❌ No oxygen data available")
        return
    
    # Load drug metadata for toxicity labels
    drug_targets = pd.read_csv('data/database/drug_rows.csv')
    
    # Filter for drugs with toxicity data
    target_drugs = drug_targets['drug'].tolist()
    data = oxygen_data[oxygen_data['drug'].isin(target_drugs)]
    
    # Take first 10 drugs for detailed analysis
    analysis_drugs = data['drug'].unique()[:10]
    data = data[data['drug'].isin(analysis_drugs)]
    
    print(f"\n📊 DATASET FOR CORRECTED ANALYSIS:")
    print(f"  • {len(analysis_drugs)} drugs analyzed")
    print(f"  • {len(data)} total measurements")
    
    # 1. Analyze oxygen presence patterns by concentration
    print(f"\n1️⃣ OXYGEN PRESENCE PATTERNS BY CONCENTRATION:")
    analyze_oxygen_by_concentration(data)
    
    # 2. Re-interpret "toxicity" with correct oxygen understanding
    print(f"\n2️⃣ CORRECTED TOXICITY INTERPRETATION:")
    reinterpret_toxicity_with_correct_oxygen(data, drug_targets)
    
    # 3. Control vs treatment comparison with correct interpretation
    print(f"\n3️⃣ CONTROL vs TREATMENT (CORRECTED):")
    analyze_control_vs_treatment_corrected(data)
    
    # 4. DILI correlation with corrected oxygen interpretation
    print(f"\n4️⃣ DILI CORRELATION (CORRECTED):")
    analyze_dili_correlation_corrected(data, drug_targets)
    
    # 5. Create corrected visualization
    create_corrected_oxygen_visualization(data, drug_targets)
    
    # 6. Implications for concentration-toxicity prediction
    print(f"\n5️⃣ IMPLICATIONS FOR TOXICITY PREDICTION:")
    discuss_corrected_implications()

def analyze_oxygen_by_concentration(data):
    """Analyze oxygen presence patterns across concentrations."""
    
    # For each drug, look at oxygen levels across concentrations
    concentration_analysis = []
    
    for drug in data['drug'].unique():
        drug_data = data[data['drug'] == drug]
        
        # Group by concentration and calculate statistics
        conc_stats = drug_data.groupby('concentration').agg({
            'o2': ['mean', 'std', 'count']
        }).round(3)
        
        for conc in conc_stats.index:
            concentration_analysis.append({
                'drug': drug,
                'concentration': conc,
                'o2_mean': conc_stats.loc[conc, ('o2', 'mean')],
                'o2_std': conc_stats.loc[conc, ('o2', 'std')],
                'n_measurements': conc_stats.loc[conc, ('o2', 'count')]
            })
    
    conc_df = pd.DataFrame(concentration_analysis)
    
    print(f"📊 OXYGEN PRESENCE ACROSS CONCENTRATIONS:")
    
    # Overall patterns
    overall_stats = conc_df.groupby('concentration')['o2_mean'].agg(['mean', 'std']).round(3)
    
    print(f"\nOverall oxygen presence by concentration:")
    for conc in sorted(overall_stats.index):
        mean_o2 = overall_stats.loc[conc, 'mean']
        std_o2 = overall_stats.loc[conc, 'std']
        
        # Interpret what this means for oxygen consumption
        if conc == 0:
            consumption_interpretation = "BASELINE consumption (control)"
        else:
            # Compare to control (concentration 0)
            if 0 in overall_stats.index:
                control_o2 = overall_stats.loc[0, 'mean']
                if mean_o2 < control_o2:
                    consumption_interpretation = "HIGHER consumption (more metabolically active)"
                elif mean_o2 > control_o2:
                    consumption_interpretation = "LOWER consumption (less metabolically active)"
                else:
                    consumption_interpretation = "SAME consumption as control"
            else:
                consumption_interpretation = "No control for comparison"
        
        print(f"  • Concentration {conc:.2e} M: O2 = {mean_o2:.3f} ± {std_o2:.3f}")
        print(f"    → {consumption_interpretation}")

def reinterpret_toxicity_with_correct_oxygen(data, drug_targets):
    """Re-interpret toxicity patterns with correct oxygen understanding."""
    
    print(f"🎯 CORRECTED TOXICITY INTERPRETATION:")
    print(f"  • Previous assumption: Lower oxygen = toxicity (WRONG)")
    print(f"  • Correct understanding: Higher oxygen presence = lower consumption = potential toxicity")
    print(f"  • Toxic cells may have REDUCED metabolic activity (higher O2 remaining)")
    
    # For each drug, compare control vs highest concentration
    toxicity_reanalysis = []
    
    for drug in data['drug'].unique():
        drug_data = data[data['drug'] == drug]
        
        # Get control (concentration = 0)
        control_data = drug_data[drug_data['concentration'] == 0]
        if control_data.empty:
            continue
        
        # Get highest concentration
        max_conc = drug_data['concentration'].max()
        high_conc_data = drug_data[drug_data['concentration'] == max_conc]
        
        if high_conc_data.empty:
            continue
        
        control_o2_mean = control_data['o2'].mean()
        high_conc_o2_mean = high_conc_data['o2'].mean()
        
        # CORRECTED interpretation
        o2_change = high_conc_o2_mean - control_o2_mean
        
        if o2_change > 0:
            interpretation = "DECREASED oxygen consumption (potential toxicity - cells less active)"
        elif o2_change < 0:
            interpretation = "INCREASED oxygen consumption (metabolic activation - cells more active)"
        else:
            interpretation = "NO CHANGE in oxygen consumption"
        
        # Get DILI category for this drug
        drug_info = drug_targets[drug_targets['drug'] == drug]
        dili_category = drug_info.iloc[0]['dili'] if not drug_info.empty else 'Unknown'
        
        toxicity_reanalysis.append({
            'drug': drug,
            'control_o2': control_o2_mean,
            'high_conc_o2': high_conc_o2_mean,
            'o2_change': o2_change,
            'o2_change_percent': (o2_change / control_o2_mean) * 100,
            'interpretation': interpretation,
            'dili_category': dili_category
        })
        
        print(f"\n{drug}:")
        print(f"  • Control O2: {control_o2_mean:.3f}")
        print(f"  • High dose O2: {high_conc_o2_mean:.3f}")
        print(f"  • Change: {o2_change:+.3f} ({(o2_change / control_o2_mean) * 100:+.1f}%)")
        print(f"  • DILI category: {dili_category}")
        print(f"  • Interpretation: {interpretation}")
    
    return toxicity_reanalysis

def analyze_control_vs_treatment_corrected(data):
    """Analyze control vs treatment with correct oxygen interpretation."""
    
    # Compare concentration = 0 vs all other concentrations
    control_data = data[data['concentration'] == 0]
    treatment_data = data[data['concentration'] > 0]
    
    if control_data.empty:
        print("❌ No control data (concentration = 0) available")
        return
    
    control_o2_mean = control_data['o2'].mean()
    control_o2_std = control_data['o2'].std()
    
    treatment_o2_mean = treatment_data['o2'].mean()
    treatment_o2_std = treatment_data['o2'].std()
    
    print(f"📊 CONTROL vs TREATMENT COMPARISON (CORRECTED):")
    print(f"  • Control (conc=0): O2 = {control_o2_mean:.3f} ± {control_o2_std:.3f}")
    print(f"  • Treatment (conc>0): O2 = {treatment_o2_mean:.3f} ± {treatment_o2_std:.3f}")
    
    o2_change = treatment_o2_mean - control_o2_mean
    percent_change = (o2_change / control_o2_mean) * 100
    
    print(f"  • Change: {o2_change:+.3f} ({percent_change:+.1f}%)")
    
    # Correct interpretation
    if o2_change > 0:
        print(f"  • CORRECTED INTERPRETATION: Drug treatment DECREASES oxygen consumption")
        print(f"    → Cells become LESS metabolically active (potential toxicity/stress)")
    elif o2_change < 0:
        print(f"  • CORRECTED INTERPRETATION: Drug treatment INCREASES oxygen consumption")
        print(f"    → Cells become MORE metabolically active (stimulation/activation)")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(control_data['o2'], treatment_data['o2'])
    print(f"  • Statistical significance: t={t_stat:.3f}, p={p_value:.4f}")

def analyze_dili_correlation_corrected(data, drug_targets):
    """Analyze DILI correlation with corrected oxygen interpretation."""
    
    # Merge data with DILI categories
    merged_data = []
    
    for _, row in data.iterrows():
        drug = row['drug']
        drug_info = drug_targets[drug_targets['drug'] == drug]
        
        if not drug_info.empty:
            dili_category = drug_info.iloc[0]['dili']
            merged_data.append({
                'drug': drug,
                'concentration': row['concentration'],
                'o2': row['o2'],
                'dili_category': dili_category
            })
    
    if not merged_data:
        print("❌ No DILI data available for correlation")
        return
    
    merged_df = pd.DataFrame(merged_data)
    
    # Group by DILI category
    dili_o2_analysis = merged_df.groupby('dili_category')['o2'].agg(['mean', 'std', 'count']).round(3)
    
    print(f"📊 OXYGEN PRESENCE by DILI CATEGORY (CORRECTED):")
    
    dili_order = ['vNo-DILI-Concern', 'Ambiguous DILI-concern', 'vLess-DILI-Concern', 'vMost-DILI-Concern']
    
    for dili_cat in dili_order:
        if dili_cat in dili_o2_analysis.index:
            mean_o2 = dili_o2_analysis.loc[dili_cat, 'mean']
            std_o2 = dili_o2_analysis.loc[dili_cat, 'std']
            count = dili_o2_analysis.loc[dili_cat, 'count']
            
            print(f"  • {dili_cat}: O2 = {mean_o2:.3f} ± {std_o2:.3f} (n={count})")
    
    # CORRECTED interpretation of patterns
    print(f"\n🎯 CORRECTED DILI-OXYGEN CORRELATION:")
    
    # Compare most vs no DILI concern
    if 'vMost-DILI-Concern' in dili_o2_analysis.index and 'vNo-DILI-Concern' in dili_o2_analysis.index:
        most_dili_o2 = dili_o2_analysis.loc['vMost-DILI-Concern', 'mean']
        no_dili_o2 = dili_o2_analysis.loc['vNo-DILI-Concern', 'mean']
        
        if most_dili_o2 > no_dili_o2:
            print(f"  • Most-DILI drugs have HIGHER O2 (lower consumption) than No-DILI drugs")
            print(f"  • This suggests hepatotoxic drugs REDUCE cellular metabolic activity")
            print(f"  • Interpretation: Toxicity manifests as metabolic suppression")
        elif most_dili_o2 < no_dili_o2:
            print(f"  • Most-DILI drugs have LOWER O2 (higher consumption) than No-DILI drugs")
            print(f"  • This suggests hepatotoxic drugs INCREASE cellular metabolic activity")
            print(f"  • Interpretation: Toxicity manifests as metabolic stress/hyperactivity")
        else:
            print(f"  • No clear difference in O2 levels between DILI categories")

def create_corrected_oxygen_visualization(data, drug_targets):
    """Create visualization with corrected oxygen interpretation."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔬 CORRECTED OXYGEN INTERPRETATION ANALYSIS', fontsize=16, fontweight='bold')
    
    # 1. Oxygen levels by concentration (example drug)
    ax1 = axes[0, 0]
    
    example_drug = data['drug'].iloc[0]
    drug_data = data[data['drug'] == example_drug]
    
    conc_stats = drug_data.groupby('concentration')['o2'].agg(['mean', 'std'])
    concentrations = conc_stats.index
    o2_means = conc_stats['mean']
    o2_stds = conc_stats['std']
    
    ax1.errorbar(concentrations, o2_means, yerr=o2_stds, marker='o', capsize=5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Concentration (M)')
    ax1.set_ylabel('O2 Presence (Higher = Less Consumption)')
    ax1.set_title(f'{example_drug}\nCORRECTED: O2 Presence vs Concentration', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add interpretation annotation
    if len(o2_means) >= 2:
        if o2_means.iloc[-1] > o2_means.iloc[0]:
            interpretation = "Higher dose → Higher O2 → LESS consumption"
        else:
            interpretation = "Higher dose → Lower O2 → MORE consumption"
        ax1.text(0.05, 0.95, interpretation, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 2. Control vs treatment comparison
    ax2 = axes[0, 1]
    
    control_data = data[data['concentration'] == 0]
    treatment_data = data[data['concentration'] > 0]
    
    if not control_data.empty and not treatment_data.empty:
        ax2.boxplot([control_data['o2'], treatment_data['o2']], 
                   labels=['Control\n(Conc=0)', 'Treatment\n(Conc>0)'])
        ax2.set_ylabel('O2 Presence')
        ax2.set_title('Control vs Treatment\n(CORRECTED Interpretation)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation
        control_mean = control_data['o2'].mean()
        treatment_mean = treatment_data['o2'].mean()
        
        if treatment_mean > control_mean:
            interp_text = "Treatment → Higher O2\n→ LESS consumption\n→ Potential toxicity"
        else:
            interp_text = "Treatment → Lower O2\n→ MORE consumption\n→ Metabolic activation"
        
        ax2.text(0.05, 0.95, interp_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 3. DILI category comparison
    ax3 = axes[0, 2]
    
    # Merge with DILI data
    dili_o2_data = {}
    for dili_cat in ['vNo-DILI-Concern', 'vMost-DILI-Concern']:
        dili_drugs = drug_targets[drug_targets['dili'] == dili_cat]['drug'].tolist()
        dili_data = data[data['drug'].isin(dili_drugs)]
        if not dili_data.empty:
            dili_o2_data[dili_cat] = dili_data['o2'].values
    
    if len(dili_o2_data) >= 2:
        ax3.boxplot(list(dili_o2_data.values()), 
                   labels=['No-DILI\nConcern', 'Most-DILI\nConcern'])
        ax3.set_ylabel('O2 Presence')
        ax3.set_title('DILI Categories\n(CORRECTED Interpretation)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Concentration-response curves for multiple drugs
    ax4 = axes[1, 0]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data['drug'].unique())))
    
    for i, drug in enumerate(data['drug'].unique()[:5]):  # First 5 drugs
        drug_data = data[data['drug'] == drug]
        conc_stats = drug_data.groupby('concentration')['o2'].mean()
        
        ax4.semilogx(conc_stats.index, conc_stats.values, 
                    marker='o', label=drug, color=colors[i])
    
    ax4.set_xlabel('Concentration (M)')
    ax4.set_ylabel('O2 Presence')
    ax4.set_title('Concentration-Response Curves\n(CORRECTED: Higher O2 = Less Consumption)', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution of O2 changes
    ax5 = axes[1, 1]
    
    # Calculate O2 change from control for each drug
    o2_changes = []
    for drug in data['drug'].unique():
        drug_data = data[data['drug'] == drug]
        control_data = drug_data[drug_data['concentration'] == 0]
        treatment_data = drug_data[drug_data['concentration'] > 0]
        
        if not control_data.empty and not treatment_data.empty:
            control_mean = control_data['o2'].mean()
            treatment_mean = treatment_data['o2'].mean()
            change = treatment_mean - control_mean
            o2_changes.append(change)
    
    if o2_changes:
        ax5.hist(o2_changes, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        ax5.set_xlabel('O2 Change from Control')
        ax5.set_ylabel('Number of Drugs')
        ax5.set_title('Distribution of O2 Changes\n(CORRECTED: +ve = Less Consumption)', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add interpretation zones
        ax5.axvspan(0, max(o2_changes), alpha=0.2, color='red', label='Reduced consumption')
        ax5.axvspan(min(o2_changes), 0, alpha=0.2, color='blue', label='Increased consumption')
    
    # 6. Summary text with corrected interpretation
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = """🔬 CORRECTED OXYGEN INTERPRETATION

🚨 CRITICAL CORRECTION:
• O2 column = OXYGEN PRESENCE in medium
• Lower O2 = Higher consumption (metabolically active)
• Higher O2 = Lower consumption (metabolically suppressed)

📊 KEY FINDINGS (CORRECTED):
• Most drugs appear to REDUCE oxygen consumption
• This suggests metabolic suppression, not activation
• Potential toxicity manifests as cellular inactivity

🎯 TOXICITY IMPLICATIONS:
• Toxic drugs may INCREASE O2 levels (reduce consumption)
• This represents cellular stress/death
• Metabolically active cells consume more O2

💡 PREVIOUS ANALYSIS ERRORS:
• We incorrectly assumed lower O2 = toxicity
• This led to wrong concentration-response interpretation
• Need to re-analyze ALL previous results

🔄 CORRECTED APPROACH:
• Higher O2 presence = Potential toxicity
• Lower O2 presence = Metabolic activation
• Control vs treatment comparisons reversed"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/figures/corrected_oxygen_interpretation.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ Corrected oxygen analysis saved to: {output_path}")

def discuss_corrected_implications():
    """Discuss implications of corrected oxygen interpretation."""
    
    print(f"💡 IMPLICATIONS FOR CONCENTRATION-TOXICITY PREDICTION:")
    print(f"=" * 60)
    
    print(f"\n🚨 MAJOR CORRECTION REQUIRED:")
    print(f"  • ALL previous analyses interpreted oxygen data BACKWARDS")
    print(f"  • We measured oxygen PRESENCE, not consumption rate")
    print(f"  • This completely changes toxicity interpretation")
    
    print(f"\n🔄 CORRECTED TOXICITY DETECTION:")
    print(f"  • PREVIOUS (WRONG): Lower O2 = toxicity")
    print(f"  • CORRECTED: Higher O2 = potential toxicity")
    print(f"  • Rationale: Toxic/dead cells consume LESS oxygen")
    
    print(f"\n📊 CORRECTED FEATURE ENGINEERING:")
    print(f"  • Oxygen INCREASE from control = potential toxicity")
    print(f"  • Oxygen DECREASE from control = metabolic activation")
    print(f"  • Thresholds need to be reversed")
    
    print(f"\n🎯 CORRECTED CONCENTRATION-RESPONSE:")
    print(f"  • Rising O2 with concentration = increasing toxicity")
    print(f"  • Falling O2 with concentration = increasing activation")
    print(f"  • Concentration where O2 starts rising = toxicity onset")
    
    print(f"\n⚠️  VALIDATION REQUIREMENTS:")
    print(f"  • Confirm higher O2 = reduced cell viability")
    print(f"  • Test with known toxic vs non-toxic drugs")
    print(f"  • Validate with orthogonal toxicity assays")
    
    print(f"\n🔧 IMMEDIATE NEXT STEPS:")
    print(f"  1. Re-run ALL concentration-toxicity analyses with corrected interpretation")
    print(f"  2. Flip all 'oxygen decline' calculations to 'oxygen increase'")
    print(f"  3. Validate corrected interpretation with known hepatotoxic drugs")
    print(f"  4. Update ALL visualizations and interpretations")
    print(f"  5. Correct the CLAUDE.md documentation")

def main():
    """Run corrected oxygen interpretation analysis."""
    analyze_corrected_oxygen_interpretation()

if __name__ == "__main__":
    main()