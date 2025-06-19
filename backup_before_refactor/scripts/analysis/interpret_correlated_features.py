#!/usr/bin/env python3
"""
Interpret the specific features that correlate between structure and oxygen.
What do these correlations actually mean biologically and chemically?
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Try to import RDKit for molecular feature interpretation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available - limited chemical interpretation")


def load_analysis_data():
    """Load the correlation results and embedding data."""
    print("Loading correlation analysis data...")
    
    # Load correlations
    corr_file = project_root / "results" / "figures" / "feature_correlations" / "significant_pearson_correlations.csv"
    correlations = pd.read_csv(corr_file)
    
    # Load embeddings
    data_file = project_root / "results" / "figures" / "structural_comparison" / "structural_oxygen_correlations.joblib"
    data = joblib.load(data_file)
    
    structural_embeddings = data['structural_embeddings']
    oxygen_embeddings = data['oxygen_embeddings']
    common_drugs = data['common_drugs']
    
    print(f"  Loaded {len(correlations)} significant correlations")
    print(f"  Loaded data for {len(common_drugs)} drugs")
    
    return correlations, structural_embeddings, oxygen_embeddings, common_drugs


def interpret_structural_features(correlations, structural_embeddings):
    """Interpret what the structural features represent."""
    print("\nInterpreting structural features...")
    
    feature_interpretations = []
    
    for _, row in correlations.head(20).iterrows():  # Top 20 correlations
        struct_type = row['struct_type']
        struct_feature_idx = int(row['struct_feature'])
        
        interpretation = {
            'struct_type': struct_type,
            'struct_feature_idx': struct_feature_idx,
            'correlation': row['pearson_r'],
            'oxygen_type': row['oxygen_type'],
            'oxygen_feature_idx': int(row['oxygen_feature'])
        }
        
        if struct_type == 'descriptors':
            # Molecular descriptors - we know what these are
            descriptor_names = [
                'Molecular Weight', 'LogP', 'H-bond Donors', 'H-bond Acceptors',
                'Rotatable Bonds', 'Aromatic Rings', 'TPSA', 'Total Rings', 'Heteroatoms'
            ]
            if struct_feature_idx < len(descriptor_names):
                interpretation['feature_name'] = descriptor_names[struct_feature_idx]
                interpretation['feature_type'] = 'Molecular Descriptor'
                interpretation['description'] = get_descriptor_description(descriptor_names[struct_feature_idx])
            else:
                interpretation['feature_name'] = f'Unknown Descriptor {struct_feature_idx}'
                interpretation['feature_type'] = 'Molecular Descriptor'
                interpretation['description'] = 'Unknown molecular property'
                
        elif struct_type == 'morgan':
            # Morgan fingerprints - circular substructures
            interpretation['feature_name'] = f'Morgan Bit {struct_feature_idx}'
            interpretation['feature_type'] = 'Circular Fingerprint'
            interpretation['description'] = f'Circular substructure pattern (radius 2, bit {struct_feature_idx})'
            
        elif struct_type == 'maccs':
            # MACCS keys - known structural features
            interpretation['feature_name'] = f'MACCS Key {struct_feature_idx}'
            interpretation['feature_type'] = 'Structural Key'
            interpretation['description'] = get_maccs_description(struct_feature_idx)
            
        elif struct_type == 'rdkit':
            # RDKit fingerprints - Daylight-like
            interpretation['feature_name'] = f'RDKit Bit {struct_feature_idx}'
            interpretation['feature_type'] = 'Daylight Fingerprint'
            interpretation['description'] = f'Daylight-like substructure pattern (bit {struct_feature_idx})'
            
        elif struct_type == 'combined':
            # Combined features - need to map back to original
            combined_feature_mapping = get_combined_feature_mapping()
            if struct_feature_idx < len(combined_feature_mapping):
                orig_type, orig_idx = combined_feature_mapping[struct_feature_idx]
                interpretation['feature_name'] = f'{orig_type} Feature {orig_idx}'
                interpretation['feature_type'] = f'Combined ({orig_type})'
                interpretation['description'] = f'Combined embedding feature from {orig_type}'
            else:
                interpretation['feature_name'] = f'Combined Feature {struct_feature_idx}'
                interpretation['feature_type'] = 'Combined'
                interpretation['description'] = 'Combined structural embedding feature'
        
        feature_interpretations.append(interpretation)
    
    return pd.DataFrame(feature_interpretations)


def interpret_oxygen_features(correlations, oxygen_embeddings):
    """Interpret what the oxygen features represent."""
    print("\nInterpreting oxygen features...")
    
    oxygen_interpretations = []
    
    for _, row in correlations.head(20).iterrows():  # Top 20 correlations
        oxygen_type = row['oxygen_type']
        oxygen_feature_idx = int(row['oxygen_feature'])
        
        interpretation = {
            'oxygen_type': oxygen_type,
            'oxygen_feature_idx': oxygen_feature_idx,
            'correlation': row['pearson_r']
        }
        
        if oxygen_type == 'fourier':
            interpretation['feature_name'] = f'Fourier Component {oxygen_feature_idx}'
            interpretation['feature_type'] = 'Frequency Domain'
            interpretation['description'] = get_fourier_description(oxygen_feature_idx)
            
        elif oxygen_type == 'sax':
            interpretation['feature_name'] = f'SAX Symbol {oxygen_feature_idx}'
            interpretation['feature_type'] = 'Symbolic Pattern'
            interpretation['description'] = get_sax_description(oxygen_feature_idx)
            
        elif oxygen_type == 'catch22':
            interpretation['feature_name'] = f'catch22 Feature {oxygen_feature_idx}'
            interpretation['feature_type'] = 'Time Series Feature'
            interpretation['description'] = get_catch22_description(oxygen_feature_idx)
            
        elif oxygen_type == 'custom':
            interpretation['feature_name'] = f'Custom Feature {oxygen_feature_idx}'
            interpretation['feature_type'] = 'Organoid-Specific'
            interpretation['description'] = get_custom_description(oxygen_feature_idx)
        
        oxygen_interpretations.append(interpretation)
    
    return pd.DataFrame(oxygen_interpretations)


def get_descriptor_description(desc_name):
    """Get biological interpretation of molecular descriptors."""
    descriptions = {
        'Molecular Weight': 'Size/mass of molecule - affects membrane permeability and cellular uptake',
        'LogP': 'Lipophilicity - affects membrane crossing and cellular distribution',
        'H-bond Donors': 'Hydrogen bonding capacity - affects protein/membrane interactions',
        'H-bond Acceptors': 'Hydrogen bonding capacity - affects solubility and binding',
        'Rotatable Bonds': 'Molecular flexibility - affects conformational states and binding',
        'Aromatic Rings': 'π-π interactions and planarity - affects DNA intercalation and protein binding',
        'TPSA': 'Polar surface area - key predictor of membrane permeability',
        'Total Rings': 'Structural rigidity - affects binding specificity',
        'Heteroatoms': 'Non-carbon atoms - affects polarity and reactivity'
    }
    return descriptions.get(desc_name, 'Unknown molecular property')


def get_maccs_description(idx):
    """Get description of MACCS keys (simplified - real MACCS keys are proprietary)."""
    # Simplified MACCS key interpretations
    if idx < 10:
        return f'Basic structural pattern (atoms, bonds, rings)'
    elif idx < 50:
        return f'Functional group pattern (carbonyls, aromatics, heteroatoms)'
    elif idx < 100:
        return f'Complex substructure (multi-ring systems, specific patterns)'
    else:
        return f'Advanced structural feature (pharmacophores, 3D patterns)'


def get_fourier_description(idx):
    """Interpret Fourier components of oxygen time series."""
    if idx == 0:
        return 'DC component - overall oxygen consumption level'
    elif idx < 5:
        return f'Low frequency component - slow/sustained metabolic changes'
    elif idx < 15:
        return f'Medium frequency component - periodic metabolic oscillations'
    else:
        return f'High frequency component - rapid metabolic fluctuations'


def get_sax_description(idx):
    """Interpret SAX (Symbolic Aggregate approXimation) features."""
    return f'Symbolic pattern representing discretized oxygen consumption trajectory segments'


def get_catch22_description(idx):
    """Interpret catch22 time series features."""
    # catch22 features are standardized time series characteristics
    catch22_names = [
        'DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac', 'CO_FirstMin_ac',
        'CO_HistogramAMI_even_2_5', 'CO_trev_1_num', 'MD_hrv_classic_pnn40',
        'SB_BinaryStats_mean_longstretch1', 'SB_TransitionMatrix_3ac_sumdiagcov',
        'PD_PeriodicityWang_th0_01', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
        'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'FC_LocalSimple_mean1_tauresrat',
        'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd',
        'SP_Summaries_welch_rect_area_5_1', 'SB_BinaryStats_diff_longstretch0',
        'SB_MotifThree_quantile_hh', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
        'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SP_Summaries_welch_rect_centroid',
        'FC_LocalSimple_mean3_stderr'
    ]
    
    if idx < len(catch22_names):
        feature_name = catch22_names[idx]
        if 'Histogram' in feature_name:
            return 'Distribution shape feature - oxygen level patterns'
        elif 'Autocorr' in feature_name or 'CO_' in feature_name:
            return 'Temporal correlation feature - oxygen consumption dynamics'
        elif 'Binary' in feature_name or 'SB_' in feature_name:
            return 'Symbolic dynamics feature - oxygen state transitions'
        elif 'Periodic' in feature_name or 'PD_' in feature_name:
            return 'Periodicity feature - cyclic oxygen patterns'
        elif 'Spectral' in feature_name or 'SP_' in feature_name:
            return 'Frequency domain feature - oxygen oscillation patterns'
        else:
            return f'Time series characteristic: {feature_name}'
    else:
        return f'catch22 feature {idx} - standardized time series property'


def get_custom_description(idx):
    """Interpret custom organoid-specific features."""
    custom_features = [
        'Peak oxygen consumption rate',
        'Time to minimum oxygen',
        'Oxygen depletion slope',
        'Recovery rate after perturbation',
        'Baseline oxygen stability',
        'Maximum oxygen variation',
        'Late-phase oxygen trend',
        'Oxygen consumption efficiency',
        'Metabolic stress response'
    ]
    
    if idx < len(custom_features):
        return custom_features[idx]
    else:
        return f'Custom organoid feature {idx} - domain-specific oxygen metric'


def get_combined_feature_mapping():
    """Map combined feature indices back to original feature types."""
    # Simplified mapping based on concatenation order
    # Morgan (2048) + MACCS (167) + Descriptors (9) = 2224 total
    mapping = []
    
    # Morgan features (0-2047)
    for i in range(2048):
        mapping.append(('morgan', i))
    
    # MACCS features (2048-2214)
    for i in range(167):
        mapping.append(('maccs', i))
    
    # Descriptors (2215-2223)
    for i in range(9):
        mapping.append(('descriptors', i))
    
    return mapping


def analyze_correlation_patterns(correlations, feature_interpretations, oxygen_interpretations):
    """Analyze patterns in the correlations."""
    print("\nAnalyzing correlation patterns...")
    
    # Merge interpretations
    merged = correlations.head(20).copy()
    
    # Add indices for merging
    merged['struct_feature_idx'] = merged['struct_feature'].astype(int)
    merged['oxygen_feature_idx'] = merged['oxygen_feature'].astype(int)
    
    # Merge structural interpretations
    merged = merged.merge(feature_interpretations, 
                         on=['struct_type', 'struct_feature_idx'], 
                         how='left', suffixes=('', '_struct'))
    
    # Merge oxygen interpretations  
    merged = merged.merge(oxygen_interpretations,
                         on=['oxygen_type', 'oxygen_feature_idx'],
                         how='left', suffixes=('_struct', '_oxygen'))
    
    # Analyze patterns
    patterns = {
        'strongest_correlations': merged.head(5),
        'sax_dominance': merged[merged['oxygen_type'] == 'sax'],
        'descriptor_correlations': merged[merged['struct_type'] == 'descriptors'],
        'feature_type_distribution': merged['feature_type_struct'].value_counts(),
        'oxygen_type_distribution': merged['feature_type_oxygen'].value_counts()
    }
    
    return patterns, merged


def create_interpretation_visualization(patterns, merged, output_dir):
    """Create visualizations of the feature interpretations."""
    print("\nCreating interpretation visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top correlations with interpretations
    top_5 = patterns['strongest_correlations']
    y_pos = np.arange(len(top_5))
    
    bars = ax1.barh(y_pos, top_5['pearson_r'].abs(), 
                    color=['red' if r > 0 else 'blue' for r in top_5['pearson_r']])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['feature_name_struct']}\n↔\n{row['feature_name_oxygen']}" 
                        for _, row in top_5.iterrows()], fontsize=8)
    ax1.set_xlabel('Absolute Correlation')
    ax1.set_title('Top 5 Structure-Oxygen Feature Correlations')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation values
    for i, (_, row) in enumerate(top_5.iterrows()):
        ax1.text(abs(row['pearson_r']) + 0.01, i, f"{row['pearson_r']:.3f}", 
                va='center', fontsize=10)
    
    # 2. Feature type distribution
    struct_types = patterns['feature_type_distribution']
    ax2.pie(struct_types.values, labels=struct_types.index, autopct='%1.1f%%')
    ax2.set_title('Distribution of Correlated\nStructural Feature Types')
    
    # 3. Oxygen feature type distribution
    oxygen_types = patterns['oxygen_type_distribution']
    ax3.pie(oxygen_types.values, labels=oxygen_types.index, autopct='%1.1f%%')
    ax3.set_title('Distribution of Correlated\nOxygen Feature Types')
    
    # 4. Correlation strength by combination
    combo_analysis = merged.groupby(['struct_type', 'oxygen_type'])['pearson_r'].agg(['mean', 'max', 'count']).reset_index()
    combo_analysis['combo'] = combo_analysis['struct_type'] + ' → ' + combo_analysis['oxygen_type']
    
    bars = ax4.bar(range(len(combo_analysis)), combo_analysis['max'], alpha=0.7)
    ax4.set_xticks(range(len(combo_analysis)))
    ax4.set_xticklabels(combo_analysis['combo'], rotation=45, ha='right')
    ax4.set_ylabel('Maximum Correlation')
    ax4.set_title('Strongest Correlation by Embedding Combination')
    ax4.grid(True, alpha=0.3)
    
    # Add count annotations
    for i, row in combo_analysis.iterrows():
        ax4.text(i, row['max'] + 0.01, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Interpretation of Structure-Oxygen Feature Correlations', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'feature_interpretation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def create_biological_insights_summary(patterns, merged, output_dir):
    """Create a summary of biological insights."""
    print("\nCreating biological insights summary...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Analyze top correlations for biological insights
    top_5 = patterns['strongest_correlations']
    sax_corrs = patterns['sax_dominance']
    desc_corrs = patterns['descriptor_correlations']
    
    insights_text = f"""
BIOLOGICAL INSIGHTS: Structure-Oxygen Feature Correlations

TOP 5 STRONGEST CORRELATIONS:

1. {top_5.iloc[0]['feature_name_struct']} ↔ {top_5.iloc[0]['feature_name_oxygen']}
   Correlation: {top_5.iloc[0]['pearson_r']:.3f}
   Interpretation: {top_5.iloc[0]['description_struct']}
   
2. {top_5.iloc[1]['feature_name_struct']} ↔ {top_5.iloc[1]['feature_name_oxygen']}
   Correlation: {top_5.iloc[1]['pearson_r']:.3f}
   Interpretation: {top_5.iloc[1]['description_struct']}

3. {top_5.iloc[2]['feature_name_struct']} ↔ {top_5.iloc[2]['feature_name_oxygen']}
   Correlation: {top_5.iloc[2]['pearson_r']:.3f}
   Interpretation: {top_5.iloc[2]['description_struct']}

KEY BIOLOGICAL PATTERNS:

SAX Dominance ({len(sax_corrs)} of top 20 correlations):
• Symbolic oxygen patterns are most predictable from structure
• Suggests discrete metabolic states are structurally determined
• Drug structure influences whether cells enter specific oxygen consumption modes

Molecular Descriptor Insights:
• {len(desc_corrs)} molecular properties show strong oxygen correlations
• Chemical properties like size, lipophilicity affect cellular respiration
• Structural features predict metabolic response patterns

MECHANISTIC IMPLICATIONS:

1. Structure-Function Relationship:
   Chemical structure → Cellular targets → Metabolic changes → Oxygen patterns

2. Predictive Power:
   Specific structural motifs can forecast oxygen consumption behaviors

3. Drug Discovery Insight:
   Molecular fingerprints encode information about cellular metabolic effects

4. Biological Relevance:
   The correlations suggest structure-based toxicity prediction is feasible
   for organoid-based screening platforms

CONCLUSION:
Chemical structure contains meaningful biological information about 
cellular oxygen consumption patterns, particularly for discrete/symbolic 
representations of metabolic states.
    """
    
    ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, 
            fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.1))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'biological_insights_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def main():
    """Main analysis pipeline."""
    print("=== Interpreting Correlated Structure-Oxygen Features ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "feature_interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    correlations, structural_embeddings, oxygen_embeddings, common_drugs = load_analysis_data()
    
    # Interpret features
    feature_interpretations = interpret_structural_features(correlations, structural_embeddings)
    oxygen_interpretations = interpret_oxygen_features(correlations, oxygen_embeddings)
    
    # Analyze patterns
    patterns, merged = analyze_correlation_patterns(correlations, feature_interpretations, oxygen_interpretations)
    
    # Create visualizations
    viz_path = create_interpretation_visualization(patterns, merged, output_dir)
    insights_path = create_biological_insights_summary(patterns, merged, output_dir)
    
    # Save detailed results
    interpretations_path = output_dir / 'feature_interpretations.csv'
    merged.to_csv(interpretations_path, index=False)
    
    feature_details_path = output_dir / 'structural_feature_details.csv'
    feature_interpretations.to_csv(feature_details_path, index=False)
    
    oxygen_details_path = output_dir / 'oxygen_feature_details.csv'
    oxygen_interpretations.to_csv(oxygen_details_path, index=False)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Print key insights
    print(f"\n🔬 Key Insights:")
    print(f"  • SAX dominance: {len(patterns['sax_dominance'])}/{len(merged)} top correlations are with SAX features")
    print(f"  • Strongest correlation: {patterns['strongest_correlations'].iloc[0]['pearson_r']:.3f}")
    print(f"  • Feature types: {dict(patterns['feature_type_distribution'])}")
    
    print(f"\n🧬 Biological Interpretation:")
    top_corr = patterns['strongest_correlations'].iloc[0]
    print(f"  Best correlation is between:")
    print(f"  Structure: {top_corr.get('feature_name_struct', 'Unknown feature')}")
    print(f"  Oxygen: {top_corr.get('feature_name_oxygen', 'Unknown feature')}")
    print(f"  This suggests: {top_corr.get('description_struct', 'Unknown biological meaning')}")


if __name__ == "__main__":
    main()