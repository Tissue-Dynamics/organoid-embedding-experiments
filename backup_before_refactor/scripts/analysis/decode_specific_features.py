#!/usr/bin/env python3
"""
Decode the specific features that correlate strongly.
What do Morgan bit 208, SAX symbol 173, MACCS key 28 actually represent?
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Try to import RDKit for molecular analysis
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available - limited chemical interpretation")


def load_drug_smiles():
    """Load SMILES data for structure analysis."""
    print("Loading drug SMILES data...")
    
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    attach_query = f"""
    ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
    user={parsed.username} password={parsed.password}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    conn.execute(attach_query)
    
    query = """
    SELECT 
        drug,
        smiles,
        molecular_weight,
        logp,
        binary_dili,
        hepatotoxicity_boxed_warning
    FROM supabase.public.drugs
    WHERE smiles IS NOT NULL AND smiles != ''
    """
    
    drugs_df = conn.execute(query).df()
    conn.close()
    
    print(f"  Loaded SMILES for {len(drugs_df)} drugs")
    return drugs_df


def load_analysis_data():
    """Load the correlation and embedding data."""
    print("Loading analysis data...")
    
    # Load correlations
    corr_file = project_root / "results" / "figures" / "feature_correlations" / "significant_pearson_correlations.csv"
    correlations = pd.read_csv(corr_file)
    
    # Load embeddings
    data_file = project_root / "results" / "figures" / "structural_comparison" / "structural_oxygen_correlations.joblib"
    data = joblib.load(data_file)
    
    structural_embeddings = data['structural_embeddings']
    oxygen_embeddings = data['oxygen_embeddings']
    common_drugs = data['common_drugs']
    
    print(f"  Loaded {len(correlations)} correlations, {len(common_drugs)} drugs")
    return correlations, structural_embeddings, oxygen_embeddings, common_drugs


def analyze_morgan_bit_208(drugs_df, structural_embeddings, common_drugs):
    """Analyze what Morgan bit 208 represents by looking at which drugs have it."""
    print("\nAnalyzing Morgan bit 208...")
    
    if not RDKIT_AVAILABLE:
        print("  RDKit not available - cannot analyze Morgan fingerprints")
        return None
    
    # Get Morgan fingerprints for all drugs
    morgan_data = structural_embeddings['morgan']
    
    # Find drugs with Morgan bit 208 active
    bit_208_values = morgan_data.iloc[:, 208]  # Note: this might be sampled, so check if 208 exists
    
    if bit_208_values.shape[0] == 0:
        print("  Morgan bit 208 not found in sampled data")
        return None
    
    # Get drugs where this bit is active (value > 0.5 after standardization)
    scaler = StandardScaler()
    morgan_scaled = scaler.fit_transform(morgan_data)
    bit_208_scaled = morgan_scaled[:, 208] if morgan_scaled.shape[1] > 208 else None
    
    if bit_208_scaled is None:
        print("  Morgan bit 208 not in current embedding")
        return None
    
    # Find drugs with high/low values for this bit
    high_threshold = np.percentile(bit_208_scaled, 75)
    low_threshold = np.percentile(bit_208_scaled, 25)
    
    high_drugs = [common_drugs[i] for i in range(len(common_drugs)) if bit_208_scaled[i] > high_threshold]
    low_drugs = [common_drugs[i] for i in range(len(common_drugs)) if bit_208_scaled[i] < low_threshold]
    
    print(f"  High Morgan bit 208 drugs ({len(high_drugs)}): {high_drugs[:10]}")
    print(f"  Low Morgan bit 208 drugs ({len(low_drugs)}): {low_drugs[:10]}")
    
    # Analyze SMILES for pattern
    high_smiles = drugs_df[drugs_df['drug'].isin(high_drugs)]['smiles'].tolist()
    low_smiles = drugs_df[drugs_df['drug'].isin(low_drugs)]['smiles'].tolist()
    
    # Look for common substructures
    common_substructures_high = find_common_substructures(high_smiles[:10])
    common_substructures_low = find_common_substructures(low_smiles[:10])
    
    return {
        'high_drugs': high_drugs[:10],
        'low_drugs': low_drugs[:10],
        'high_smiles': high_smiles[:10],
        'low_smiles': low_smiles[:10],
        'common_high': common_substructures_high,
        'common_low': common_substructures_low,
        'bit_values': bit_208_scaled
    }


def analyze_sax_symbol_173(oxygen_embeddings, common_drugs):
    """Analyze what SAX symbol 173 represents in terms of oxygen patterns."""
    print("\nAnalyzing SAX symbol 173...")
    
    # Get SAX data
    sax_data = oxygen_embeddings['sax']
    
    if sax_data.shape[1] <= 173:
        print(f"  SAX symbol 173 not found (only {sax_data.shape[1]} features)")
        return None
    
    # Get values for SAX symbol 173
    symbol_173_values = sax_data.iloc[:, 173]
    
    # Standardize
    scaler = StandardScaler()
    sax_scaled = scaler.fit_transform(sax_data)
    symbol_173_scaled = sax_scaled[:, 173]
    
    # Find drugs with high/low values for this symbol
    high_threshold = np.percentile(symbol_173_scaled, 75)
    low_threshold = np.percentile(symbol_173_scaled, 25)
    
    high_drugs = [common_drugs[i] for i in range(len(common_drugs)) if symbol_173_scaled[i] > high_threshold]
    low_drugs = [common_drugs[i] for i in range(len(common_drugs)) if symbol_173_scaled[i] < low_threshold]
    
    print(f"  High SAX symbol 173 drugs ({len(high_drugs)}): {high_drugs[:10]}")
    print(f"  Low SAX symbol 173 drugs ({len(low_drugs)}): {low_drugs[:10]}")
    
    return {
        'high_drugs': high_drugs[:10],
        'low_drugs': low_drugs[:10],
        'symbol_values': symbol_173_scaled,
        'raw_values': symbol_173_values.values
    }


def analyze_maccs_key_28(drugs_df, structural_embeddings, common_drugs):
    """Analyze what MACCS key 28 represents."""
    print("\nAnalyzing MACCS key 28...")
    
    if not RDKIT_AVAILABLE:
        print("  RDKit not available - cannot analyze MACCS keys")
        return None
    
    # Get MACCS data
    maccs_data = structural_embeddings['maccs']
    
    if maccs_data.shape[1] <= 28:
        print(f"  MACCS key 28 not found (only {maccs_data.shape[1]} features)")
        return None
    
    # Get values for MACCS key 28
    key_28_values = maccs_data.iloc[:, 28]
    
    # MACCS keys are binary, so find drugs with this key present
    drugs_with_key = [common_drugs[i] for i in range(len(common_drugs)) if key_28_values.iloc[i] > 0.5]
    drugs_without_key = [common_drugs[i] for i in range(len(common_drugs)) if key_28_values.iloc[i] <= 0.5]
    
    print(f"  Drugs WITH MACCS key 28 ({len(drugs_with_key)}): {drugs_with_key[:10]}")
    print(f"  Drugs WITHOUT MACCS key 28 ({len(drugs_without_key)}): {drugs_without_key[:10]}")
    
    # Analyze what this key represents by looking at structures
    with_smiles = drugs_df[drugs_df['drug'].isin(drugs_with_key)]['smiles'].tolist()
    without_smiles = drugs_df[drugs_df['drug'].isin(drugs_without_key)]['smiles'].tolist()
    
    # Look for patterns that distinguish the two groups
    with_patterns = analyze_structural_patterns(with_smiles[:10])
    without_patterns = analyze_structural_patterns(without_smiles[:10])
    
    return {
        'drugs_with_key': drugs_with_key[:10],
        'drugs_without_key': drugs_without_key[:10],
        'with_smiles': with_smiles[:10],
        'without_smiles': without_smiles[:10],
        'with_patterns': with_patterns,
        'without_patterns': without_patterns
    }


def find_common_substructures(smiles_list):
    """Find common substructures in a list of SMILES."""
    if not RDKIT_AVAILABLE or len(smiles_list) < 2:
        return []
    
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
    
    if len(mols) < 2:
        return []
    
    # Simple approach: find common fragments
    common_fragments = []
    
    # Get fragments from first molecule
    if len(mols) > 0:
        first_mol = mols[0]
        # Look for rings
        ring_info = first_mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            common_fragments.append(f"Contains {ring_info.NumRings()} ring(s)")
        
        # Look for aromatic atoms
        aromatic_atoms = sum(1 for atom in first_mol.GetAtoms() if atom.GetIsAromatic())
        if aromatic_atoms > 0:
            common_fragments.append(f"Contains {aromatic_atoms} aromatic atoms")
    
    return common_fragments


def analyze_structural_patterns(smiles_list):
    """Analyze structural patterns in SMILES list."""
    if not RDKIT_AVAILABLE:
        return {}
    
    patterns = {
        'avg_mw': 0,
        'avg_logp': 0,
        'ring_counts': [],
        'aromatic_counts': [],
        'heteroatom_counts': []
    }
    
    valid_mols = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            patterns['avg_mw'] += Descriptors.MolWt(mol)
            patterns['avg_logp'] += Descriptors.MolLogP(mol)
            patterns['ring_counts'].append(rdMolDescriptors.CalcNumRings(mol))
            patterns['aromatic_counts'].append(Descriptors.NumAromaticRings(mol))
            patterns['heteroatom_counts'].append(rdMolDescriptors.CalcNumHeteroatoms(mol))
            valid_mols += 1
    
    if valid_mols > 0:
        patterns['avg_mw'] /= valid_mols
        patterns['avg_logp'] /= valid_mols
    
    return patterns


def create_feature_analysis_visualization(morgan_analysis, sax_analysis, maccs_analysis, output_dir):
    """Create visualization of feature analysis."""
    print("\nCreating feature analysis visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Morgan bit 208 analysis
    if morgan_analysis:
        ax1 = axes[0, 0]
        if morgan_analysis['bit_values'] is not None:
            ax1.hist(morgan_analysis['bit_values'], bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title('Morgan Bit 208 Distribution')
            ax1.set_xlabel('Standardized Value')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
        
        # Show drug examples
        ax2 = axes[1, 0]
        ax2.axis('off')
        if morgan_analysis['high_drugs']:
            text = f"HIGH Morgan Bit 208:\n" + "\n".join(morgan_analysis['high_drugs'][:5])
            text += f"\n\nLOW Morgan Bit 208:\n" + "\n".join(morgan_analysis['low_drugs'][:5])
            ax2.text(0.1, 0.9, text, transform=ax2.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    # SAX symbol 173 analysis
    if sax_analysis:
        ax3 = axes[0, 1]
        if sax_analysis['symbol_values'] is not None:
            ax3.hist(sax_analysis['symbol_values'], bins=30, alpha=0.7, edgecolor='black', color='orange')
            ax3.set_title('SAX Symbol 173 Distribution')
            ax3.set_xlabel('Standardized Value')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
        
        # Show drug examples
        ax4 = axes[1, 1]
        ax4.axis('off')
        if sax_analysis['high_drugs']:
            text = f"HIGH SAX Symbol 173:\n" + "\n".join(sax_analysis['high_drugs'][:5])
            text += f"\n\nLOW SAX Symbol 173:\n" + "\n".join(sax_analysis['low_drugs'][:5])
            ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.3))
    
    # MACCS key 28 analysis
    if maccs_analysis:
        ax5 = axes[0, 2]
        # MACCS is binary
        with_key = len(maccs_analysis['drugs_with_key'])
        without_key = len(maccs_analysis['drugs_without_key'])
        ax5.bar(['With Key 28', 'Without Key 28'], [with_key, without_key], 
                color=['green', 'red'], alpha=0.7)
        ax5.set_title('MACCS Key 28 Presence')
        ax5.set_ylabel('Number of Drugs')
        ax5.grid(True, alpha=0.3)
        
        # Show drug examples
        ax6 = axes[1, 2]
        ax6.axis('off')
        if maccs_analysis['drugs_with_key']:
            text = f"WITH MACCS Key 28:\n" + "\n".join(maccs_analysis['drugs_with_key'][:5])
            text += f"\n\nWITHOUT MACCS Key 28:\n" + "\n".join(maccs_analysis['drugs_without_key'][:5])
            ax6.text(0.1, 0.9, text, transform=ax6.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
    
    plt.suptitle('Analysis of Specific Correlated Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'specific_feature_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def analyze_correlation_mechanism(morgan_analysis, sax_analysis, structural_embeddings, oxygen_embeddings, common_drugs):
    """Analyze the mechanism behind the correlation."""
    print("\nAnalyzing correlation mechanism...")
    
    if not morgan_analysis or not sax_analysis:
        print("  Cannot analyze - missing data")
        return None
    
    # Get the actual feature values
    morgan_data = structural_embeddings['combined']  # Since the correlation was with combined
    sax_data = oxygen_embeddings['sax']
    
    # Check if we have the right features
    if morgan_data.shape[1] <= 208 or sax_data.shape[1] <= 173:
        print("  Features not available in current embedding")
        return None
    
    # Get feature values
    morgan_feature = StandardScaler().fit_transform(morgan_data)[:, 208]
    sax_feature = StandardScaler().fit_transform(sax_data)[:, 173]
    
    # Correlation analysis
    correlation = np.corrcoef(morgan_feature, sax_feature)[0, 1]
    
    # Find drugs in different quadrants
    high_both = []
    low_both = []
    high_morgan_low_sax = []
    low_morgan_high_sax = []
    
    for i, drug in enumerate(common_drugs):
        m_val = morgan_feature[i]
        s_val = sax_feature[i]
        
        if m_val > 0 and s_val > 0:
            high_both.append(drug)
        elif m_val < 0 and s_val < 0:
            low_both.append(drug)
        elif m_val > 0 and s_val < 0:
            high_morgan_low_sax.append(drug)
        elif m_val < 0 and s_val > 0:
            low_morgan_high_sax.append(drug)
    
    return {
        'correlation': correlation,
        'high_both': high_both[:5],
        'low_both': low_both[:5],
        'high_morgan_low_sax': high_morgan_low_sax[:5],
        'low_morgan_high_sax': low_morgan_high_sax[:5],
        'morgan_values': morgan_feature,
        'sax_values': sax_feature
    }


def main():
    """Main analysis pipeline."""
    print("=== Decoding Specific Correlated Features ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "specific_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    drugs_df = load_drug_smiles()
    correlations, structural_embeddings, oxygen_embeddings, common_drugs = load_analysis_data()
    
    # Analyze specific features
    morgan_analysis = analyze_morgan_bit_208(drugs_df, structural_embeddings, common_drugs)
    sax_analysis = analyze_sax_symbol_173(oxygen_embeddings, common_drugs)
    maccs_analysis = analyze_maccs_key_28(drugs_df, structural_embeddings, common_drugs)
    
    # Analyze correlation mechanism
    mechanism = analyze_correlation_mechanism(morgan_analysis, sax_analysis, structural_embeddings, oxygen_embeddings, common_drugs)
    
    # Create visualization
    viz_path = create_feature_analysis_visualization(morgan_analysis, sax_analysis, maccs_analysis, output_dir)
    
    # Save results
    results = {
        'morgan_analysis': morgan_analysis,
        'sax_analysis': sax_analysis,
        'maccs_analysis': maccs_analysis,
        'correlation_mechanism': mechanism
    }
    
    results_path = output_dir / 'specific_feature_analysis.joblib'
    joblib.dump(results, results_path)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Print key findings
    if morgan_analysis and morgan_analysis['high_drugs']:
        print(f"\n🧪 Morgan Bit 208 (High): {morgan_analysis['high_drugs'][:3]}")
        print(f"🧪 Morgan Bit 208 (Low): {morgan_analysis['low_drugs'][:3]}")
    
    if sax_analysis and sax_analysis['high_drugs']:
        print(f"\n📊 SAX Symbol 173 (High): {sax_analysis['high_drugs'][:3]}")
        print(f"📊 SAX Symbol 173 (Low): {sax_analysis['low_drugs'][:3]}")
    
    if mechanism:
        print(f"\n🔗 Correlation: {mechanism['correlation']:.3f}")
        print(f"🔗 High both: {mechanism['high_both'][:3]}")
        print(f"🔗 Low both: {mechanism['low_both'][:3]}")


if __name__ == "__main__":
    main()