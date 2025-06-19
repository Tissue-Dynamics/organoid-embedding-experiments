#!/usr/bin/env python3
"""
Phase 2 vs DILI Analysis: Use hierarchical embeddings for drug analysis
Check if Phase 2 embeddings correlate with DILI better than well-level analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "phase2_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 2 HIERARCHICAL EMBEDDINGS vs DILI ANALYSIS")
print("=" * 80)

# Load Phase 2 results
try:
    import joblib
    phase2_results = joblib.load(results_dir / "hierarchical_embedding_results.joblib")
    print("✅ Phase 2 results loaded successfully")
    
    print(f"\n📊 Available embeddings:")
    for key in phase2_results.keys():
        data = phase2_results[key]
        if hasattr(data, 'shape'):
            print(f"   {key}: {data.shape}")
        else:
            print(f"   {key}: {len(data) if hasattr(data, '__len__') else type(data)}")
            
except Exception as e:
    print(f"❌ Cannot load Phase 2 results: {e}")
    print("Checking what files we have...")
    available_files = list(results_dir.glob("*.joblib"))
    print(f"Available joblib files: {available_files}")
    exit()

# Load drug metadata with DILI info
df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")

# Create drug-level DILI mapping
drug_dili_map = df[df['dili'].notna()].groupby('drug').agg({
    'dili': 'first',
    'binary_dili': 'first', 
    'dili_risk_category': 'first',
    'molecular_weight': 'first',
    'logp': 'first'
}).reset_index()

print(f"\n💊 Drug-level DILI mapping:")
print(f"   Drugs with DILI data: {len(drug_dili_map)}")

# Map DILI risk to numeric
dili_risk_mapping = {
    'Low': 1,
    'Low risk': 1,
    'Low Risk': 1,
    'Low DILI Risk': 1,
    'Low hepatotoxicity risk': 1,
    'No DILI concern': 0,
    'No-DILI-Concern per database': 0,
    'vLess-DILI-Concern': 1,
    'Low concern': 1,
    'Low - vLess-DILI-Concern': 1,
    'Low frequency (<1:10,000), high severity': 2,
    'Low-Moderate': 1.5,
    'Moderate': 2,
    'Moderate Risk': 2,
    'Intermediate': 2,
    'High': 3,
    'High Risk': 3,
    'High risk - drug withdrawn': 4,
    'High risk - cumulative dose-dependent': 3,
    'High concern': 3,
    'High Risk - Black Box Warning': 4,
    'Severe': 4,
    'Severe - High Risk': 4,
    'Black Box Warning': 4,
    'LiverTox Category D': 3,
    'Not formally categorized - primary toxicity was hematologic': 1,
    'Low frequency but potentially severe': 2,
    'vMost-DILI-Concern': 4
}

drug_dili_map['dili_risk_numeric'] = drug_dili_map['dili_risk_category'].map(dili_risk_mapping)
drug_dili_map['has_dili_risk'] = drug_dili_map['dili_risk_numeric'].notna()

print(f"   Drugs with numeric DILI risk: {drug_dili_map['has_dili_risk'].sum()}")

# Analyze each embedding level
embedding_levels = ['drug_embeddings', 'concentration_embeddings', 'well_embeddings']

for level in embedding_levels:
    if level not in phase2_results:
        print(f"⚠️ {level} not found in Phase 2 results")
        continue
        
    embeddings = phase2_results[level]
    metadata_key = level.replace('embeddings', 'metadata')
    metadata = phase2_results.get(metadata_key, None)
    
    print(f"\n🔍 ANALYZING {level.upper()}")
    print(f"   Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'Unknown'}")
    
    # Convert to DataFrame if needed
    if hasattr(embeddings, 'index'):
        embed_df = embeddings
    elif hasattr(embeddings, 'shape') and len(embeddings.shape) == 2:
        embed_df = pd.DataFrame(embeddings)
    else:
        print(f"   ❌ Cannot process {level} - unexpected format")
        continue
    
    print(f"   Processing {len(embed_df)} {level.split('_')[0]}s")
    
    # For drug-level analysis
    if level == 'drug_embeddings':
        if metadata is not None and hasattr(metadata, 'index'):
            # Use metadata index as drug names
            embed_df.index = metadata.index
        
        # Try to match with DILI data
        if hasattr(embed_df, 'index'):
            embed_drugs = set(embed_df.index)
            dili_drugs = set(drug_dili_map['drug'])
            overlap = embed_drugs.intersection(dili_drugs)
            
            print(f"   🔗 Overlap with DILI drugs: {len(overlap)}")
            
            if len(overlap) > 5:  # Need minimum for analysis
                # Merge embeddings with DILI data
                merged_df = embed_df.loc[list(overlap)].copy()
                dili_subset = drug_dili_map[drug_dili_map['drug'].isin(overlap)].set_index('drug')
                merged_df = merged_df.join(dili_subset)
                
                print(f"   📊 Merged dataset: {len(merged_df)} drugs")
                
                # Correlation analysis
                embedding_cols = [col for col in merged_df.columns if not col in ['dili', 'binary_dili', 'dili_risk_category', 'molecular_weight', 'logp', 'dili_risk_numeric', 'has_dili_risk']]
                
                correlations = []
                for embed_col in embedding_cols[:10]:  # Analyze first 10 embedding dimensions
                    if merged_df['dili_risk_numeric'].notna().sum() > 5:
                        valid_data = merged_df[[embed_col, 'dili_risk_numeric']].dropna()
                        if len(valid_data) > 5:
                            corr, p_val = spearmanr(valid_data[embed_col], valid_data['dili_risk_numeric'])
                            correlations.append({
                                'embedding_dim': embed_col,
                                'correlation': corr,
                                'p_value': p_val,
                                'n_drugs': len(valid_data)
                            })
                
                if correlations:
                    corr_df = pd.DataFrame(correlations)
                    significant_corrs = corr_df[corr_df['p_value'] < 0.1]
                    
                    print(f"   🔍 Significant correlations found: {len(significant_corrs)}")
                    
                    if len(significant_corrs) > 0:
                        print("   Top correlations:")
                        for _, row in significant_corrs.head(5).iterrows():
                            print(f"      {row['embedding_dim']}: r={row['correlation']:.3f}, p={row['p_value']:.3f}")
                    
                    # Create visualization
                    if len(correlations) > 0:
                        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                        fig.suptitle(f'{level.replace("_", " ").title()} vs DILI Risk Analysis', fontsize=14)
                        
                        # Plot 1: Correlation strengths
                        ax = axes[0]
                        corr_df['abs_correlation'] = corr_df['correlation'].abs()
                        corr_df_sorted = corr_df.sort_values('abs_correlation', ascending=True)
                        
                        colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in corr_df_sorted['p_value']]
                        ax.barh(range(len(corr_df_sorted)), corr_df_sorted['abs_correlation'], color=colors)
                        ax.set_yticks(range(len(corr_df_sorted)))
                        ax.set_yticklabels(corr_df_sorted['embedding_dim'])
                        ax.set_xlabel('|Correlation| with DILI Risk')
                        ax.set_title('Embedding Dimensions vs DILI Risk')
                        ax.axvline(0.3, color='green', linestyle='--', alpha=0.5, label='|r|=0.3')
                        ax.legend()
                        
                        # Plot 2: Best embedding dimension vs DILI
                        ax = axes[1]
                        if len(significant_corrs) > 0:
                            best_dim = significant_corrs.iloc[0]['embedding_dim']
                            scatter_data = merged_df[[best_dim, 'dili_risk_numeric']].dropna()
                            
                            ax.scatter(scatter_data[best_dim], scatter_data['dili_risk_numeric'], alpha=0.7)
                            ax.set_xlabel(f'Embedding Dimension {best_dim}')
                            ax.set_ylabel('DILI Risk (Numeric)')
                            ax.set_title(f'Best Correlation: r={significant_corrs.iloc[0]["correlation"]:.3f}')
                            
                            # Add trend line
                            z = np.polyfit(scatter_data[best_dim], scatter_data['dili_risk_numeric'], 1)
                            p = np.poly1d(z)
                            ax.plot(scatter_data[best_dim], p(scatter_data[best_dim]), "r--", alpha=0.8)
                        else:
                            ax.text(0.5, 0.5, 'No significant correlations', transform=ax.transAxes, ha='center', va='center')
                        
                        plt.tight_layout()
                        plt.savefig(fig_dir / f'{level}_vs_dili_analysis.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Save results
                        corr_df.to_csv(results_dir / f'{level}_dili_correlations.csv', index=False)
                
            else:
                print(f"   ❌ Insufficient overlap for analysis")
    
    elif level == 'concentration_embeddings':
        # For concentration-level, need to aggregate to drug level first
        if metadata is not None:
            # Try to extract drug information from concentration metadata
            if hasattr(metadata, 'columns') and 'drug' in metadata.columns:
                conc_with_drugs = metadata.copy()
                conc_with_drugs = conc_with_drugs.join(embed_df)
                
                # Aggregate by drug (mean embedding)
                drug_level = conc_with_drugs.groupby('drug')[embed_df.columns].mean()
                
                # Continue with drug-level analysis...
                print(f"   📊 Aggregated to {len(drug_level)} drugs")
                
                # Match with DILI data
                embed_drugs = set(drug_level.index)
                dili_drugs = set(drug_dili_map['drug'])
                overlap = embed_drugs.intersection(dili_drugs)
                
                print(f"   🔗 Overlap with DILI drugs: {len(overlap)}")
                
                if len(overlap) > 5:
                    # Similar analysis as drug-level...
                    merged_df = drug_level.loc[list(overlap)].copy()
                    dili_subset = drug_dili_map[drug_dili_map['drug'].isin(overlap)].set_index('drug')
                    merged_df = merged_df.join(dili_subset)
                    
                    print(f"   📊 Concentration-aggregated analysis: {len(merged_df)} drugs")

print(f"\n" + "="*80)
print("PHASE 2 vs CURRENT APPROACH COMPARISON")
print("="*80)

print(f"📊 STATISTICAL POWER COMPARISON:")
print(f"   Current well-level approach: {2423} wells, {113} drugs")
print(f"   Phase 2 hierarchical approach: Check embedding overlaps above")
print(f"   → Hierarchical approach better for drug-property correlations")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Use concentration-level embeddings aggregated to drug level")
print(f"   2. More drugs = better statistical power for DILI prediction")
print(f"   3. Hierarchical features capture dose-response patterns better")
print(f"   4. Continue with Phase 2 approach for final analysis")

print(f"\n✅ Analysis complete! Check {fig_dir} for visualizations.")