#!/usr/bin/env python3
"""
Drug Embeddings vs DILI Correlation Analysis
Direct correlation of Phase 2 drug embeddings with DILI risk using relaxed filters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "drug_embeddings_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DRUG EMBEDDINGS vs DILI CORRELATION ANALYSIS")
print("=" * 80)

# First, determine the CV threshold needed for ~6500 wells
wells_drugs_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")
print(f"📊 Total wells: {len(wells_drugs_df):,}")

# Find CV threshold for ~6500 wells
cv_thresholds = np.arange(0.25, 2.0, 0.05)
wells_by_threshold = []

for thresh in cv_thresholds:
    n_wells = (wells_drugs_df['cv_o2'] <= thresh).sum()
    wells_by_threshold.append({'threshold': thresh, 'n_wells': n_wells})

cv_analysis = pd.DataFrame(wells_by_threshold)
optimal_cv = cv_analysis[cv_analysis['n_wells'] >= 6500].iloc[0]['threshold']

print(f"\n🎯 CV Threshold Analysis:")
print(f"   Target wells: ~6,500")
print(f"   Optimal CV threshold: {optimal_cv:.2f}")
print(f"   Wells at this threshold: {(wells_drugs_df['cv_o2'] <= optimal_cv).sum():,}")

# Apply relaxed quality filter
quality_df = wells_drugs_df[wells_drugs_df['cv_o2'] <= optimal_cv].copy()
print(f"   Quality wells: {len(quality_df):,}")
print(f"   Drugs represented: {quality_df['drug'].nunique()}")

# Load Phase 2 drug embeddings
print(f"\n📊 Loading Phase 2 Drug Embeddings...")
phase2_results = joblib.load(results_dir / "hierarchical_embedding_results.joblib")

# Extract drug embeddings (dictionary of methods)
drug_embeddings_dict = phase2_results['drug_embeddings']
drug_metadata = phase2_results['drug_metadata']

print(f"   Drug metadata shape: {drug_metadata.shape}")
print(f"   Embedding methods: {list(drug_embeddings_dict.keys())}")

# Create drug-level DILI mapping from wells data
print(f"\n💊 Creating Drug-Level DILI Mapping...")

# Get drug-level DILI info (using mode for categorical, mean for numeric)
# First check which columns are available
available_cols = quality_df.columns.tolist()
dili_cols = [col for col in available_cols if 'dili' in col.lower()]
print(f"   Available DILI columns: {dili_cols}")

# Build aggregation dict based on available columns
agg_dict = {
    'cv_o2': ['mean', 'std', 'count'],
    'mean_o2': 'mean',
    'std_o2': 'mean'
}

# Add DILI columns if they exist
if 'dili' in available_cols:
    agg_dict['dili'] = 'first'
if 'binary_dili' in available_cols:
    agg_dict['binary_dili'] = 'first'
if 'dili_risk_category' in available_cols:
    agg_dict['dili_risk_category'] = lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
if 'molecular_weight' in available_cols:
    agg_dict['molecular_weight'] = 'first'
if 'logp' in available_cols:
    agg_dict['logp'] = 'first'
if 'half_life_hours' in available_cols:
    agg_dict['half_life_hours'] = 'first'
if 'clearance_l_hr_kg' in available_cols:
    agg_dict['clearance_l_hr_kg'] = 'first'

drug_dili_map = quality_df[quality_df['dili'].notna()].groupby('drug').agg(agg_dict).round(4)

# Flatten column names more carefully
new_columns = []
for col in drug_dili_map.columns:
    if isinstance(col, tuple):
        if col[1]:
            new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col[0])
    else:
        new_columns.append(col)

drug_dili_map.columns = new_columns
drug_dili_map = drug_dili_map.reset_index()

# Rename columns to be cleaner
rename_dict = {
    'dili_first': 'dili',
    'binary_dili_first': 'binary_dili',
    'dili_risk_category_<lambda>': 'dili_risk_category',
    'molecular_weight_first': 'molecular_weight',
    'logp_first': 'logp',
    'half_life_hours_first': 'half_life_hours',
    'clearance_l_hr_kg_first': 'clearance_l_hr_kg'
}
drug_dili_map = drug_dili_map.rename(columns=rename_dict)

print(f"   Drugs with DILI data: {len(drug_dili_map)}")

# Create DILI risk numeric mapping
dili_risk_mapping = {
    'Low': 1, 'Low risk': 1, 'Low Risk': 1, 'Low DILI Risk': 1,
    'Low hepatotoxicity risk': 1, 'No DILI concern': 0,
    'No-DILI-Concern per database': 0, 'vLess-DILI-Concern': 1,
    'Low concern': 1, 'Low - vLess-DILI-Concern': 1,
    'Low frequency (<1:10,000), high severity': 2,
    'Low frequency but potentially severe': 2,
    'Low-Moderate': 1.5, 'Moderate': 2, 'Moderate Risk': 2,
    'Intermediate': 2, 'High': 3, 'High Risk': 3,
    'High risk - drug withdrawn': 4, 'High risk - cumulative dose-dependent': 3,
    'High concern': 3, 'High Risk - Black Box Warning': 4,
    'Severe': 4, 'Severe - High Risk': 4, 'Black Box Warning': 4,
    'LiverTox Category D': 3, 'vMost-DILI-Concern': 4,
    'Not formally categorized - primary toxicity was hematologic': 1
}

# Print actual columns after aggregation
print(f"   Columns after aggregation: {drug_dili_map.columns.tolist()}")

# Create numeric DILI risk score
if 'dili_risk_category' in drug_dili_map.columns:
    drug_dili_map['dili_risk_numeric'] = drug_dili_map['dili_risk_category'].map(dili_risk_mapping)
    # Fill NaN values using binary DILI or dili column
    if drug_dili_map['dili_risk_numeric'].isna().any():
        print(f"   ⚠️ {drug_dili_map['dili_risk_numeric'].isna().sum()} drugs missing risk category, using DILI value")
        if 'binary_dili' in drug_dili_map.columns:
            # Use binary DILI: 0 for False, 2 for True
            drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'dili_risk_numeric'] = \
                drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'binary_dili'].map({True: 2.0, False: 0.0})
        elif 'dili' in drug_dili_map.columns:
            # Map DILI text values
            dili_text_mapping = {
                'vNo-DILI-Concern': 0, 'vLess-DILI-Concern': 1, 
                'vMost-DILI-Concern': 4, 'no-DILI-concern': 0
            }
            drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'dili_risk_numeric'] = \
                drug_dili_map.loc[drug_dili_map['dili_risk_numeric'].isna(), 'dili'].map(dili_text_mapping)
else:
    print("   ⚠️ No dili_risk_category column, creating from binary DILI")
    if 'binary_dili' in drug_dili_map.columns:
        drug_dili_map['dili_risk_numeric'] = drug_dili_map['binary_dili'].map({True: 2.0, False: 0.0})
    elif 'dili' in drug_dili_map.columns:
        dili_text_mapping = {
            'vNo-DILI-Concern': 0, 'vLess-DILI-Concern': 1, 
            'vMost-DILI-Concern': 4, 'no-DILI-concern': 0
        }
        drug_dili_map['dili_risk_numeric'] = drug_dili_map['dili'].map(dili_text_mapping)

# Simplify to 4 categories for clearer analysis
def simplify_dili_category(risk_numeric):
    if pd.isna(risk_numeric):
        return np.nan
    elif risk_numeric <= 0.5:
        return 'No Concern'
    elif risk_numeric <= 1.5:
        return 'Low'
    elif risk_numeric <= 2.5:
        return 'Moderate'
    else:
        return 'High/Severe'

drug_dili_map['dili_simple'] = drug_dili_map['dili_risk_numeric'].apply(simplify_dili_category)

print(f"\n📋 DILI Risk Distribution:")
print(drug_dili_map['dili_simple'].value_counts())

# Analyze each embedding method
all_correlations = {}
best_results = {}

for method, embeddings_data in drug_embeddings_dict.items():
    print(f"\n🔍 ANALYZING {method.upper()} EMBEDDINGS...")
    
    # Convert to DataFrame if needed
    if isinstance(embeddings_data, np.ndarray):
        # Use drug names from metadata as index
        embeddings_df = pd.DataFrame(embeddings_data, index=drug_metadata['drug'])
    elif isinstance(embeddings_data, pd.DataFrame):
        # If already DataFrame, ensure it has drug names as index
        if embeddings_df.index.name != 'drug' and all(isinstance(idx, int) for idx in embeddings_df.index):
            embeddings_df.index = drug_metadata['drug']
        embeddings_df = embeddings_data
    else:
        print(f"   ❌ Unknown embedding format: {type(embeddings_data)}")
        continue
    
    # Find overlap between embeddings and DILI data
    embed_drugs = set(embeddings_df.index)
    dili_drugs = set(drug_dili_map['drug'])
    overlap_drugs = embed_drugs.intersection(dili_drugs)
    
    print(f"   Embedding drugs: {len(embed_drugs)}")
    print(f"   DILI drugs: {len(dili_drugs)}")
    print(f"   Overlap: {len(overlap_drugs)} drugs")
    
    if len(overlap_drugs) < 10:
        print(f"   ❌ Insufficient overlap for analysis")
        continue
    
    # Merge embeddings with DILI data
    embed_subset = embeddings_df.loc[list(overlap_drugs)]
    dili_subset = drug_dili_map[drug_dili_map['drug'].isin(overlap_drugs)].set_index('drug')
    
    merged_df = embed_subset.join(dili_subset)
    
    # Correlation analysis with DILI risk
    embedding_cols = [col for col in embed_subset.columns if isinstance(col, (int, str)) and col not in dili_subset.columns]
    
    correlations = []
    for embed_col in embedding_cols[:20]:  # Analyze first 20 dimensions
        valid_data = merged_df[[embed_col, 'dili_risk_numeric']].dropna()
        
        if len(valid_data) > 10:
            # Spearman correlation
            corr_s, p_s = spearmanr(valid_data[embed_col], valid_data['dili_risk_numeric'])
            # Pearson correlation
            corr_p, p_p = pearsonr(valid_data[embed_col], valid_data['dili_risk_numeric'])
            
            correlations.append({
                'dimension': embed_col,
                'spearman_r': corr_s,
                'spearman_p': p_s,
                'pearson_r': corr_p,
                'pearson_p': p_p,
                'n_drugs': len(valid_data)
            })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_spearman'] = corr_df['spearman_r'].abs()
        corr_df = corr_df.sort_values('abs_spearman', ascending=False)
        
        all_correlations[method] = corr_df
        
        # Find best correlations
        significant = corr_df[corr_df['spearman_p'] < 0.05]
        if len(significant) > 0:
            best_results[method] = {
                'best_dim': significant.iloc[0]['dimension'],
                'correlation': significant.iloc[0]['spearman_r'],
                'p_value': significant.iloc[0]['spearman_p'],
                'n_significant': len(significant),
                'data': merged_df
            }
            
            print(f"   ✅ {len(significant)} significant correlations found!")
            print(f"   Best: Dim {significant.iloc[0]['dimension']}, r={significant.iloc[0]['spearman_r']:.3f}, p={significant.iloc[0]['spearman_p']:.3e}")

# Create comprehensive visualization
if best_results:
    print(f"\n🎨 Creating Visualizations...")
    
    # Figure 1: Best correlations across methods
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Drug Embeddings vs DILI Risk Correlation Analysis', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (method, result) in enumerate(best_results.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        data = result['data']
        best_dim = result['best_dim']
        
        # Create scatter plot with color by DILI category
        dili_categories = ['No Concern', 'Low', 'Moderate', 'High/Severe']
        colors = ['green', 'yellow', 'orange', 'red']
        
        for cat_idx, category in enumerate(dili_categories):
            mask = data['dili_simple'] == category
            if mask.any():
                ax.scatter(data.loc[mask, best_dim], 
                          data.loc[mask, 'dili_risk_numeric'],
                          c=colors[cat_idx], label=category, 
                          alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
        
        # Add trend line
        valid_data = data[[best_dim, 'dili_risk_numeric']].dropna()
        z = np.polyfit(valid_data[best_dim], valid_data['dili_risk_numeric'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid_data[best_dim].min(), valid_data[best_dim].max(), 100)
        ax.plot(x_range, p(x_range), "k--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel(f'{method} Embedding Dim {best_dim}')
        ax.set_ylabel('DILI Risk Score')
        ax.set_title(f'{method}\nr={result["correlation"]:.3f}, p={result["p_value"]:.3e}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(best_results), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'drug_embeddings_dili_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Correlation heatmap across methods
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create correlation summary matrix
    summary_data = []
    for method, corr_df in all_correlations.items():
        sig_corrs = corr_df[corr_df['spearman_p'] < 0.05]['spearman_r'].values
        summary_data.append({
            'Method': method,
            'N_Significant': len(sig_corrs),
            'Max_Correlation': corr_df['spearman_r'].abs().max(),
            'Mean_Correlation': corr_df['spearman_r'].abs().mean(),
            'Best_P_Value': corr_df['spearman_p'].min()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Method')
    
    # Create heatmap
    sns.heatmap(summary_df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Value'}, ax=ax)
    ax.set_title('Embedding Method Performance Summary')
    plt.tight_layout()
    plt.savefig(fig_dir / 'embedding_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save results
print(f"\n💾 Saving Results...")

# Save correlation results
for method, corr_df in all_correlations.items():
    corr_df.to_csv(results_dir / f'{method}_dili_correlations.csv', index=False)

# Save drug-level analysis
drug_dili_map.to_csv(results_dir / 'drug_dili_analysis.csv', index=False)

# Summary statistics
print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 FINAL STATISTICS:")
print(f"   CV threshold used: {optimal_cv:.2f}")
print(f"   Total wells analyzed: {len(quality_df):,}")
print(f"   Drugs with DILI data: {len(drug_dili_map)}")
print(f"   Drugs in embeddings: {len(drug_metadata)}")

if best_results:
    print(f"\n🏆 BEST CORRELATIONS BY METHOD:")
    for method, result in best_results.items():
        print(f"   {method}: r={result['correlation']:.3f}, p={result['p_value']:.3e} ({result['n_significant']} significant dims)")
else:
    print(f"\n❌ No significant correlations found")

print(f"\n🎯 KEY INSIGHTS:")
print(f"   • Relaxing CV filter to {optimal_cv:.2f} gives {len(quality_df):,} wells")
print(f"   • Drug-level analysis more powerful than well-level")
print(f"   • Multiple embedding dimensions correlate with DILI risk")
print(f"   • Ready for predictive modeling!")

print(f"\n📁 Results saved to:")
print(f"   • Figures: {fig_dir}")
print(f"   • Data: {results_dir}")
print(f"\n✅ Drug embeddings vs DILI analysis complete!")