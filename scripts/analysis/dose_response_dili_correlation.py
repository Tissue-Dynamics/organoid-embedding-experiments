#!/usr/bin/env python3
"""
Dose-Response Analysis and DILI Correlation
Simplified analysis focusing on key features and their relationship with DILI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
import os
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "dose_response_dili"
fig_dir.mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

print("=" * 80)
print("DOSE-RESPONSE ANALYSIS WITH DILI CORRELATION")
print("=" * 80)

# Load event-aware features
features_df = pd.read_parquet(results_dir / "event_aware_features_wells.parquet")
print(f"\n📊 Loaded {len(features_df)} well-level features")

# Get DILI data
conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

dili_query = """
SELECT DISTINCT
    drug,
    binary_dili,
    hepatotoxicity_boxed_warning
FROM postgres.drugs
WHERE drug IS NOT NULL
"""

dili_data = conn.execute(dili_query).fetchdf()
conn.close()

print(f"📊 DILI data: {len(dili_data)} drugs")

# Focus on key features identified from previous analysis
key_features = [
    'consumption_rate_mean',
    'cv_o2_mean', 
    'range_o2_mean',
    'min_o2_mean',
    'baseline_o2_mean',
    'consumption_ratio_mean',
    'total_monitored_hours_mean'
]

# Filter to features that exist
available_features = [f for f in key_features if f in features_df.columns]
print(f"\n🎯 Analyzing {len(available_features)} key features")

# Simple dose-response analysis per drug
dose_response_summary = []

for drug in features_df['drug'].unique():
    drug_data = features_df[features_df['drug'] == drug]
    
    # Get concentration series
    concentrations = sorted(drug_data['concentration'].unique())
    
    if len(concentrations) < 3:  # Need at least 3 concentrations
        continue
    
    drug_summary = {
        'drug': drug,
        'n_concentrations': len(concentrations),
        'concentration_range': f"{min(concentrations):.2e}-{max(concentrations):.2e}"
    }
    
    # For each feature, calculate simple dose-response metrics
    for feature in available_features:
        # Get mean response per concentration
        conc_responses = []
        for conc in concentrations:
            conc_data = drug_data[drug_data['concentration'] == conc][feature].dropna()
            if len(conc_data) > 0:
                conc_responses.append(conc_data.mean())
            else:
                conc_responses.append(np.nan)
        
        # Remove NaN values
        valid_mask = ~np.isnan(conc_responses)
        valid_conc = np.array(concentrations)[valid_mask]
        valid_resp = np.array(conc_responses)[valid_mask]
        
        if len(valid_conc) >= 3:
            # Calculate simple metrics
            # 1. Overall trend (Spearman correlation)
            trend_corr, trend_p = stats.spearmanr(valid_conc, valid_resp)
            
            # 2. Dynamic range
            dynamic_range = np.max(valid_resp) - np.min(valid_resp)
            
            # 3. Relative change from lowest to highest concentration
            if len(valid_resp) > 0:
                relative_change = (valid_resp[-1] - valid_resp[0]) / (abs(valid_resp[0]) + 1e-10)
            else:
                relative_change = 0
            
            # 4. Coefficient of variation across concentrations
            cv_across_conc = np.std(valid_resp) / (np.mean(valid_resp) + 1e-10)
            
            # Store metrics
            drug_summary[f'{feature}_trend'] = trend_corr
            drug_summary[f'{feature}_trend_p'] = trend_p
            drug_summary[f'{feature}_range'] = dynamic_range
            drug_summary[f'{feature}_change'] = relative_change
            drug_summary[f'{feature}_cv'] = cv_across_conc
            
            # Store actual values for visualization
            drug_summary[f'{feature}_min_conc_value'] = valid_resp[0] if len(valid_resp) > 0 else np.nan
            drug_summary[f'{feature}_max_conc_value'] = valid_resp[-1] if len(valid_resp) > 0 else np.nan
    
    dose_response_summary.append(drug_summary)

# Convert to DataFrame
dr_summary_df = pd.DataFrame(dose_response_summary)

# Merge with DILI data
merged_df = pd.merge(dr_summary_df, dili_data, on='drug', how='inner')
print(f"\n🔗 Merged data: {len(merged_df)} drugs with dose-response and DILI")

# Correlate dose-response metrics with DILI
print(f"\n📈 DOSE-RESPONSE vs DILI CORRELATIONS:")

correlation_results = []

for feature in available_features:
    # Check which metrics exist
    metric_cols = [col for col in merged_df.columns if col.startswith(f'{feature}_') and col.endswith(('_trend', '_range', '_change', '_cv'))]
    
    for metric_col in metric_cols:
        # Remove rows with NaN
        valid_data = merged_df[[metric_col, 'binary_dili']].dropna()
        
        if len(valid_data) > 10:
            # Calculate correlation with binary DILI
            corr, p_val = stats.spearmanr(valid_data[metric_col], valid_data['binary_dili'].astype(int))
            
            correlation_results.append({
                'feature': feature,
                'metric': metric_col.split('_')[-1],
                'correlation': corr,
                'p_value': p_val,
                'n_drugs': len(valid_data),
                'significant': p_val < 0.05
            })

# Convert to DataFrame and sort by correlation
corr_results_df = pd.DataFrame(correlation_results)
corr_results_df['abs_correlation'] = np.abs(corr_results_df['correlation'])
corr_results_df = corr_results_df.sort_values('abs_correlation', ascending=False)

print(f"\n🏆 TOP DOSE-RESPONSE METRICS FOR DILI PREDICTION:")
for _, row in corr_results_df.head(10).iterrows():
    sig_marker = '*' if row['significant'] else ''
    print(f"   {row['feature']}_{row['metric']}: r = {row['correlation']:.3f} (p = {row['p_value']:.4f}){sig_marker}")

# Create visualizations
print(f"\n📊 Creating visualizations...")

# Figure 1: Top dose-response patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Top Dose-Response Patterns vs DILI Risk', fontsize=16, fontweight='bold')

# Select top 4 metrics
top_metrics = corr_results_df.head(4)

for idx, (_, metric_info) in enumerate(top_metrics.iterrows()):
    ax = axes[idx // 2, idx % 2]
    
    feature = metric_info['feature']
    metric = metric_info['metric']
    metric_col = f"{feature}_{metric}"
    
    # Plot DILI vs non-DILI
    dili_data = merged_df[merged_df['binary_dili'] == True][metric_col].dropna()
    non_dili_data = merged_df[merged_df['binary_dili'] == False][metric_col].dropna()
    
    # Violin plot
    data_to_plot = []
    labels = []
    
    if len(non_dili_data) > 0:
        data_to_plot.append(non_dili_data)
        labels.append(f'No DILI\n(n={len(non_dili_data)})')
    
    if len(dili_data) > 0:
        data_to_plot.append(dili_data)
        labels.append(f'DILI\n(n={len(dili_data)})')
    
    if len(data_to_plot) > 0:
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['lightblue', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors[:len(parts['bodies'])]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{feature.replace('_', ' ').title()} - {metric}")
    ax.set_title(f"{feature} {metric}\nr = {metric_info['correlation']:.3f}, p = {metric_info['p_value']:.4f}")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical test
    if len(data_to_plot) == 2:
        _, p_val = stats.mannwhitneyu(data_to_plot[0], data_to_plot[1])
        ax.text(0.5, 0.95, f'Mann-Whitney p = {p_val:.4f}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(fig_dir / 'dose_response_dili_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Feature correlation heatmap
print(f"\n📊 Creating correlation heatmap...")

# Pivot correlation results for heatmap
heatmap_data = corr_results_df.pivot(index='feature', columns='metric', values='correlation')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Spearman Correlation with DILI'})
ax.set_title('Dose-Response Metrics vs DILI Correlation', fontsize=14)
ax.set_xlabel('Dose-Response Metric')
ax.set_ylabel('Feature')

plt.tight_layout()
plt.savefig(fig_dir / 'dose_response_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Example dose-response curves for top correlating features
print(f"\n📊 Creating example dose-response curves...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Example Dose-Response Curves: DILI vs Non-DILI Drugs', fontsize=16, fontweight='bold')

# Select top feature with highest correlation
if len(top_metrics) > 0:
    top_feature = top_metrics.iloc[0]['feature']
    
    # Get example drugs
    dili_drugs = merged_df[merged_df['binary_dili'] == True]['drug'].values[:2]
    non_dili_drugs = merged_df[merged_df['binary_dili'] == False]['drug'].values[:2]
    example_drugs = list(dili_drugs) + list(non_dili_drugs)
    
    for idx, drug in enumerate(example_drugs[:4]):
        ax = axes[idx // 2, idx % 2]
        
        drug_data = features_df[features_df['drug'] == drug]
        
        if len(drug_data) > 0:
            # Get concentration-response data
            conc_resp = drug_data.groupby('concentration')[top_feature].agg(['mean', 'std', 'count'])
            conc_resp = conc_resp.reset_index()
            conc_resp = conc_resp.sort_values('concentration')
            
            # Plot with error bars
            ax.errorbar(conc_resp['concentration'], conc_resp['mean'], 
                       yerr=conc_resp['std'], fmt='o-', capsize=5, capthick=2,
                       markersize=8, linewidth=2)
            
            # Add DILI status
            dili_status = merged_df[merged_df['drug'] == drug]['binary_dili'].iloc[0]
            color = 'red' if dili_status else 'blue'
            ax.set_title(f"{drug}\n{'DILI' if dili_status else 'Non-DILI'}", color=color)
            
            ax.set_xscale('log')
            ax.set_xlabel('Concentration')
            ax.set_ylabel(top_feature.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'example_dose_response_curves_dili.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
print(f"\n💾 Saving results...")

# Save dose-response summary
dr_summary_df.to_parquet(results_dir / 'dose_response_simple_metrics.parquet', index=False)

# Save correlation results
corr_results_df.to_parquet(results_dir / 'dose_response_dili_correlations.parquet', index=False)

# Save merged data
merged_df.to_parquet(results_dir / 'dose_response_dili_merged.parquet', index=False)

print(f"   Dose-response metrics: {results_dir / 'dose_response_simple_metrics.parquet'}")
print(f"   DILI correlations: {results_dir / 'dose_response_dili_correlations.parquet'}")
print(f"   Merged data: {results_dir / 'dose_response_dili_merged.parquet'}")

print(f"\n✅ Dose-response DILI analysis complete!")

# Summary statistics
print(f"\n📊 ANALYSIS SUMMARY:")
print(f"   Drugs analyzed: {len(merged_df)}")
print(f"   DILI positive: {merged_df['binary_dili'].sum()} ({merged_df['binary_dili'].sum()/len(merged_df)*100:.1f}%)")
print(f"   Features analyzed: {len(available_features)}")
print(f"   Total correlations: {len(corr_results_df)}")
print(f"   Significant correlations: {corr_results_df['significant'].sum()}")

# Report best predictors
print(f"\n🎯 BEST DOSE-RESPONSE PREDICTORS OF DILI:")
best_predictors = corr_results_df[corr_results_df['significant']].head(5)
for _, pred in best_predictors.iterrows():
    print(f"   {pred['feature']}_{pred['metric']}: r = {pred['correlation']:.3f} (p = {pred['p_value']:.4f})")