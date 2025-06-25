#!/usr/bin/env python3
"""
Inter-Event Feature Visualizations

PURPOSE:
    Creates comprehensive visualizations for inter-event period features
    to analyze temporal patterns, progressive effects, and SAX evolution.

INPUTS:
    - results/data/inter_event_features_wells.parquet
    - results/data/inter_event_features_drugs.parquet
    - results/data/sax_pattern_evolution.parquet

OUTPUTS:
    - results/figures/inter_event_features/temporal_feature_evolution.png
    - results/figures/inter_event_features/progressive_effects.png
    - results/figures/inter_event_features/sax_pattern_changes.png
    - results/figures/inter_event_features/baseline_vs_inter_event.png
    - results/figures/inter_event_features/drug_comparison_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from scipy import stats

warnings.filterwarnings('ignore')

# Setup directories
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "inter_event_features"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INTER-EVENT FEATURE VISUALIZATIONS")
print("=" * 80)

# Load data
print("\n📊 Loading inter-event feature data...")

try:
    wells_df = pd.read_parquet(results_dir / "inter_event_features_wells.parquet")
    print(f"   Well-level features: {wells_df.shape}")
except FileNotFoundError:
    print("   ERROR: Well-level features not found!")
    exit(1)

try:
    drugs_df = pd.read_parquet(results_dir / "inter_event_features_drugs.parquet")
    print(f"   Drug-level features: {drugs_df.shape}")
except FileNotFoundError:
    print("   WARNING: Drug-level features not found, creating from wells data")
    drugs_df = pd.DataFrame()

try:
    sax_evolution_df = pd.read_parquet(results_dir / "sax_pattern_evolution.parquet")
    print(f"   SAX evolution: {sax_evolution_df.shape}")
except FileNotFoundError:
    print("   WARNING: SAX evolution data not found")
    sax_evolution_df = pd.DataFrame()

print(f"   Period types: {wells_df['period_type'].value_counts().to_dict()}")

# ========== VISUALIZATION 1: TEMPORAL FEATURE EVOLUTION ==========

def create_temporal_evolution_plot():
    """Show how features evolve across sequential inter-event periods"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Temporal Feature Evolution Across Inter-Event Periods', fontsize=16, fontweight='bold')
    
    # Filter to inter-event periods only
    inter_event_data = wells_df[wells_df['period_type'] == 'inter_event'].copy()
    
    if len(inter_event_data) == 0:
        print("   No inter-event data for temporal evolution plot")
        return
    
    # Select key features to visualize
    key_features = [
        'mean_oxygen', 'cv_oxygen', 'trend_slope',
        'catch22_DN_HistogramMode_5', 'sax_coarse_entropy', 'baseline_deviation'
    ]
    
    feature_titles = [
        'Mean Oxygen Level', 'Coefficient of Variation', 'Trend Slope',
        'Histogram Mode (catch22)', 'SAX Entropy (Coarse)', 'Baseline Deviation'
    ]
    
    # Plot evolution for each feature
    for idx, (feature, title) in enumerate(zip(key_features, feature_titles)):
        ax = axes[idx // 3, idx % 3]
        
        if feature not in inter_event_data.columns:
            ax.text(0.5, 0.5, f'Feature {feature}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Group by period number and calculate statistics
        period_stats = inter_event_data.groupby('period_number')[feature].agg(['mean', 'std', 'count']).reset_index()
        period_stats = period_stats[period_stats['count'] >= 5]  # Minimum sample size
        
        if len(period_stats) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Plot mean with error bars
        ax.errorbar(period_stats['period_number'], period_stats['mean'], 
                   yerr=period_stats['std'], fmt='o-', capsize=5, capthick=2,
                   linewidth=2, markersize=8, alpha=0.8, color='steelblue')
        
        # Add trend line
        if len(period_stats) > 2:
            z = np.polyfit(period_stats['period_number'], period_stats['mean'], 1)
            p = np.poly1d(z)
            ax.plot(period_stats['period_number'], p(period_stats['period_number']), 
                   'r--', alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.3f}x')
            ax.legend()
        
        ax.set_xlabel('Period Number (Sequential Events)')
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add sample sizes
        for _, row in period_stats.iterrows():
            ax.text(row['period_number'], row['mean'] + row['std'] * 1.1, 
                   f'n={int(row["count"])}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'temporal_feature_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== VISUALIZATION 2: PROGRESSIVE EFFECTS ==========

def create_progressive_effects_plot():
    """Analyze progressive drug effects across sequential events"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Progressive Drug Effects Analysis', fontsize=16, fontweight='bold')
    
    # Filter data
    inter_event_data = wells_df[wells_df['period_type'] == 'inter_event'].copy()
    
    if len(inter_event_data) == 0:
        print("   No inter-event data for progressive effects plot")
        return
    
    # 1. Baseline deviation over time
    ax = axes[0, 0]
    if 'baseline_deviation' in inter_event_data.columns:
        period_deviation = inter_event_data.groupby('period_number')['baseline_deviation'].agg(['mean', 'std', 'count']).reset_index()
        period_deviation = period_deviation[period_deviation['count'] >= 3]
        
        if len(period_deviation) > 0:
            ax.errorbar(period_deviation['period_number'], period_deviation['mean'], 
                       yerr=period_deviation['std'], fmt='o-', capsize=5, 
                       linewidth=2, markersize=8, color='darkred')
            
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Baseline level')
            ax.set_xlabel('Period Number')
            ax.set_ylabel('Baseline Deviation')
            ax.set_title('A. Progressive Deviation from Baseline')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 2. Stability changes over time
    ax = axes[0, 1]
    if 'cv_oxygen' in inter_event_data.columns:
        period_cv = inter_event_data.groupby('period_number')['cv_oxygen'].agg(['mean', 'std', 'count']).reset_index()
        period_cv = period_cv[period_cv['count'] >= 3]
        
        if len(period_cv) > 0:
            ax.errorbar(period_cv['period_number'], period_cv['mean'], 
                       yerr=period_cv['std'], fmt='s-', capsize=5, 
                       linewidth=2, markersize=8, color='darkgreen')
            
            ax.set_xlabel('Period Number')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('B. Stability Changes Over Time')
            ax.grid(True, alpha=0.3)
    
    # 3. Drug-specific progressive patterns
    ax = axes[1, 0]
    # Select top drugs by number of periods
    drug_period_counts = inter_event_data.groupby('drug')['period_number'].nunique().sort_values(ascending=False)
    top_drugs = drug_period_counts.head(8).index
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_drugs)))
    
    for i, drug in enumerate(top_drugs):
        drug_data = inter_event_data[inter_event_data['drug'] == drug]
        if 'baseline_deviation' in drug_data.columns and len(drug_data) > 2:
            drug_progression = drug_data.groupby('period_number')['baseline_deviation'].mean()
            ax.plot(drug_progression.index, drug_progression.values, 
                   'o-', color=colors[i], label=drug[:15], alpha=0.8, linewidth=2)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Period Number')
    ax.set_ylabel('Mean Baseline Deviation')
    ax.set_title('C. Drug-Specific Progressive Patterns')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Adaptation vs Deterioration classification
    ax = axes[1, 1]
    
    # Calculate adaptation score for each well
    adaptation_scores = []
    
    for well_id in inter_event_data['well_id'].unique():
        well_data = inter_event_data[inter_event_data['well_id'] == well_id].sort_values('period_number')
        
        if len(well_data) >= 3 and 'baseline_deviation' in well_data.columns:
            # Calculate trend in baseline deviation
            periods = well_data['period_number'].values
            deviations = well_data['baseline_deviation'].values
            
            if not np.all(np.isnan(deviations)):
                slope, _, r_value, p_value, _ = stats.linregress(periods, deviations)
                
                adaptation_scores.append({
                    'well_id': well_id,
                    'drug': well_data['drug'].iloc[0],
                    'adaptation_slope': slope,
                    'adaptation_r2': r_value ** 2,
                    'adaptation_p_value': p_value,
                    'n_periods': len(well_data),
                    'classification': 'Adaptation' if slope > 0 else 'Deterioration'
                })
    
    if adaptation_scores:
        adaptation_df = pd.DataFrame(adaptation_scores)
        
        # Plot distribution of adaptation slopes
        adaptation_slopes = adaptation_df['adaptation_slope'].dropna()
        ax.hist(adaptation_slopes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
        ax.set_xlabel('Adaptation Slope (Δ Baseline Deviation / Period)')
        ax.set_ylabel('Number of Wells')
        ax.set_title('D. Adaptation vs Deterioration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text summary
        n_adaptation = (adaptation_df['adaptation_slope'] > 0).sum()
        n_deterioration = (adaptation_df['adaptation_slope'] < 0).sum()
        ax.text(0.02, 0.98, f'Adaptation: {n_adaptation}\nDeterioration: {n_deterioration}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'progressive_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== VISUALIZATION 3: SAX PATTERN CHANGES ==========

def create_sax_pattern_visualization():
    """Visualize SAX pattern evolution over time"""
    if len(sax_evolution_df) == 0:
        print("   No SAX evolution data for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SAX Pattern Evolution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Symbol frequency trends by level
    levels = sax_evolution_df['sax_level'].unique()
    
    for i, level in enumerate(levels[:3]):  # Limit to 3 levels
        ax = axes[0, i]
        level_data = sax_evolution_df[sax_evolution_df['sax_level'] == level]
        
        # Plot frequency change distribution by symbol
        symbols = level_data['symbol'].unique()
        symbol_changes = []
        symbol_labels = []
        
        for symbol in symbols:
            symbol_data = level_data[level_data['symbol'] == symbol]['freq_change'].dropna()
            if len(symbol_data) > 5:
                symbol_changes.append(symbol_data.values)
                symbol_labels.append(f'Symbol {symbol}')
        
        if symbol_changes:
            bp = ax.boxplot(symbol_changes, labels=symbol_labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No change')
            ax.set_ylabel('Frequency Change')
            ax.set_title(f'{level.title()} Level SAX Patterns')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 2. Most changing patterns by drug
    ax = axes[1, 0]
    
    # Calculate mean absolute frequency change by drug
    drug_pattern_changes = sax_evolution_df.groupby('drug')['freq_change'].apply(
        lambda x: np.mean(np.abs(x.dropna()))
    ).sort_values(ascending=False)
    
    top_changing_drugs = drug_pattern_changes.head(15)
    
    bars = ax.barh(range(len(top_changing_drugs)), top_changing_drugs.values, 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_changing_drugs)))
    ax.set_yticklabels([d[:20] for d in top_changing_drugs.index])
    ax.set_xlabel('Mean Absolute Pattern Change')
    ax.set_title('Drugs with Most Pattern Evolution')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Pattern stability heatmap
    ax = axes[1, 1]
    
    # Create heatmap of pattern changes by drug and symbol
    pattern_matrix = sax_evolution_df.pivot_table(
        index='drug', columns='symbol', values='freq_change', aggfunc='mean'
    )
    
    # Select top drugs and symbols with most variation
    if len(pattern_matrix) > 0:
        drug_variance = pattern_matrix.var(axis=1).sort_values(ascending=False)
        symbol_variance = pattern_matrix.var(axis=0).sort_values(ascending=False)
        
        top_drugs_matrix = drug_variance.head(15).index
        top_symbols_matrix = symbol_variance.head(6).index
        
        subset_matrix = pattern_matrix.loc[top_drugs_matrix, top_symbols_matrix]
        
        sns.heatmap(subset_matrix, cmap='RdBu_r', center=0, annot=False, 
                   fmt='.2f', cbar_kws={'label': 'Frequency Change'}, ax=ax)
        ax.set_title('Pattern Change Heatmap\n(Top Variable Drugs & Symbols)')
        ax.set_xlabel('SAX Symbols')
        ax.set_ylabel('Drugs')
    
    # 4. Temporal pattern evolution
    ax = axes[1, 2]
    
    # Show how pattern entropy changes over time
    if 'n_periods' in sax_evolution_df.columns:
        # Group by number of periods and calculate mean pattern diversity
        entropy_by_periods = sax_evolution_df.groupby(['drug', 'n_periods'])['freq_change'].apply(
            lambda x: np.std(x.dropna()) if len(x.dropna()) > 1 else 0
        ).reset_index()
        
        period_entropy = entropy_by_periods.groupby('n_periods')['freq_change'].agg(['mean', 'std', 'count']).reset_index()
        period_entropy = period_entropy[period_entropy['count'] >= 3]
        
        if len(period_entropy) > 0:
            ax.errorbar(period_entropy['n_periods'], period_entropy['mean'], 
                       yerr=period_entropy['std'], fmt='o-', capsize=5, 
                       linewidth=2, markersize=8, color='purple')
            
            ax.set_xlabel('Number of Periods')
            ax.set_ylabel('Pattern Variability (SD of Changes)')
            ax.set_title('Pattern Complexity vs Time')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'sax_pattern_changes.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== VISUALIZATION 4: BASELINE VS INTER-EVENT COMPARISON ==========

def create_baseline_comparison_plot():
    """Compare baseline vs inter-event period characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline vs Inter-Event Period Comparison', fontsize=16, fontweight='bold')
    
    baseline_data = wells_df[wells_df['period_type'] == 'baseline']
    inter_event_data = wells_df[wells_df['period_type'] == 'inter_event']
    
    # Select common features for comparison
    comparison_features = [
        'mean_oxygen', 'cv_oxygen', 'catch22_DN_HistogramMode_5', 'sax_coarse_entropy'
    ]
    
    feature_titles = [
        'Mean Oxygen Level', 'Coefficient of Variation', 'Histogram Mode', 'SAX Entropy'
    ]
    
    for idx, (feature, title) in enumerate(zip(comparison_features, feature_titles)):
        ax = axes[idx // 2, idx % 2]
        
        if feature not in wells_df.columns:
            ax.text(0.5, 0.5, f'Feature {feature}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Get data for both period types
        baseline_vals = baseline_data[feature].dropna()
        inter_event_vals = inter_event_data[feature].dropna()
        
        if len(baseline_vals) == 0 or len(inter_event_vals) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Create violin plots
        data_to_plot = [baseline_vals, inter_event_vals]
        labels = ['Baseline', 'Inter-Event']
        
        parts = ax.violinplot(data_to_plot, positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['lightblue', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add statistical test
        from scipy.stats import mannwhitneyu
        try:
            statistic, p_value = mannwhitneyu(baseline_vals, inter_event_vals, alternative='two-sided')
            ax.text(0.02, 0.98, f'p = {p_value:.3e}', transform=ax.transAxes, 
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add sample sizes
        ax.text(1, ax.get_ylim()[0], f'n={len(baseline_vals)}', ha='center', va='top')
        ax.text(2, ax.get_ylim()[0], f'n={len(inter_event_vals)}', ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'baseline_vs_inter_event.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== VISUALIZATION 5: DRUG COMPARISON HEATMAP ==========

def create_drug_comparison_heatmap():
    """Create comprehensive drug comparison heatmap"""
    if len(drugs_df) == 0:
        print("   No drug-level data for heatmap")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Drug Comparison: Inter-Event Features', fontsize=16, fontweight='bold')
    
    # 1. Baseline features heatmap
    baseline_drugs = drugs_df[drugs_df['period_type'] == 'baseline']
    
    if len(baseline_drugs) > 0:
        # Select numeric features
        feature_cols = [col for col in baseline_drugs.columns 
                       if col.endswith('_mean') and 'catch22' in col or 'sax' in col or col in ['mean_oxygen_mean', 'cv_oxygen_mean']]
        feature_cols = feature_cols[:15]  # Limit to top 15 features
        
        if feature_cols:
            baseline_matrix = baseline_drugs.set_index('drug')[feature_cols]
            baseline_matrix = baseline_matrix.dropna(how='all')
            
            # Standardize features
            scaler = StandardScaler()
            baseline_matrix_scaled = pd.DataFrame(
                scaler.fit_transform(baseline_matrix.fillna(baseline_matrix.mean())),
                index=baseline_matrix.index,
                columns=baseline_matrix.columns
            )
            
            sns.heatmap(baseline_matrix_scaled.T, cmap='RdBu_r', center=0, 
                       cbar_kws={'label': 'Standardized Value'}, ax=ax1)
            ax1.set_title('Baseline Period Features')
            ax1.set_xlabel('Drugs')
            ax1.set_ylabel('Features')
    
    # 2. Inter-event features heatmap
    inter_event_drugs = drugs_df[drugs_df['period_type'] == 'inter_event']
    
    if len(inter_event_drugs) > 0:
        # Select numeric features
        feature_cols = [col for col in inter_event_drugs.columns 
                       if col.endswith('_mean') and ('catch22' in col or 'sax' in col or col in ['mean_oxygen_mean', 'cv_oxygen_mean'])]
        feature_cols = feature_cols[:15]  # Limit to top 15 features
        
        if feature_cols:
            inter_event_matrix = inter_event_drugs.set_index('drug')[feature_cols]
            inter_event_matrix = inter_event_matrix.dropna(how='all')
            
            # Standardize features
            scaler = StandardScaler()
            inter_event_matrix_scaled = pd.DataFrame(
                scaler.fit_transform(inter_event_matrix.fillna(inter_event_matrix.mean())),
                index=inter_event_matrix.index,
                columns=inter_event_matrix.columns
            )
            
            sns.heatmap(inter_event_matrix_scaled.T, cmap='RdBu_r', center=0, 
                       cbar_kws={'label': 'Standardized Value'}, ax=ax2)
            ax2.set_title('Inter-Event Period Features')
            ax2.set_xlabel('Drugs')
            ax2.set_ylabel('Features')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'drug_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== GENERATE ALL VISUALIZATIONS ==========

print("\n📊 Creating inter-event feature visualizations...")

print("   1. Temporal feature evolution...")
create_temporal_evolution_plot()

print("   2. Progressive effects analysis...")
create_progressive_effects_plot()

print("   3. SAX pattern changes...")
create_sax_pattern_visualization()

print("   4. Baseline vs inter-event comparison...")
create_baseline_comparison_plot()

print("   5. Drug comparison heatmap...")
create_drug_comparison_heatmap()

print(f"\n✅ All visualizations saved to: {fig_dir}/")
print("\n📈 Generated files:")
print("   - temporal_feature_evolution.png")
print("   - progressive_effects.png") 
print("   - sax_pattern_changes.png")
print("   - baseline_vs_inter_event.png")
print("   - drug_comparison_heatmap.png")

print(f"\n🎯 These visualizations reveal:")
print("   - How features evolve across sequential media changes")
print("   - Progressive drug effects and adaptation patterns")
print("   - SAX symbolic pattern changes over time")
print("   - Differences between baseline and treatment periods")
print("   - Drug-specific response signatures")