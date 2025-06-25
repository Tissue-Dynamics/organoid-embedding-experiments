#!/usr/bin/env python3
"""
Baseline-Specific Feature Extraction

PURPOSE:
    Extracts features specifically from the baseline period (0-48h) before drug treatment.
    These features serve as reference values for normalization and help identify
    wells with stable pre-treatment conditions.

METHODOLOGY:
    - Identifies first 48 hours as baseline period for each well
    - Extracts catch22 features from baseline period only
    - Extracts hierarchical SAX features at multiple resolutions
    - Calculates summary statistics (mean, std, cv, trend)
    - Assesses baseline stability and flags problematic wells

INPUTS:
    - Database connection via DATABASE_URL environment variable
    - Queries oxygen consumption data from first 48 hours

OUTPUTS:
    - results/data/baseline_catch22_features.parquet
      catch22 features extracted from baseline period
    - results/data/baseline_sax_features.parquet
      Hierarchical SAX features from baseline
    - results/data/baseline_summary_statistics.parquet
      Summary statistics and stability metrics
    - results/data/baseline_quality_assessment.parquet
      Quality flags specific to baseline period
    - results/figures/baseline_features/
      Comprehensive visualizations of baseline characteristics

REQUIREMENTS:
    - numpy, pandas, pycatch22, scipy, matplotlib, seaborn
    - pyts for SAX transformation
    - Minimum 20 timepoints in first 48h for valid baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pycatch22
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from collections import Counter
from sklearn.preprocessing import StandardScaler
from pyts.approximation import SymbolicAggregateApproximation

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader

# Setup directories
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures" / "baseline_features"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BASELINE-SPECIFIC FEATURE EXTRACTION")
print("=" * 80)

# Configuration
BASELINE_HOURS = 48  # First 48 hours as baseline
MIN_BASELINE_POINTS = 20  # Minimum points needed for valid baseline
STABILITY_CV_THRESHOLD = 0.1  # CV threshold for stable baseline

# SAX configuration (same as hierarchical_sax.py)
SAX_CONFIGS = [
    {'n_symbols': 4, 'alphabet_size': 3, 'name': 'coarse'},
    {'n_symbols': 8, 'alphabet_size': 4, 'name': 'medium'},
    {'n_symbols': 16, 'alphabet_size': 6, 'name': 'fine'}
]

# Initialize data loader
loader = DataLoader()

print("\n📊 Loading oxygen consumption data...")
# Load only a subset for faster processing
df = loader.load_oxygen_data(limit=10)  # Process 10 plates for testing
print(f"   Loaded {len(df):,} measurements from {df['well_id'].nunique():,} wells")

# Rename o2 column to oxygen for consistency
df = df.rename(columns={'o2': 'oxygen'})

# ========== BASELINE EXTRACTION FUNCTIONS ==========

def extract_baseline_period(well_data, baseline_hours=48):
    """Extract data from the baseline period"""
    baseline_data = well_data[well_data['elapsed_hours'] <= baseline_hours].copy()
    
    # Check if we have enough data
    if len(baseline_data) < MIN_BASELINE_POINTS:
        return None, {
            'baseline_valid': False,
            'n_points': len(baseline_data),
            'reason': 'insufficient_points'
        }
    
    # Calculate baseline statistics
    stats_dict = {
        'baseline_valid': True,
        'n_points': len(baseline_data),
        'duration_hours': baseline_data['elapsed_hours'].max(),
        'mean_o2': baseline_data['oxygen'].mean(),
        'std_o2': baseline_data['oxygen'].std(),
        'cv_o2': baseline_data['oxygen'].std() / baseline_data['oxygen'].mean() if baseline_data['oxygen'].mean() > 0 else np.nan,
        'min_o2': baseline_data['oxygen'].min(),
        'max_o2': baseline_data['oxygen'].max(),
        'range_o2': baseline_data['oxygen'].max() - baseline_data['oxygen'].min()
    }
    
    # Calculate trend (linear fit)
    if len(baseline_data) > 5:
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                baseline_data['elapsed_hours'], baseline_data['oxygen']
            )
            stats_dict.update({
                'baseline_slope': slope,
                'baseline_intercept': intercept,
                'baseline_r2': r_value ** 2,
                'baseline_trend_pvalue': p_value
            })
        except:
            stats_dict.update({
                'baseline_slope': np.nan,
                'baseline_intercept': np.nan,
                'baseline_r2': np.nan,
                'baseline_trend_pvalue': np.nan
            })
    
    # Assess stability
    stats_dict['baseline_stable'] = stats_dict['cv_o2'] < STABILITY_CV_THRESHOLD
    
    return baseline_data, stats_dict

def extract_baseline_catch22(baseline_data, well_id):
    """Extract catch22 features from baseline period"""
    if baseline_data is None or len(baseline_data) < MIN_BASELINE_POINTS:
        return None
    
    try:
        # Extract all catch22 features
        features = pycatch22.catch22_all(baseline_data['oxygen'].values)
        
        # Create feature dictionary
        feature_dict = {
            'well_id': well_id,
            'baseline_hours': baseline_data['elapsed_hours'].max()
        }
        
        # Add each catch22 feature
        for name, value in zip(features['names'], features['values']):
            feature_dict[f'baseline_{name}'] = value
        
        return feature_dict
        
    except Exception as e:
        print(f"Error extracting catch22 for {well_id}: {e}")
        return None

def extract_baseline_sax(baseline_data, well_id):
    """Extract hierarchical SAX features from baseline period"""
    if baseline_data is None or len(baseline_data) < MIN_BASELINE_POINTS:
        return None
    
    all_features = {'well_id': well_id}
    
    for config in SAX_CONFIGS:
        try:
            # Initialize SAX transformer (pyts)
            sax = SymbolicAggregateApproximation(
                n_bins=config['n_symbols'],
                strategy='uniform'
            )
            
            # Transform to SAX (returns integer representation)
            ts_values = baseline_data['oxygen'].values.reshape(1, -1)
            sax_array = sax.fit_transform(ts_values)[0]
            
            # Map to alphabet based on quantiles
            alphabet_size = config['alphabet_size']
            # Scale integer values to alphabet range
            sax_scaled = np.clip(
                (sax_array * alphabet_size).astype(int), 
                0, alphabet_size - 1
            )
            
            # Convert to string representation
            sax_str = ''.join([chr(65 + s) for s in sax_scaled])
            
            # Extract features
            prefix = f"baseline_sax_{config['name']}"
            
            # Store SAX string
            all_features[f"{prefix}_string"] = sax_str
            
            # Pattern frequencies
            pattern_counts = Counter(sax_str)
            for symbol in range(alphabet_size):
                symbol_char = chr(65 + symbol)
                all_features[f"{prefix}_freq_{symbol_char}"] = pattern_counts.get(symbol_char, 0) / len(sax_str)
            
            # Transitions
            transitions = sum(1 for i in range(len(sax_str)-1) if sax_str[i] != sax_str[i+1])
            all_features[f"{prefix}_transitions"] = transitions / (len(sax_str) - 1) if len(sax_str) > 1 else 0
            
            # Entropy
            probs = np.array(list(pattern_counts.values())) / len(sax_str)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            all_features[f"{prefix}_entropy"] = entropy
            
            # Complexity (unique patterns)
            all_features[f"{prefix}_unique_symbols"] = len(pattern_counts)
            
        except Exception as e:
            print(f"Error in SAX transformation for {well_id} at {config['name']} level: {e}")
            # Add NaN features
            prefix = f"baseline_sax_{config['name']}"
            for symbol in range(config['alphabet_size']):
                all_features[f"{prefix}_freq_{chr(65 + symbol)}"] = np.nan
            all_features[f"{prefix}_transitions"] = np.nan
            all_features[f"{prefix}_entropy"] = np.nan
            all_features[f"{prefix}_unique_symbols"] = np.nan
            all_features[f"{prefix}_string"] = ''
    
    return all_features

# ========== PROCESS ALL WELLS ==========

baseline_stats_list = []
baseline_catch22_list = []
baseline_sax_list = []

print(f"\n🔄 Processing baseline period for {df['well_id'].nunique():,} wells...")

for well_id in tqdm(df['well_id'].unique(), desc="Extracting baseline features"):
    well_data = df[df['well_id'] == well_id].sort_values('elapsed_hours')
    
    # Extract baseline period
    baseline_data, stats = extract_baseline_period(well_data, BASELINE_HOURS)
    
    # Add metadata
    stats['well_id'] = well_id
    stats['drug'] = well_data['drug'].iloc[0]
    stats['concentration'] = well_data['concentration'].iloc[0]
    stats['plate_id'] = well_data['plate_id'].iloc[0]
    
    baseline_stats_list.append(stats)
    
    # Extract features if baseline is valid
    if stats['baseline_valid']:
        # catch22 features
        catch22_features = extract_baseline_catch22(baseline_data, well_id)
        if catch22_features:
            catch22_features['drug'] = stats['drug']
            catch22_features['concentration'] = stats['concentration']
            baseline_catch22_list.append(catch22_features)
        
        # SAX features
        sax_features = extract_baseline_sax(baseline_data, well_id)
        if sax_features:
            sax_features['drug'] = stats['drug']
            sax_features['concentration'] = stats['concentration']
            baseline_sax_list.append(sax_features)

# Create DataFrames
baseline_stats_df = pd.DataFrame(baseline_stats_list)
baseline_catch22_df = pd.DataFrame(baseline_catch22_list) if baseline_catch22_list else pd.DataFrame()
baseline_sax_df = pd.DataFrame(baseline_sax_list) if baseline_sax_list else pd.DataFrame()

print(f"\n📊 BASELINE EXTRACTION RESULTS:")
print(f"   Total wells: {len(baseline_stats_df):,}")
print(f"   Valid baselines: {baseline_stats_df['baseline_valid'].sum():,} ({baseline_stats_df['baseline_valid'].mean()*100:.1f}%)")
print(f"   Stable baselines: {baseline_stats_df['baseline_stable'].sum():,} ({baseline_stats_df['baseline_stable'].mean()*100:.1f}%)")
print(f"   catch22 features extracted: {len(baseline_catch22_df):,}")
print(f"   SAX features extracted: {len(baseline_sax_df):,}")

# ========== QUALITY ASSESSMENT ==========

# Create quality assessment DataFrame
quality_df = baseline_stats_df[['well_id', 'drug', 'concentration', 'plate_id']].copy()
quality_df['has_baseline'] = baseline_stats_df['baseline_valid']
quality_df['baseline_stable'] = baseline_stats_df['baseline_stable']
quality_df['baseline_cv'] = baseline_stats_df['cv_o2']
quality_df['baseline_points'] = baseline_stats_df['n_points']
quality_df['baseline_trend_significant'] = baseline_stats_df['baseline_trend_pvalue'] < 0.05
quality_df['baseline_slope'] = baseline_stats_df['baseline_slope']

# Overall quality score
quality_df['baseline_quality_score'] = (
    quality_df['has_baseline'].astype(int) * 0.4 +
    quality_df['baseline_stable'].astype(int) * 0.3 +
    (~quality_df['baseline_trend_significant']).astype(int) * 0.2 +
    (quality_df['baseline_points'] > 30).astype(int) * 0.1
)

# ========== VISUALIZATION FUNCTIONS ==========

def visualize_baseline_overview(baseline_stats_df, save_path):
    """Overview of baseline characteristics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline Period Overview', fontsize=16, fontweight='bold')
    
    # 1. Valid vs invalid baselines
    ax = axes[0, 0]
    valid_counts = baseline_stats_df['baseline_valid'].value_counts()
    ax.pie(valid_counts.values, labels=['Valid', 'Invalid'], autopct='%1.1f%%', 
           colors=['lightgreen', 'lightcoral'])
    ax.set_title('A. Baseline Validity', fontsize=12, loc='left')
    
    # 2. Baseline stability (CV distribution)
    ax = axes[0, 1]
    valid_baselines = baseline_stats_df[baseline_stats_df['baseline_valid']]
    ax.hist(valid_baselines['cv_o2'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(STABILITY_CV_THRESHOLD, color='red', linestyle='--', 
               label=f'Stability threshold (CV={STABILITY_CV_THRESHOLD})')
    ax.set_xlabel('Coefficient of Variation')
    ax.set_ylabel('Number of Wells')
    ax.set_title('B. Baseline Stability Distribution', fontsize=12, loc='left')
    ax.legend()
    
    # 3. Baseline duration
    ax = axes[0, 2]
    ax.hist(valid_baselines['duration_hours'], bins=20, edgecolor='black', 
            alpha=0.7, color='lightgreen')
    ax.axvline(BASELINE_HOURS, color='red', linestyle='--', 
               label=f'Target duration ({BASELINE_HOURS}h)')
    ax.set_xlabel('Baseline Duration (hours)')
    ax.set_ylabel('Number of Wells')
    ax.set_title('C. Baseline Duration', fontsize=12, loc='left')
    ax.legend()
    
    # 4. Baseline oxygen levels by drug
    ax = axes[1, 0]
    top_drugs = valid_baselines['drug'].value_counts().head(15).index
    drug_baseline_o2 = valid_baselines[valid_baselines['drug'].isin(top_drugs)].groupby('drug')['mean_o2'].mean().sort_values()
    
    ax.barh(range(len(drug_baseline_o2)), drug_baseline_o2.values, color='steelblue')
    ax.set_yticks(range(len(drug_baseline_o2)))
    ax.set_yticklabels([d[:20] for d in drug_baseline_o2.index])
    ax.set_xlabel('Mean Baseline O₂')
    ax.set_title('D. Baseline O₂ by Drug (Top 15)', fontsize=12, loc='left')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Baseline trend analysis
    ax = axes[1, 1]
    slopes = valid_baselines['baseline_slope'].dropna()
    ax.hist(slopes, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(0, color='red', linestyle='--', label='No trend')
    ax.set_xlabel('Baseline Slope (O₂/hour)')
    ax.set_ylabel('Number of Wells')
    ax.set_title('E. Baseline Trends', fontsize=12, loc='left')
    ax.legend()
    
    # Add statistics
    increasing = (slopes > 0.1).sum()
    decreasing = (slopes < -0.1).sum()
    stable = ((slopes >= -0.1) & (slopes <= 0.1)).sum()
    ax.text(0.02, 0.95, f'Increasing: {increasing}\nStable: {stable}\nDecreasing: {decreasing}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Quality score distribution
    ax = axes[1, 2]
    ax.hist(quality_df['baseline_quality_score'], bins=20, edgecolor='black', 
            alpha=0.7, color='purple')
    ax.set_xlabel('Baseline Quality Score')
    ax.set_ylabel('Number of Wells')
    ax.set_title('F. Overall Baseline Quality', fontsize=12, loc='left')
    ax.axvline(quality_df['baseline_quality_score'].median(), color='red', 
               linestyle='--', label=f'Median: {quality_df["baseline_quality_score"].median():.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_baseline_examples(df, baseline_stats_df, save_path):
    """Show examples of different baseline types"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline Period Examples', fontsize=16, fontweight='bold')
    
    # Find examples of different baseline types
    valid_stats = baseline_stats_df[baseline_stats_df['baseline_valid']]
    
    examples = [
        ('Stable Baseline', valid_stats[valid_stats['cv_o2'] < 0.05].iloc[0] if len(valid_stats[valid_stats['cv_o2'] < 0.05]) > 0 else None),
        ('High Noise', valid_stats[valid_stats['cv_o2'] > 0.15].iloc[0] if len(valid_stats[valid_stats['cv_o2'] > 0.15]) > 0 else None),
        ('Increasing Trend', valid_stats[valid_stats['baseline_slope'] > 0.5].iloc[0] if len(valid_stats[valid_stats['baseline_slope'] > 0.5]) > 0 else None),
        ('Decreasing Trend', valid_stats[valid_stats['baseline_slope'] < -0.5].iloc[0] if len(valid_stats[valid_stats['baseline_slope'] < -0.5]) > 0 else None),
        ('Short Baseline', baseline_stats_df[baseline_stats_df['n_points'] < 15].iloc[0] if len(baseline_stats_df[baseline_stats_df['n_points'] < 15]) > 0 else None),
        ('Ideal Baseline', valid_stats[(valid_stats['cv_o2'] < 0.05) & (valid_stats['baseline_slope'].abs() < 0.1)].iloc[0] if len(valid_stats[(valid_stats['cv_o2'] < 0.05) & (valid_stats['baseline_slope'].abs() < 0.1)]) > 0 else None)
    ]
    
    for idx, (title, example) in enumerate(examples):
        ax = axes[idx // 3, idx % 3]
        
        if example is not None:
            # Get the time series data
            well_data = df[df['well_id'] == example['well_id']]
            baseline_data = well_data[well_data['elapsed_hours'] <= BASELINE_HOURS]
            
            # Plot baseline period
            ax.plot(baseline_data['elapsed_hours'], baseline_data['oxygen'], 
                   'b-', alpha=0.8, linewidth=2, label='Baseline')
            
            # Plot rest of data in gray
            post_baseline = well_data[well_data['elapsed_hours'] > BASELINE_HOURS]
            if len(post_baseline) > 0:
                ax.plot(post_baseline['elapsed_hours'], post_baseline['oxygen'], 
                       'gray', alpha=0.5, linewidth=1, label='Post-baseline')
            
            # Add baseline statistics
            ax.axhline(example['mean_o2'], color='red', linestyle='--', alpha=0.5, 
                      label=f'Mean: {example["mean_o2"]:.1f}')
            
            # Add trend line if significant
            if not pd.isna(example['baseline_slope']):
                x = baseline_data['elapsed_hours'].values
                y = example['baseline_intercept'] + example['baseline_slope'] * x
                ax.plot(x, y, 'g--', alpha=0.7, label=f'Trend: {example["baseline_slope"]:.3f}/h')
            
            # Shade baseline period
            ax.axvspan(0, BASELINE_HOURS, alpha=0.1, color='blue')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Oxygen')
            ax.set_title(f'{title}\n{example["drug"]} - CV: {example["cv_o2"]:.3f}', 
                        fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No example found', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_baseline_catch22_features(baseline_catch22_df, save_path):
    """Visualize catch22 features from baseline period"""
    if len(baseline_catch22_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline catch22 Feature Analysis', fontsize=16, fontweight='bold')
    
    # Get feature columns
    feature_cols = [col for col in baseline_catch22_df.columns 
                   if col.startswith('baseline_') and col not in ['baseline_hours']]
    
    # 1. Feature correlation matrix
    ax = axes[0, 0]
    if len(feature_cols) > 5:
        # Select subset of features for visualization
        selected_features = feature_cols[:10]
        corr_matrix = baseline_catch22_df[selected_features].corr()
        
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, 
                   ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_xticklabels([f.replace('baseline_', '')[:15] for f in selected_features], 
                          rotation=45, ha='right')
        ax.set_yticklabels([f.replace('baseline_', '')[:15] for f in selected_features], 
                          rotation=0)
        ax.set_title('A. Feature Correlations (Top 10)', fontsize=12, loc='left')
    
    # 2. Feature distributions
    ax = axes[0, 1]
    # Calculate coefficient of variation for each feature
    feature_cvs = []
    for feat in feature_cols[:20]:  # Top 20 features
        values = baseline_catch22_df[feat].dropna()
        if len(values) > 10 and values.std() > 0:
            cv = values.std() / values.mean() if values.mean() != 0 else 0
            feature_cvs.append({'feature': feat.replace('baseline_', ''), 'cv': abs(cv)})
    
    if feature_cvs:
        cv_df = pd.DataFrame(feature_cvs).sort_values('cv', ascending=False).head(15)
        ax.barh(range(len(cv_df)), cv_df['cv'], color='coral', alpha=0.7)
        ax.set_yticks(range(len(cv_df)))
        ax.set_yticklabels([f[:20] for f in cv_df['feature']])
        ax.set_xlabel('Coefficient of Variation')
        ax.set_title('B. Most Variable Features', fontsize=12, loc='left')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Drug clustering based on baseline features
    ax = axes[1, 0]
    # Aggregate features by drug
    drug_features = baseline_catch22_df.groupby('drug')[feature_cols].mean()
    
    if len(drug_features) > 10:
        # PCA on drug features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(drug_features.fillna(0))
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=60)
        
        # Annotate some points
        for i, drug in enumerate(drug_features.index[:20]):
            if i % 3 == 0:  # Show every 3rd drug to avoid crowding
                ax.annotate(drug[:15], (X_pca[i, 0], X_pca[i, 1]), 
                          fontsize=8, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('C. Drug Clustering by Baseline Features', fontsize=12, loc='left')
        ax.grid(True, alpha=0.3)
    
    # 4. Feature importance for baseline stability
    ax = axes[1, 1]
    # Correlate features with baseline CV (stability metric)
    correlations = []
    merged_df = baseline_catch22_df.merge(
        baseline_stats_df[['well_id', 'cv_o2']], on='well_id'
    )
    
    for feat in feature_cols[:20]:
        if feat in merged_df.columns:
            corr = merged_df[[feat, 'cv_o2']].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append({'feature': feat.replace('baseline_', ''), 
                                   'correlation': abs(corr)})
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False).head(10)
        ax.barh(range(len(corr_df)), corr_df['correlation'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(corr_df)))
        ax.set_yticklabels([f[:20] for f in corr_df['feature']])
        ax.set_xlabel('|Correlation| with Baseline CV')
        ax.set_title('D. Features Predicting Baseline Stability', fontsize=12, loc='left')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_baseline_sax_patterns(baseline_sax_df, save_path):
    """Visualize SAX patterns from baseline period"""
    if len(baseline_sax_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline SAX Pattern Analysis', fontsize=16, fontweight='bold')
    
    for idx, config in enumerate(SAX_CONFIGS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        prefix = f"baseline_sax_{config['name']}"
        
        # Get pattern strings
        if f"{prefix}_string" in baseline_sax_df.columns:
            patterns = baseline_sax_df[f"{prefix}_string"].dropna()
            
            if len(patterns) > 0:
                # Count pattern frequencies
                pattern_counts = patterns.value_counts().head(15)
                
                ax.barh(range(len(pattern_counts)), pattern_counts.values, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(pattern_counts))))
                ax.set_yticks(range(len(pattern_counts)))
                ax.set_yticklabels(pattern_counts.index)
                ax.set_xlabel('Frequency')
                ax.set_title(f'{config["name"].title()} Level Patterns (Top 15)', 
                           fontsize=11)
                ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'No patterns found', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{config["name"].title()} Level', fontsize=11)
    
    # SAX feature statistics
    ax = axes[1, 0]
    # Entropy distribution
    entropy_data = []
    for config in SAX_CONFIGS:
        col = f"baseline_sax_{config['name']}_entropy"
        if col in baseline_sax_df.columns:
            values = baseline_sax_df[col].dropna()
            if len(values) > 0:
                entropy_data.append(values)
    
    if entropy_data:
        ax.boxplot(entropy_data, labels=[c['name'] for c in SAX_CONFIGS[:len(entropy_data)]])
        ax.set_ylabel('Entropy')
        ax.set_title('Pattern Entropy by Resolution', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Transition rates
    ax = axes[1, 1]
    transition_data = []
    for config in SAX_CONFIGS:
        col = f"baseline_sax_{config['name']}_transitions"
        if col in baseline_sax_df.columns:
            values = baseline_sax_df[col].dropna()
            if len(values) > 0:
                transition_data.append(values)
    
    if transition_data:
        ax.boxplot(transition_data, labels=[c['name'] for c in SAX_CONFIGS[:len(transition_data)]])
        ax.set_ylabel('Transition Rate')
        ax.set_title('Pattern Transitions by Resolution', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Drug clustering by SAX features
    ax = axes[1, 2]
    # Collect all SAX features
    sax_feature_cols = [col for col in baseline_sax_df.columns 
                       if col.startswith('baseline_sax_') and 
                       not col.endswith('_string')]
    
    if len(sax_feature_cols) > 5:
        drug_sax_features = baseline_sax_df.groupby('drug')[sax_feature_cols].mean()
        
        if len(drug_sax_features) > 10:
            # UMAP projection
            try:
                reducer = umap.UMAP(n_neighbors=min(15, len(drug_sax_features)-1), 
                                   random_state=42)
                X_scaled = StandardScaler().fit_transform(drug_sax_features.fillna(0))
                X_umap = reducer.fit_transform(X_scaled)
                
                scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7, s=60)
                
                # Annotate some drugs
                for i, drug in enumerate(drug_sax_features.index[:20]):
                    if i % 3 == 0:
                        ax.annotate(drug[:15], (X_umap[i, 0], X_umap[i, 1]), 
                                  fontsize=8, alpha=0.7)
                
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title('Drug Clustering by SAX Features', fontsize=11)
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'UMAP failed', ha='center', va='center', 
                       transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_baseline_quality_heatmap(quality_df, baseline_stats_df, save_path):
    """Create heatmap of baseline quality by drug and plate"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Baseline Quality Assessment', fontsize=16, fontweight='bold')
    
    # 1. Quality score by drug
    drug_quality = quality_df.groupby('drug')['baseline_quality_score'].agg(['mean', 'std', 'count'])
    drug_quality = drug_quality[drug_quality['count'] >= 4].sort_values('mean', ascending=False).head(30)
    
    if len(drug_quality) > 0:
        # Create heatmap data
        y_pos = np.arange(len(drug_quality))
        
        ax1.barh(y_pos, drug_quality['mean'], xerr=drug_quality['std'], 
                color=plt.cm.RdYlGn(drug_quality['mean']), 
                alpha=0.7, capsize=3)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([d[:25] for d in drug_quality.index])
        ax1.set_xlabel('Mean Baseline Quality Score')
        ax1.set_title('A. Baseline Quality by Drug (Top 30)', fontsize=12, loc='left')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, 1)
        
        # Add sample size
        for i, (_, row) in enumerate(drug_quality.iterrows()):
            ax1.text(row['mean'] + row['std'] + 0.02, i, f'n={int(row["count"])}', 
                    va='center', fontsize=8)
    
    # 2. Baseline characteristics correlation
    ax = ax2
    # Select numeric columns for correlation
    corr_cols = ['baseline_cv', 'baseline_points', 'baseline_slope', 
                 'mean_o2', 'std_o2', 'baseline_quality_score']
    corr_data = baseline_stats_df[[col for col in corr_cols if col in baseline_stats_df.columns]]
    
    if len(corr_data.columns) > 2:
        corr_matrix = corr_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax,
                   xticklabels=[c.replace('baseline_', '').replace('_', ' ').title() 
                               for c in corr_matrix.columns],
                   yticklabels=[c.replace('baseline_', '').replace('_', ' ').title() 
                               for c in corr_matrix.columns])
        ax.set_title('B. Baseline Characteristic Correlations', fontsize=12, loc='left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========== GENERATE VISUALIZATIONS ==========

print("\n📊 Creating visualizations...")

# 1. Baseline overview
print("   Creating baseline overview...")
visualize_baseline_overview(baseline_stats_df, fig_dir / 'baseline_overview.png')

# 2. Baseline examples
print("   Creating baseline examples...")
visualize_baseline_examples(df, baseline_stats_df, fig_dir / 'baseline_examples.png')

# 3. catch22 features
if len(baseline_catch22_df) > 0:
    print("   Creating catch22 feature analysis...")
    visualize_baseline_catch22_features(baseline_catch22_df, fig_dir / 'baseline_catch22_features.png')

# 4. SAX patterns
if len(baseline_sax_df) > 0:
    print("   Creating SAX pattern analysis...")
    visualize_baseline_sax_patterns(baseline_sax_df, fig_dir / 'baseline_sax_patterns.png')

# 5. Quality heatmap
print("   Creating quality assessment heatmap...")
visualize_baseline_quality_heatmap(quality_df, baseline_stats_df, fig_dir / 'baseline_quality_heatmap.png')

print(f"\n✅ All visualizations saved to: {fig_dir}/")

# ========== SAVE RESULTS ==========

print("\n💾 Saving baseline features...")

# Save all DataFrames
baseline_stats_df.to_parquet(results_dir / 'baseline_summary_statistics.parquet', index=False)
print(f"   Summary statistics: {results_dir / 'baseline_summary_statistics.parquet'}")

if len(baseline_catch22_df) > 0:
    baseline_catch22_df.to_parquet(results_dir / 'baseline_catch22_features.parquet', index=False)
    print(f"   catch22 features: {results_dir / 'baseline_catch22_features.parquet'}")

if len(baseline_sax_df) > 0:
    baseline_sax_df.to_parquet(results_dir / 'baseline_sax_features.parquet', index=False)
    print(f"   SAX features: {results_dir / 'baseline_sax_features.parquet'}")

quality_df.to_parquet(results_dir / 'baseline_quality_assessment.parquet', index=False)
print(f"   Quality assessment: {results_dir / 'baseline_quality_assessment.parquet'}")

# Summary statistics
print(f"\n📈 BASELINE FEATURE SUMMARY:")
print(f"   Mean baseline O₂: {baseline_stats_df['mean_o2'].mean():.1f} ± {baseline_stats_df['mean_o2'].std():.1f}")
print(f"   Mean baseline CV: {baseline_stats_df['cv_o2'].mean():.3f} ± {baseline_stats_df['cv_o2'].std():.3f}")
print(f"   Drugs with stable baselines: {quality_df.groupby('drug')['baseline_stable'].mean().sum():.0f}/{quality_df['drug'].nunique()}")
print(f"   Plates with >80% valid baselines: {(baseline_stats_df.groupby('plate_id')['baseline_valid'].mean() > 0.8).sum()}")

print("\n✅ Baseline feature extraction complete!")
print("\n🎯 Next steps:")
print("   1. Use baseline features for normalization")
print("   2. Include baseline quality in predictive models")
print("   3. Filter analyses to stable baselines only")
print("   4. Compare drug effects relative to baseline")