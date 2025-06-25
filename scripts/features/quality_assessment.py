#!/usr/bin/env python3
"""
Quality Assessment Flags for Oxygen Consumption Data

PURPOSE:
    Systematically assesses data quality and generates quality flags for each well.
    Rather than filtering out problematic data, this creates quality features that
    can be embedded in analyses, allowing models to learn from quality patterns.

METHODOLOGY:
    Evaluates 7 quality dimensions:
    1. Low points: Insufficient measurements (<200 timepoints)
    2. High noise: Excessive variability (rolling CV > 0.3)
    3. Sensor drift: High correlation between O2 and time (|r| > 0.8)
    4. Baseline instability: High variation in first 48h (CV > 0.1)
    5. Data gaps: Missing measurements > 2x expected interval
    6. Spike outliers: Media change responses > 3 MAD from baseline
    7. Replicate discord: High variability across 4 replicates (CV > 0.5)
    
    Generates overall quality score (0-1) and binary quality flags.

INPUTS:
    - Database connection via DATABASE_URL environment variable
    - Queries raw oxygen consumption data
    - Optional: results/data/improved_media_change_events.parquet
      For spike outlier analysis around media changes

OUTPUTS:
    - results/data/quality_assessment_flags.parquet
      Complete quality assessment for all wells with detailed metrics
    - results/data/quality_aware_features.parquet  
      Binary quality flags formatted for feature embedding

REQUIREMENTS:
    - numpy, pandas, scipy, tqdm
    - Database connection with oxygen consumption data
    - scipy.stats.pearsonr for drift detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)
fig_dir = project_root / "results" / "figures" / "quality_assessment"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("QUALITY ASSESSMENT FLAG EXTRACTION")
print("=" * 80)

class QualityAssessor:
    """
    Assess time series quality and generate quality flags
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'min_points': 200,              # Minimum timepoints needed
            'high_noise_cv': 0.3,           # CV threshold for noise
            'drift_correlation': 0.8,       # Correlation threshold for drift
            'replicate_discord_cv': 0.5,    # CV threshold across replicates
            'baseline_unstable_cv': 0.1,    # CV threshold for baseline
            'spike_outlier_factor': 3       # Factor for spike outliers
        }
    
    def assess_low_points(self, time_series):
        """Flag if too few measurement points"""
        n_points = len(time_series)
        return {
            'low_points': n_points < self.quality_thresholds['min_points'],
            'n_points': n_points,
            'points_deficit': max(0, self.quality_thresholds['min_points'] - n_points)
        }
    
    def assess_high_noise(self, time_series, window_hours=24):
        """Flag if high noise level (rolling CV)"""
        if len(time_series) < 10:
            return {
                'high_noise': True,
                'mean_cv': np.nan,
                'max_cv': np.nan,
                'noise_windows': 0
            }
        
        # Calculate rolling CV
        window_size = max(15, int(window_hours * 0.625))  # ~15 points for 24h
        rolling_mean = pd.Series(time_series).rolling(window_size, center=True).mean()
        rolling_std = pd.Series(time_series).rolling(window_size, center=True).std()
        
        # Calculate CV where mean is not zero
        valid_mask = (rolling_mean.notna()) & (rolling_mean != 0)
        if valid_mask.sum() == 0:
            return {
                'high_noise': True,
                'mean_cv': np.nan,
                'max_cv': np.nan,
                'noise_windows': 0
            }
        
        rolling_cv = rolling_std[valid_mask] / np.abs(rolling_mean[valid_mask])
        
        mean_cv = rolling_cv.mean()
        max_cv = rolling_cv.max()
        noise_windows = (rolling_cv > self.quality_thresholds['high_noise_cv']).sum()
        
        return {
            'high_noise': mean_cv > self.quality_thresholds['high_noise_cv'],
            'mean_cv': mean_cv,
            'max_cv': max_cv,
            'noise_windows': noise_windows,
            'noise_fraction': noise_windows / len(rolling_cv) if len(rolling_cv) > 0 else 0
        }
    
    def assess_sensor_drift(self, time_series, time_hours):
        """Flag if sensor drift detected (high correlation with time)"""
        if len(time_series) < 10 or len(time_hours) < 10:
            return {
                'sensor_drift': False,
                'drift_correlation': 0,
                'drift_slope': 0
            }
        
        # Calculate correlation between O2 and elapsed time
        try:
            correlation, p_value = pearsonr(time_hours, time_series)
            
            # Fit linear trend
            slope = np.polyfit(time_hours, time_series, 1)[0]
            
            return {
                'sensor_drift': abs(correlation) > self.quality_thresholds['drift_correlation'],
                'drift_correlation': correlation,
                'drift_p_value': p_value,
                'drift_slope': slope,
                'drift_magnitude': abs(slope) * (time_hours[-1] - time_hours[0])
            }
        except:
            return {
                'sensor_drift': False,
                'drift_correlation': 0,
                'drift_slope': 0
            }
    
    def assess_baseline_stability(self, time_series, time_hours, baseline_hours=48):
        """Flag if baseline period is unstable"""
        # Get baseline period
        baseline_mask = time_hours <= baseline_hours
        baseline_data = time_series[baseline_mask]
        
        if len(baseline_data) < 20:
            return {
                'baseline_unstable': True,
                'baseline_cv': np.nan,
                'baseline_mean': np.nan,
                'baseline_points': len(baseline_data)
            }
        
        baseline_mean = baseline_data.mean()
        baseline_std = baseline_data.std()
        baseline_cv = baseline_std / abs(baseline_mean) if baseline_mean != 0 else np.inf
        
        # Also check for trend in baseline
        baseline_time = time_hours[baseline_mask]
        baseline_slope = np.polyfit(baseline_time, baseline_data, 1)[0] if len(baseline_time) > 1 else 0
        
        return {
            'baseline_unstable': baseline_cv > self.quality_thresholds['baseline_unstable_cv'],
            'baseline_cv': baseline_cv,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'baseline_slope': baseline_slope,
            'baseline_points': len(baseline_data)
        }
    
    def assess_replicate_discord(self, replicate_values):
        """Flag if high discord across replicates"""
        if len(replicate_values) < 2:
            return {
                'replicate_discord': False,
                'replicate_cv': 0,
                'n_replicates': len(replicate_values)
            }
        
        rep_mean = np.mean(replicate_values)
        rep_std = np.std(replicate_values)
        rep_cv = rep_std / abs(rep_mean) if rep_mean != 0 else np.inf
        
        return {
            'replicate_discord': rep_cv > self.quality_thresholds['replicate_discord_cv'],
            'replicate_cv': rep_cv,
            'replicate_mean': rep_mean,
            'replicate_std': rep_std,
            'replicate_range': np.max(replicate_values) - np.min(replicate_values),
            'n_replicates': len(replicate_values)
        }
    
    def assess_spike_outliers(self, time_series, spike_heights):
        """Flag if spike responses are extreme outliers"""
        if len(spike_heights) == 0:
            return {
                'spike_outlier': False,
                'n_spikes': 0,
                'max_spike_z': 0
            }
        
        # Calculate baseline statistics (excluding spikes)
        baseline_mean = np.median(time_series)  # Use median for robustness
        baseline_mad = np.median(np.abs(time_series - baseline_mean))
        
        # Calculate z-scores for spikes using MAD
        spike_z_scores = np.abs(spike_heights - baseline_mean) / (1.4826 * baseline_mad) if baseline_mad > 0 else np.zeros_like(spike_heights)
        
        max_z = np.max(spike_z_scores)
        outlier_spikes = (spike_z_scores > self.quality_thresholds['spike_outlier_factor']).sum()
        
        return {
            'spike_outlier': max_z > self.quality_thresholds['spike_outlier_factor'],
            'n_spikes': len(spike_heights),
            'max_spike_z': max_z,
            'outlier_spikes': outlier_spikes,
            'spike_outlier_fraction': outlier_spikes / len(spike_heights) if len(spike_heights) > 0 else 0
        }
    
    def assess_data_gaps(self, time_hours):
        """Detect gaps in time series"""
        if len(time_hours) < 2:
            return {
                'has_gaps': False,
                'n_gaps': 0,
                'max_gap_hours': 0
            }
        
        # Calculate time differences
        time_diffs = np.diff(time_hours)
        
        # Expected sampling interval (median)
        expected_interval = np.median(time_diffs)
        
        # Gaps are intervals > 2x expected
        gaps = time_diffs[time_diffs > 2 * expected_interval]
        
        return {
            'has_gaps': len(gaps) > 0,
            'n_gaps': len(gaps),
            'max_gap_hours': np.max(gaps) if len(gaps) > 0 else 0,
            'total_gap_hours': np.sum(gaps) if len(gaps) > 0 else 0,
            'gap_fraction': np.sum(gaps) / (time_hours[-1] - time_hours[0]) if len(gaps) > 0 and time_hours[-1] != time_hours[0] else 0
        }

def visualize_quality_flag_distribution(quality_df, save_path):
    """Visualize distribution of quality flags across all wells"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Quality flag prevalence
    ax = axes[0]
    flags = ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable', 
             'has_gaps', 'spike_outlier', 'replicate_discord']
    
    flag_counts = []
    flag_labels = []
    for flag in flags:
        if flag in quality_df.columns:
            count = quality_df[flag].sum()
            percentage = count / len(quality_df) * 100
            flag_counts.append(count)
            flag_labels.append(f'{flag.replace("_", " ").title()}\n({percentage:.1f}%)')
    
    bars = ax.bar(range(len(flag_counts)), flag_counts, color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(flag_counts))))
    ax.set_xticks(range(len(flag_counts)))
    ax.set_xticklabels(flag_labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Wells')
    ax.set_title('Quality Flag Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, flag_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom')
    
    # Quality score distribution
    ax = axes[1]
    quality_scores = quality_df['quality_score'].values
    
    # Create bins for histogram
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(quality_scores, bins=bins)
    
    # Plot histogram
    bars = ax.bar(bins[:-1], counts, width=0.09, align='edge', 
                   color=plt.cm.RdYlGn(bins[:-1]), edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Quality Score')
    ax.set_ylabel('Number of Wells')
    ax.set_title('Overall Quality Score Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    ax.axvline(quality_scores.mean(), color='red', linestyle='--', 
               label=f'Mean: {quality_scores.mean():.3f}')
    ax.axvline(np.median(quality_scores), color='blue', linestyle='--', 
               label=f'Median: {np.median(quality_scores):.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_quality_examples(df, quality_df, save_path):
    """Show examples of each quality issue type"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    quality_issues = [
        ('high_noise', 'High Noise', 'red'),
        ('sensor_drift', 'Sensor Drift', 'orange'),
        ('baseline_unstable', 'Unstable Baseline', 'purple'),
        ('has_gaps', 'Data Gaps', 'brown'),
        ('spike_outlier', 'Spike Outliers', 'magenta'),
        ('low_points', 'Low Points', 'gray')
    ]
    
    # Find example wells for each issue
    for idx, (flag, title, color) in enumerate(quality_issues):
        ax = axes[idx]
        
        # Find a well with this specific issue
        flagged_wells = quality_df[quality_df[flag] == True]['well_id'].values
        
        if len(flagged_wells) > 0:
            # Get data for example well
            example_well = flagged_wells[0]
            well_data = df[df['well_id'] == example_well].sort_values('elapsed_hours')
            
            # Plot time series
            ax.plot(well_data['elapsed_hours'], well_data['o2'], 'b-', alpha=0.7, linewidth=1)
            
            # Add specific annotations for each issue type
            if flag == 'high_noise':
                # Show rolling CV
                window_size = 15
                rolling_mean = well_data['o2'].rolling(window_size, center=True).mean()
                rolling_std = well_data['o2'].rolling(window_size, center=True).std()
                upper = rolling_mean + 2*rolling_std
                lower = rolling_mean - 2*rolling_std
                ax.fill_between(well_data['elapsed_hours'], lower, upper, alpha=0.3, color=color)
                
            elif flag == 'sensor_drift':
                # Show trend line
                z = np.polyfit(well_data['elapsed_hours'], well_data['o2'], 1)
                p = np.poly1d(z)
                ax.plot(well_data['elapsed_hours'], p(well_data['elapsed_hours']), 
                       '--', color=color, linewidth=2, label=f'Drift: {z[0]:.3f}/hr')
                ax.legend()
                
            elif flag == 'baseline_unstable':
                # Highlight baseline period
                baseline_mask = well_data['elapsed_hours'] <= 48
                ax.axvspan(0, 48, alpha=0.2, color=color)
                ax.axhline(well_data[baseline_mask]['o2'].mean(), 
                          color=color, linestyle='--', alpha=0.5)
                
            elif flag == 'has_gaps':
                # Mark gaps
                time_diffs = np.diff(well_data['elapsed_hours'])
                gap_indices = np.where(time_diffs > 2 * np.median(time_diffs))[0]
                for gap_idx in gap_indices:
                    gap_start = well_data['elapsed_hours'].iloc[gap_idx]
                    gap_end = well_data['elapsed_hours'].iloc[gap_idx + 1]
                    ax.axvspan(gap_start, gap_end, alpha=0.3, color=color)
            
            # Get quality metrics for this well
            well_quality = quality_df[quality_df['well_id'] == example_well].iloc[0]
            drug = well_quality['drug']
            score = well_quality['quality_score']
            
            ax.set_title(f'{title} Example - {drug}\nQuality Score: {score:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Elapsed Hours')
            ax.set_ylabel('O₂ (%)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No wells with\n{title}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    # Add good quality example
    ax = axes[6]
    good_wells = quality_df[quality_df['is_high_quality'] == True]['well_id'].values
    if len(good_wells) > 0:
        example_well = good_wells[0]
        well_data = df[df['well_id'] == example_well].sort_values('elapsed_hours')
        ax.plot(well_data['elapsed_hours'], well_data['o2'], 'g-', alpha=0.8, linewidth=1)
        
        well_quality = quality_df[quality_df['well_id'] == example_well].iloc[0]
        drug = well_quality['drug']
        ax.set_title(f'High Quality Example - {drug}\nQuality Score: 1.000', 
                    fontweight='bold', color='green')
        ax.set_xlabel('Elapsed Hours')
        ax.set_ylabel('O₂ (%)')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[7])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_quality_by_drug_heatmap(quality_df, save_path):
    """Create heatmap showing quality patterns across drugs"""
    # Aggregate by drug
    flags = ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable', 
             'has_gaps', 'spike_outlier', 'replicate_discord']
    
    drug_quality = quality_df.groupby('drug')[flags].mean()
    
    # Sort by overall quality
    drug_quality['mean_issues'] = drug_quality.mean(axis=1)
    drug_quality = drug_quality.sort_values('mean_issues', ascending=False)
    drug_quality = drug_quality.drop('mean_issues', axis=1)
    
    # Select top 30 drugs with most issues
    drug_quality = drug_quality.head(30)
    
    # Create heatmap
    plt.figure(figsize=(10, 12))
    
    # Create custom colormap
    colors = ['white', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
    
    sns.heatmap(drug_quality, cmap=cmap, vmin=0, vmax=1, 
                cbar_kws={'label': 'Fraction of Wells with Issue'},
                linewidths=0.5, linecolor='gray',
                annot=True, fmt='.2f', annot_kws={'size': 8})
    
    plt.title('Quality Issues by Drug (Top 30 Most Problematic)', fontsize=14, fontweight='bold')
    plt.xlabel('Quality Flag')
    plt.ylabel('Drug')
    plt.xticks(rotation=45, ha='right')
    
    # Clean up flag names for display
    ax = plt.gca()
    xlabels = [label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_quality_correlation_matrix(quality_df, save_path):
    """Show correlations between different quality issues"""
    flags = ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable', 
             'has_gaps', 'spike_outlier', 'replicate_discord']
    
    # Calculate correlation matrix
    flag_data = quality_df[flags].astype(int)
    corr_matrix = flag_data.corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-0.5, vmax=0.5, annot=True, fmt='.2f')
    
    # Clean up labels
    ax = plt.gca()
    labels = [label.replace('_', ' ').title() for label in flags]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.title('Quality Flag Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_replicate_consistency(quality_df, save_path):
    """Visualize replicate consistency across drugs"""
    # Get drugs with replicate discord data
    replicate_data = quality_df[quality_df['replicate_cv'].notna()].copy()
    
    if len(replicate_data) == 0:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Replicate CV distribution
    ax = axes[0]
    
    # Group by drug and get mean replicate CV
    drug_replicate_cv = replicate_data.groupby('drug')['replicate_cv'].agg(['mean', 'std', 'count'])
    drug_replicate_cv = drug_replicate_cv[drug_replicate_cv['count'] >= 4]  # At least 4 measurements
    drug_replicate_cv = drug_replicate_cv.sort_values('mean', ascending=False).head(20)
    
    if len(drug_replicate_cv) > 0:
        x = range(len(drug_replicate_cv))
        ax.bar(x, drug_replicate_cv['mean'], yerr=drug_replicate_cv['std'], 
               capsize=5, color=plt.cm.RdYlBu_r(drug_replicate_cv['mean'] / drug_replicate_cv['mean'].max()))
        
        ax.set_xticks(x)
        ax.set_xticklabels(drug_replicate_cv.index, rotation=45, ha='right')
        ax.set_ylabel('Mean Replicate CV')
        ax.set_title('Replicate Consistency by Drug (Top 20 Most Variable)', fontsize=14, fontweight='bold')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Discord Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Concentration effect on replicate consistency
    ax = axes[1]
    
    # Bin concentrations
    replicate_data['conc_bin'] = pd.qcut(replicate_data['concentration'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Create violin plot
    if 'conc_bin' in replicate_data.columns:
        sns.violinplot(data=replicate_data, x='conc_bin', y='replicate_cv', ax=ax)
        ax.set_xlabel('Concentration Range')
        ax.set_ylabel('Replicate CV')
        ax.set_title('Replicate Consistency vs Concentration', fontsize=14, fontweight='bold')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Discord Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_quality_summary_dashboard(quality_df, save_path):
    """Create a summary dashboard of quality metrics"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall quality pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    quality_bins = pd.cut(quality_df['quality_score'], bins=[0, 0.4, 0.7, 0.9, 1.0], 
                         labels=['Poor', 'Fair', 'Good', 'Excellent'])
    quality_counts = quality_bins.value_counts()
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    wedges, texts, autotexts = ax1.pie(quality_counts.values, labels=quality_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Overall Quality Distribution', fontweight='bold')
    
    # 2. Flag frequency bar chart
    ax2 = fig.add_subplot(gs[0, 1:])
    flags = ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable', 
             'has_gaps', 'spike_outlier', 'replicate_discord']
    flag_percentages = [(quality_df[flag].sum() / len(quality_df) * 100) for flag in flags]
    
    bars = ax2.barh(range(len(flags)), flag_percentages, 
                     color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(flags))))
    ax2.set_yticks(range(len(flags)))
    ax2.set_yticklabels([f.replace('_', ' ').title() for f in flags])
    ax2.set_xlabel('Percentage of Wells (%)')
    ax2.set_title('Quality Flag Prevalence', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for bar, pct in zip(bars, flag_percentages):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center')
    
    # 3. Quality score by concentration
    ax3 = fig.add_subplot(gs[1, :])
    
    # Bin concentrations
    conc_bins = pd.qcut(quality_df['concentration'], q=10, duplicates='drop')
    quality_by_conc = quality_df.groupby(conc_bins)['quality_score'].agg(['mean', 'std', 'count'])
    
    x = range(len(quality_by_conc))
    ax3.errorbar(x, quality_by_conc['mean'], yerr=quality_by_conc['std'], 
                 marker='o', capsize=5, capthick=2, linewidth=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{interval.left:.1f}-{interval.right:.1f}' 
                        for interval in quality_by_conc.index], rotation=45, ha='right')
    ax3.set_xlabel('Concentration Bins (μM)')
    ax3.set_ylabel('Mean Quality Score')
    ax3.set_title('Quality Score vs Concentration', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. Number of flags distribution
    ax4 = fig.add_subplot(gs[2, 0])
    flag_counts = quality_df['n_quality_flags'].value_counts().sort_index()
    
    bars = ax4.bar(flag_counts.index, flag_counts.values, 
                    color=plt.cm.RdYlGn_r(flag_counts.index / flag_counts.index.max()))
    ax4.set_xlabel('Number of Quality Flags')
    ax4.set_ylabel('Number of Wells')
    ax4.set_title('Distribution of Quality Flag Counts', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, flag_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom')
    
    # 5. Top/bottom drugs table
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Get top and bottom 5 drugs
    drug_quality = quality_df.groupby('drug')['quality_score'].mean().sort_values(ascending=False)
    top_drugs = drug_quality.head(5)
    bottom_drugs = drug_quality.tail(5)
    
    # Create table data
    table_data = []
    table_data.append(['Top 5 Drugs', 'Quality Score', '', 'Bottom 5 Drugs', 'Quality Score'])
    for i in range(5):
        row = [top_drugs.index[i][:20], f'{top_drugs.iloc[i]:.3f}', '  ',
               bottom_drugs.index[i][:20], f'{bottom_drugs.iloc[i]:.3f}']
        table_data.append(row)
    
    table = ax5.table(cellText=table_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code quality scores
    for i in range(1, 6):
        # Top drugs (green gradient)
        table[(i, 1)].set_facecolor(plt.cm.Greens(top_drugs.iloc[i-1]))
        # Bottom drugs (red gradient)  
        table[(i, 4)].set_facecolor(plt.cm.Reds(1 - bottom_drugs.iloc[i-1]))
    
    ax5.set_title('Best and Worst Quality Drugs', fontweight='bold', pad=20)
    
    plt.suptitle('Quality Assessment Summary Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Load oxygen consumption data
print("\n📊 Loading oxygen consumption data...")
import sys
sys.path.append(str(project_root / "src"))
from utils.data_loader import DataLoader

loader = DataLoader()
# Load limited data for faster demonstration
df = loader.load_oxygen_data(limit=3)  # Only 3 plates for demo
print(f"   Loaded {len(df)} measurements")
print(f"   Wells: {df['well_id'].nunique()}")

# Load media change events if available
spike_data = {}
media_changes_file = results_dir / "improved_media_change_events.parquet"
if media_changes_file.exists():
    print("\n📊 Loading media change events for spike analysis...")
    media_changes = pd.read_parquet(media_changes_file)
    
    # Get spike heights for each well (media changes affect all wells on plate)
    for _, event in media_changes.iterrows():
        plate_id = event['plate_id']
        event_time = event['event_time_hours']
        
        # Get all wells on this plate
        plate_wells = df[df['plate_id'] == plate_id]['well_id'].unique()
        
        for well_id in plate_wells:
            if well_id not in spike_data:
                spike_data[well_id] = []
            
            # Get O2 values around spike for this well
            spike_mask = (df['well_id'] == well_id) & \
                        (df['elapsed_hours'] >= event_time - 2) & \
                        (df['elapsed_hours'] <= event_time + 2)
            
            spike_values = df[spike_mask]['o2'].values
            if len(spike_values) > 0:
                spike_data[well_id].append(np.max(spike_values))

# Initialize quality assessor
qa = QualityAssessor()

# Process each well
print(f"\n🔄 Assessing quality for each well...")
quality_results = []

wells = df['well_id'].unique()
for well_id in tqdm(wells, desc="Processing wells"):
    well_data = df[df['well_id'] == well_id].sort_values('elapsed_hours')
    
    # Get metadata
    drug = well_data['drug'].iloc[0]
    concentration = well_data['concentration'].iloc[0]
    
    # Extract time series
    time_series = well_data['o2'].values
    time_hours = well_data['elapsed_hours'].values
    
    # Initialize quality record
    quality_record = {
        'well_id': well_id,
        'drug': drug,
        'concentration': concentration
    }
    
    # Run quality assessments
    # 1. Low points
    low_points_result = qa.assess_low_points(time_series)
    quality_record.update(low_points_result)
    
    # 2. High noise
    noise_result = qa.assess_high_noise(time_series)
    quality_record.update(noise_result)
    
    # 3. Sensor drift
    drift_result = qa.assess_sensor_drift(time_series, time_hours)
    quality_record.update(drift_result)
    
    # 4. Baseline stability
    baseline_result = qa.assess_baseline_stability(time_series, time_hours)
    quality_record.update(baseline_result)
    
    # 5. Data gaps
    gaps_result = qa.assess_data_gaps(time_hours)
    quality_record.update(gaps_result)
    
    # 6. Spike outliers (if available)
    if well_id in spike_data:
        spike_result = qa.assess_spike_outliers(time_series, spike_data[well_id])
        quality_record.update(spike_result)
    else:
        quality_record.update({
            'spike_outlier': False,
            'n_spikes': 0,
            'max_spike_z': 0
        })
    
    # Overall quality score (0-1, higher is better)
    quality_flags = [
        quality_record.get('low_points', False),
        quality_record.get('high_noise', False),
        quality_record.get('sensor_drift', False),
        quality_record.get('baseline_unstable', False),
        quality_record.get('has_gaps', False),
        quality_record.get('spike_outlier', False)
    ]
    
    quality_record['n_quality_flags'] = sum(quality_flags)
    quality_record['quality_score'] = 1 - (sum(quality_flags) / len(quality_flags))
    quality_record['is_high_quality'] = quality_record['n_quality_flags'] == 0
    
    quality_results.append(quality_record)

# Convert to DataFrame
quality_df = pd.DataFrame(quality_results)

# Add replicate discord assessment
print("\n🔄 Assessing replicate discord...")
replicate_quality = []

for (drug, concentration), group in quality_df.groupby(['drug', 'concentration']):
    # Get feature values for replicates (use baseline_mean as proxy)
    replicate_values = group['baseline_mean'].dropna().values
    
    if len(replicate_values) >= 2:
        discord_result = qa.assess_replicate_discord(replicate_values)
        
        # Update all wells in this group
        for idx in group.index:
            quality_df.loc[idx, 'replicate_discord'] = discord_result['replicate_discord']
            quality_df.loc[idx, 'replicate_cv'] = discord_result['replicate_cv']
            quality_df.loc[idx, 'n_replicates_assessed'] = discord_result['n_replicates']

# Update overall quality score with replicate discord
quality_df['n_quality_flags'] = quality_df[['low_points', 'high_noise', 'sensor_drift', 
                                            'baseline_unstable', 'has_gaps', 'spike_outlier',
                                            'replicate_discord']].fillna(False).sum(axis=1)
quality_df['quality_score'] = 1 - (quality_df['n_quality_flags'] / 7)
quality_df['is_high_quality'] = quality_df['n_quality_flags'] == 0

# Save results
print("\n💾 Saving results...")
quality_df.to_parquet(results_dir / 'quality_assessment_flags.parquet', index=False)
print(f"   Saved to: {results_dir / 'quality_assessment_flags.parquet'}")

# Summary statistics
print("\n📊 QUALITY ASSESSMENT SUMMARY:")
print(f"   Total wells assessed: {len(quality_df)}")
print(f"   High quality wells: {quality_df['is_high_quality'].sum()} ({quality_df['is_high_quality'].sum()/len(quality_df)*100:.1f}%)")

print("\n📈 Quality flag prevalence:")
for flag in ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable', 
             'has_gaps', 'spike_outlier', 'replicate_discord']:
    if flag in quality_df.columns:
        n_flagged = quality_df[flag].sum()
        print(f"   {flag}: {n_flagged} wells ({n_flagged/len(quality_df)*100:.1f}%)")

# Drug-level quality summary
print("\n🔍 Drug-level quality:")
drug_quality = quality_df.groupby('drug').agg({
    'quality_score': ['mean', 'std', 'min'],
    'is_high_quality': 'mean',
    'n_quality_flags': 'mean'
}).round(3)

print("\nTop 5 highest quality drugs:")
top_drugs = drug_quality.sort_values(('quality_score', 'mean'), ascending=False).head(5)
for drug, row in top_drugs.iterrows():
    print(f"   {drug}: score={row[('quality_score', 'mean')]:.3f}, high_quality_fraction={row[('is_high_quality', 'mean')]:.3f}")

print("\nBottom 5 lowest quality drugs:")
bottom_drugs = drug_quality.sort_values(('quality_score', 'mean'), ascending=True).head(5)
for drug, row in bottom_drugs.iterrows():
    print(f"   {drug}: score={row[('quality_score', 'mean')]:.3f}, flags={row[('n_quality_flags', 'mean')]:.1f}")

# Create quality-aware feature flags
print("\n🚩 Creating quality-aware feature flags...")
quality_features = quality_df[['well_id', 'drug', 'concentration', 'quality_score', 'is_high_quality',
                              'low_points', 'high_noise', 'sensor_drift', 'baseline_unstable',
                              'has_gaps', 'spike_outlier', 'replicate_discord']].copy()

# Convert boolean flags to binary features
for flag in ['low_points', 'high_noise', 'sensor_drift', 'baseline_unstable', 
             'has_gaps', 'spike_outlier', 'replicate_discord']:
    quality_features[f'qflag_{flag}'] = quality_features[flag].fillna(False).astype(int)

quality_features.to_parquet(results_dir / 'quality_aware_features.parquet', index=False)
print(f"   Saved quality features to: {results_dir / 'quality_aware_features.parquet'}")

# Generate visualizations
print("\n📊 Generating visualizations...")

# 1. Quality flag distribution and scores
print("   Creating quality flag distribution visualization...")
visualize_quality_flag_distribution(quality_df, fig_dir / 'quality_flag_distribution.png')
print(f"      Saved: quality_flag_distribution.png")

# 2. Quality examples 
print("   Creating quality issue examples...")
visualize_quality_examples(df, quality_df, fig_dir / 'quality_examples.png')
print(f"      Saved: quality_examples.png")

# 3. Quality by drug heatmap
print("   Creating quality by drug heatmap...")
visualize_quality_by_drug_heatmap(quality_df, fig_dir / 'quality_by_drug_heatmap.png')
print(f"      Saved: quality_by_drug_heatmap.png")

# 4. Quality correlation matrix
print("   Creating quality correlation matrix...")
visualize_quality_correlation_matrix(quality_df, fig_dir / 'quality_correlation_matrix.png')
print(f"      Saved: quality_correlation_matrix.png")

# 5. Replicate consistency
print("   Creating replicate consistency visualization...")
visualize_replicate_consistency(quality_df, fig_dir / 'replicate_consistency.png')
print(f"      Saved: replicate_consistency.png")

# 6. Summary dashboard
print("   Creating quality summary dashboard...")
visualize_quality_summary_dashboard(quality_df, fig_dir / 'quality_summary_dashboard.png')
print(f"      Saved: quality_summary_dashboard.png")

print("\n✅ Quality assessment complete!")
print(f"   Generated quality flags for {len(quality_df)} wells")
print(f"   High quality wells: {quality_df['is_high_quality'].sum()} ({quality_df['is_high_quality'].sum()/len(quality_df)*100:.1f}%)")
print(f"   Created 6 visualization figures in: {fig_dir}")
print(f"\n   Next steps:")
print(f"   1. Use quality_score to weight features in embeddings")
print(f"   2. Include quality flags as features (not filters)")
print(f"   3. Compare model performance with/without quality weighting")
print(f"   4. Analyze if poor quality wells have predictive value")