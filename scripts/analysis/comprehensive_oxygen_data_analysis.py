#!/usr/bin/env python3
"""
Comprehensive analysis of oxygen time series data characteristics.
Analyzes events, concentrations, temporal patterns, and data quality.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()


def load_oxygen_data():
    """Load comprehensive oxygen data with all metadata."""
    print("Loading oxygen time series data...")
    
    # Load from parquet files
    processed_data = pd.read_parquet(project_root / "data" / "raw" / "processed_data_updated.parquet")
    well_map = pd.read_parquet(project_root / "data" / "raw" / "well_map_data_updated.parquet")
    
    # Merge on plate_id and well_number
    result = processed_data.merge(well_map, on=['plate_id', 'well_number'], how='left', suffixes=('', '_map'))
    
    # Rename columns to match expected names
    result['value'] = result['median_o2']
    result['drug_name'] = result['drug']
    result['well_id'] = result['plate_id'].astype(str) + '_' + result['well_number'].astype(str)
    result['experiment_id'] = result['plate_id']  # Use plate_id as experiment_id
    result['well_name'] = result['well_number']
    
    # Calculate elapsed time in hours and days from timestamps
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result = result.sort_values(['plate_id', 'well_number', 'timestamp'])
    
    # Calculate elapsed time per well
    result['elapsed_time'] = 0.0
    result['elapsed_days'] = 0.0
    
    for plate_id in result['plate_id'].unique():
        plate_data = result[result['plate_id'] == plate_id]
        min_time = plate_data['timestamp'].min()
        
        # Calculate elapsed time in hours
        elapsed = (plate_data['timestamp'] - min_time).dt.total_seconds() / 3600
        result.loc[result['plate_id'] == plate_id, 'elapsed_time'] = elapsed
        result.loc[result['plate_id'] == plate_id, 'elapsed_days'] = elapsed / 24
    
    # Add readout_id (unique identifier for each timepoint)
    result['readout_id'] = range(len(result))
    
    # Determine if control (no drug or DMSO)
    result['is_control'] = result['drug'].isna() | (result['drug'] == 'DMSO') | (result['concentration'] == 0)
    
    # Add basic event columns
    result['media_change'] = 1  # Assume all have media changes
    result['dosing'] = 1  # Assume all have dosing
    result['readout'] = 'O2'  # All oxygen data
    
    # Calculate experiment-level stats
    exp_stats = result.groupby('experiment_id')['elapsed_time'].agg(['min', 'max', 'count']).reset_index()
    exp_stats.columns = ['experiment_id', 'experiment_start', 'experiment_end', 'n_timepoints']
    
    # Add to result
    result = result.merge(exp_stats, on='experiment_id', how='left')
    
    # Use the correct exclusion column and ensure it's boolean
    if 'is_excluded_map' in result.columns:
        result['is_excluded'] = result['is_excluded_map'].fillna(False).astype(bool)
    else:
        result['is_excluded'] = result['is_excluded'].fillna(False).astype(bool)
    
    print(f"  Loaded {len(result):,} oxygen measurements")
    print(f"  Unique wells: {result['well_id'].nunique():,}")
    print(f"  Unique experiments: {result['experiment_id'].nunique():,}")
    print(f"  Unique drugs: {result['drug_name'].nunique():,}")
    
    return result


def analyze_temporal_characteristics(data):
    """Analyze temporal patterns in the data."""
    print("\nAnalyzing temporal characteristics...")
    
    temporal_stats = {}
    
    # Overall time range
    temporal_stats['overall'] = {
        'min_elapsed_days': data['elapsed_days'].min(),
        'max_elapsed_days': data['elapsed_days'].max(),
        'mean_experiment_duration': data.groupby('experiment_id')['elapsed_days'].max().mean(),
        'std_experiment_duration': data.groupby('experiment_id')['elapsed_days'].max().std()
    }
    
    # Sampling intervals
    well_intervals = []
    for well_id in data['well_id'].unique()[:100]:  # Sample for efficiency
        well_data = data[data['well_id'] == well_id].sort_values('elapsed_time')
        if len(well_data) > 1:
            intervals = np.diff(well_data['elapsed_time'].values)
            well_intervals.extend(intervals)
    
    temporal_stats['sampling'] = {
        'mean_interval_hours': np.mean(well_intervals),
        'std_interval_hours': np.std(well_intervals),
        'min_interval_hours': np.min(well_intervals),
        'max_interval_hours': np.max(well_intervals)
    }
    
    # Time points per well
    timepoints_per_well = data.groupby('well_id')['readout_id'].nunique()
    temporal_stats['coverage'] = {
        'mean_timepoints_per_well': timepoints_per_well.mean(),
        'std_timepoints_per_well': timepoints_per_well.std(),
        'min_timepoints_per_well': timepoints_per_well.min(),
        'max_timepoints_per_well': timepoints_per_well.max()
    }
    
    return temporal_stats


def analyze_concentration_patterns(data):
    """Analyze concentration distributions and patterns."""
    print("\nAnalyzing concentration patterns...")
    
    # Filter non-control wells
    drug_data = data[~data['is_control'] & ~data['is_excluded']].copy()
    
    # Get unique concentrations per drug
    conc_stats = drug_data.groupby('drug_name')['concentration'].agg(['nunique', 'min', 'max', 'mean', 'std'])
    conc_stats = conc_stats.sort_values('nunique', ascending=False)
    
    # Overall concentration distribution
    all_concentrations = drug_data.groupby(['drug_name', 'concentration']).size().reset_index()
    
    # Concentration ranges
    conc_patterns = {
        'n_drugs': conc_stats.shape[0],
        'mean_concentrations_per_drug': conc_stats['nunique'].mean(),
        'std_concentrations_per_drug': conc_stats['nunique'].std(),
        'global_min_concentration': drug_data['concentration'].min(),
        'global_max_concentration': drug_data['concentration'].max(),
        'common_concentrations': drug_data['concentration'].value_counts().head(10).to_dict()
    }
    
    return conc_stats, conc_patterns


def analyze_events(data):
    """Analyze media changes and dosing events."""
    print("\nAnalyzing experimental events...")
    
    # Get event summary
    event_summary = data.groupby('experiment_id').agg({
        'media_change': 'first',
        'dosing': 'first',
        'experiment_start': 'first',
        'experiment_end': 'first',
        'n_timepoints': 'first'
    }).reset_index()
    
    event_stats = {
        'experiments_with_media_change': event_summary['media_change'].sum(),
        'experiments_with_dosing': event_summary['dosing'].sum(),
        'total_experiments': len(event_summary),
        'mean_duration_days': (event_summary['experiment_end'] - event_summary['experiment_start']).mean() / 24,
        'mean_timepoints': event_summary['n_timepoints'].mean()
    }
    
    # Analyze media change timing
    media_change_wells = data[data['media_change'] == 1]['well_id'].unique()
    media_change_timings = []
    
    for well_id in media_change_wells[:50]:  # Sample for analysis
        well_data = data[data['well_id'] == well_id].sort_values('elapsed_time')
        values = well_data['value'].values
        times = well_data['elapsed_time'].values
        
        # Detect sudden changes (potential media changes)
        if len(values) > 10:
            diffs = np.abs(np.diff(values))
            threshold = np.percentile(diffs, 95)
            change_indices = np.where(diffs > threshold)[0]
            
            for idx in change_indices:
                media_change_timings.append(times[idx] / 24)  # Convert to days
    
    if media_change_timings:
        event_stats['media_change_timing'] = {
            'mean_days': np.mean(media_change_timings),
            'std_days': np.std(media_change_timings),
            'common_days': np.histogram(media_change_timings, bins=range(0, 15))[0].tolist()
        }
    
    return event_stats


def analyze_data_quality(data):
    """Analyze data quality issues."""
    print("\nAnalyzing data quality...")
    
    quality_stats = {}
    
    # Missing values
    quality_stats['missing_values'] = {
        'total_missing': data['value'].isna().sum(),
        'percent_missing': (data['value'].isna().sum() / len(data)) * 100
    }
    
    # Outliers (simple statistical approach)
    q1 = data['value'].quantile(0.25)
    q3 = data['value'].quantile(0.75)
    iqr = q3 - q1
    outliers = ((data['value'] < (q1 - 3 * iqr)) | (data['value'] > (q3 + 3 * iqr))).sum()
    
    quality_stats['outliers'] = {
        'n_outliers': outliers,
        'percent_outliers': (outliers / len(data)) * 100
    }
    
    # Excluded wells
    excluded_wells = data[data['is_excluded']]['well_id'].nunique()
    total_wells = data['well_id'].nunique()
    
    quality_stats['exclusions'] = {
        'excluded_wells': excluded_wells,
        'total_wells': total_wells,
        'percent_excluded': (excluded_wells / total_wells) * 100
    }
    
    # Value ranges
    quality_stats['value_range'] = {
        'min': data['value'].min(),
        'max': data['value'].max(),
        'mean': data['value'].mean(),
        'std': data['value'].std(),
        'percentiles': {
            '1%': data['value'].quantile(0.01),
            '5%': data['value'].quantile(0.05),
            '25%': data['value'].quantile(0.25),
            '50%': data['value'].quantile(0.50),
            '75%': data['value'].quantile(0.75),
            '95%': data['value'].quantile(0.95),
            '99%': data['value'].quantile(0.99)
        }
    }
    
    return quality_stats


def analyze_control_patterns(data):
    """Analyze control well patterns."""
    print("\nAnalyzing control patterns...")
    
    control_data = data[data['is_control']].copy()
    
    control_stats = {
        'n_control_wells': control_data['well_id'].nunique(),
        'n_control_experiments': control_data['experiment_id'].nunique(),
        'control_compounds': control_data['drug_name'].value_counts().to_dict() if 'drug_name' in control_data.columns else {}
    }
    
    # Control stability over time
    control_timeseries = []
    for well_id in control_data['well_id'].unique()[:20]:  # Sample
        well_data = control_data[control_data['well_id'] == well_id].sort_values('elapsed_time')
        if len(well_data) > 50:
            control_timeseries.append({
                'well_id': well_id,
                'mean': well_data['value'].mean(),
                'std': well_data['value'].std(),
                'cv': well_data['value'].std() / well_data['value'].mean() if well_data['value'].mean() > 0 else np.nan
            })
    
    if control_timeseries:
        cv_values = [ts['cv'] for ts in control_timeseries if not np.isnan(ts['cv'])]
        control_stats['stability'] = {
            'mean_cv': np.mean(cv_values),
            'std_cv': np.std(cv_values),
            'stable_controls': sum(1 for cv in cv_values if cv < 0.1)
        }
    
    return control_stats


def create_visualizations(data, output_dir):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # 1. Temporal coverage
    ax1 = axes[0, 0]
    exp_durations = data.groupby('experiment_id')['elapsed_days'].max()
    ax1.hist(exp_durations, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Experiment Duration (days)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Experiment Durations')
    ax1.axvline(14, color='red', linestyle='--', label='14 days')
    ax1.legend()
    
    # 2. Concentration distribution
    ax2 = axes[0, 1]
    drug_data = data[~data['is_control'] & ~data['is_excluded']]
    conc_counts = drug_data.groupby('drug_name')['concentration'].nunique()
    ax2.hist(conc_counts, bins=range(1, 12), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Concentrations')
    ax2.set_ylabel('Number of Drugs')
    ax2.set_title('Concentrations per Drug')
    ax2.axvline(4, color='red', linestyle='--', label='Min threshold')
    ax2.legend()
    
    # 3. Sampling frequency
    ax3 = axes[0, 2]
    # Sample some wells for interval analysis
    sample_wells = data['well_id'].unique()[:50]
    all_intervals = []
    for well_id in sample_wells:
        well_data = data[data['well_id'] == well_id].sort_values('elapsed_time')
        if len(well_data) > 1:
            intervals = np.diff(well_data['elapsed_time'].values)
            all_intervals.extend(intervals)
    
    ax3.hist(all_intervals, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Sampling Interval (hours)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Sampling Intervals')
    ax3.set_xlim(0, 10)
    
    # 4. Value distribution
    ax4 = axes[1, 0]
    ax4.hist(data['value'].sample(10000), bins=100, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Oxygen Value')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Oxygen Values')
    
    # 5. Control vs treatment comparison
    ax5 = axes[1, 1]
    control_values = data[data['is_control']]['value'].sample(min(5000, len(data[data['is_control']])))
    treatment_values = data[~data['is_control']]['value'].sample(min(5000, len(data[~data['is_control']])))
    
    ax5.hist(control_values, bins=50, alpha=0.5, label='Control', density=True)
    ax5.hist(treatment_values, bins=50, alpha=0.5, label='Treatment', density=True)
    ax5.set_xlabel('Oxygen Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Control vs Treatment Value Distributions')
    ax5.legend()
    
    # 6. Time series examples
    ax6 = axes[1, 2]
    # Plot a few example time series
    example_wells = data[~data['is_control'] & ~data['is_excluded']]['well_id'].unique()[:5]
    for well_id in example_wells:
        well_data = data[data['well_id'] == well_id].sort_values('elapsed_time')
        ax6.plot(well_data['elapsed_days'], well_data['value'], alpha=0.7)
    
    ax6.set_xlabel('Days')
    ax6.set_ylabel('Oxygen Value')
    ax6.set_title('Example Time Series')
    ax6.grid(True, alpha=0.3)
    
    # 7. Concentration vs response
    ax7 = axes[2, 0]
    # Sample one drug with multiple concentrations
    multi_conc_drugs = drug_data.groupby('drug_name')['concentration'].nunique()
    example_drug = multi_conc_drugs[multi_conc_drugs >= 8].index[0] if any(multi_conc_drugs >= 8) else None
    
    if example_drug:
        drug_subset = drug_data[drug_data['drug_name'] == example_drug]
        conc_means = drug_subset.groupby('concentration')['value'].mean().sort_index()
        ax7.semilogx(conc_means.index, conc_means.values, 'o-')
        ax7.set_xlabel('Concentration')
        ax7.set_ylabel('Mean Oxygen Value')
        ax7.set_title(f'Dose Response: {example_drug}')
        ax7.grid(True, alpha=0.3)
    
    # 8. Missing data patterns
    ax8 = axes[2, 1]
    missing_by_exp = data.groupby('experiment_id')['value'].apply(lambda x: x.isna().sum() / len(x) * 100)
    ax8.hist(missing_by_exp, bins=30, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('% Missing Values')
    ax8.set_ylabel('Number of Experiments')
    ax8.set_title('Missing Data by Experiment')
    
    # 9. Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    summary_text = f"""
DATA SUMMARY

Total Measurements: {len(data):,}
Unique Wells: {data['well_id'].nunique():,}
Unique Drugs: {data['drug_name'].nunique():,}
Unique Experiments: {data['experiment_id'].nunique():,}

Time Coverage:
• Mean duration: {data.groupby('experiment_id')['elapsed_days'].max().mean():.1f} days
• Max duration: {data['elapsed_days'].max():.1f} days

Concentrations:
• Mean per drug: {drug_data.groupby('drug_name')['concentration'].nunique().mean():.1f}
• Range: {drug_data['concentration'].min():.2f} - {drug_data['concentration'].max():.0f}

Data Quality:
• Missing: {(data['value'].isna().sum() / len(data) * 100):.2f}%
• Excluded wells: {data[data['is_excluded']]['well_id'].nunique()}
"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11, 
             va='top', fontfamily='monospace')
    
    plt.suptitle('Oxygen Time Series Data Characteristics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'oxygen_data_characteristics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_documentation(temporal_stats, conc_stats, conc_patterns, event_stats, 
                         quality_stats, control_stats, output_dir):
    """Generate comprehensive documentation."""
    print("\nGenerating documentation...")
    
    doc_content = f"""# Oxygen Real-Time Data Documentation

## Overview

This document provides a comprehensive analysis of the oxygen consumption time series data from liver organoid experiments. The data consists of real-time oxygen measurements from organoids treated with various drugs at multiple concentrations, along with control wells.

## Data Characteristics

### Temporal Coverage

- **Mean experiment duration**: {temporal_stats['overall']['mean_experiment_duration']:.1f} ± {temporal_stats['overall']['std_experiment_duration']:.1f} days
- **Maximum duration**: {temporal_stats['overall']['max_elapsed_days']:.1f} days
- **Sampling interval**: {temporal_stats['sampling']['mean_interval_hours']:.1f} ± {temporal_stats['sampling']['std_interval_hours']:.1f} hours
- **Timepoints per well**: {temporal_stats['coverage']['mean_timepoints_per_well']:.0f} ± {temporal_stats['coverage']['std_timepoints_per_well']:.0f}

### Concentration Patterns

- **Number of drugs**: {conc_patterns['n_drugs']}
- **Concentrations per drug**: {conc_patterns['mean_concentrations_per_drug']:.1f} ± {conc_patterns['std_concentrations_per_drug']:.1f}
- **Global concentration range**: {conc_patterns['global_min_concentration']:.4f} - {conc_patterns['global_max_concentration']:.0f}

#### Most Common Concentrations:
{chr(10).join([f"- {conc:.2f}: {count} wells" for conc, count in list(conc_patterns['common_concentrations'].items())[:5]])}

### Experimental Events

- **Experiments with media changes**: {event_stats['experiments_with_media_change']} / {event_stats['total_experiments']} ({event_stats['experiments_with_media_change']/event_stats['total_experiments']*100:.1f}%)
- **Experiments with dosing events**: {event_stats['experiments_with_dosing']} / {event_stats['total_experiments']} ({event_stats['experiments_with_dosing']/event_stats['total_experiments']*100:.1f}%)
- **Mean experiment duration**: {event_stats['mean_duration_days']:.1f} days

### Data Quality

#### Missing Values
- **Total missing**: {quality_stats['missing_values']['total_missing']:,} ({quality_stats['missing_values']['percent_missing']:.2f}%)
- **Outliers**: {quality_stats['outliers']['n_outliers']:,} ({quality_stats['outliers']['percent_outliers']:.2f}%)

#### Exclusions
- **Excluded wells**: {quality_stats['exclusions']['excluded_wells']} / {quality_stats['exclusions']['total_wells']} ({quality_stats['exclusions']['percent_excluded']:.1f}%)

#### Value Distribution
- **Range**: {quality_stats['value_range']['min']:.2f} - {quality_stats['value_range']['max']:.2f}
- **Mean ± SD**: {quality_stats['value_range']['mean']:.2f} ± {quality_stats['value_range']['std']:.2f}
- **Median (IQR)**: {quality_stats['value_range']['percentiles']['50%']:.2f} ({quality_stats['value_range']['percentiles']['25%']:.2f} - {quality_stats['value_range']['percentiles']['75%']:.2f})

### Control Wells

- **Number of control wells**: {control_stats['n_control_wells']}
- **Control experiments**: {control_stats['n_control_experiments']}
- **Control compounds**: {', '.join([f"{comp} (n={count})" for comp, count in list(control_stats['control_compounds'].items())[:3]])}

## Critical Considerations for Feature Engineering

### 1. Media Change Artifacts

Media changes cause sudden jumps in oxygen consumption that are not biologically relevant. Features must either:
- Detect and exclude media change periods
- Use robust statistics that are insensitive to sudden jumps
- Model media changes explicitly as events

### 2. Concentration-Aware Features

With drugs tested at 4-10 concentrations (typically 8), features should:
- Capture dose-response relationships
- Allow for non-monotonic responses
- Consider concentration as a continuous variable, not categorical

### 3. Temporal Alignment

Key considerations:
- Experiments have different durations (10-20+ days)
- Sampling is irregular (~1-2 hour intervals)
- Early time points may reflect adaptation rather than drug response
- Late time points may show secondary effects

### 4. Control Normalization

Controls provide baseline oxygen consumption:
- Each experiment has control wells
- Controls show temporal drift and plate effects
- Normalization to controls can improve drug effect detection

### 5. Data Quality Issues

Common problems to handle:
- Missing values (sporadic, not systematic)
- Outliers from measurement artifacts
- Excluded wells that failed quality control
- Variable number of replicates per condition

## Recommended Feature Engineering Strategies

### 1. Robust Summary Statistics
- Median-based statistics instead of mean
- Trimmed means to reduce outlier influence
- MAD (Median Absolute Deviation) for variability

### 2. Change Point Detection
- Identify media change events automatically
- Segment time series into pre/post media change
- Calculate features separately for each segment

### 3. Dose-Response Features
- Hill equation parameters (EC50, slope, max effect)
- Area under the dose-response curve
- Concentration at maximum effect

### 4. Temporal Features
- Early response (days 1-3)
- Sustained response (days 4-10)
- Late/adaptive response (days 10+)
- Rate of change over specific windows

### 5. Frequency Domain Features
- Fourier coefficients for periodic patterns
- Power spectral density for oscillation detection
- Wavelet features for multi-scale analysis

### 6. Event-Aware Features
- Time to first significant change
- Recovery time after media change
- Stability metrics between events

## Data Structure for Analysis

### Hierarchical Organization
1. **Well level**: Individual time series (4 replicates per condition)
2. **Concentration level**: Aggregate across replicates
3. **Drug level**: Aggregate across concentrations

### Recommended Workflow
1. Filter by quality (is_excluded=false)
2. Filter by coverage (≥14 days, ≥4 concentrations)
3. Detect and handle media changes
4. Normalize to controls within experiment
5. Extract features at appropriate hierarchy level
6. Validate features across multiple drugs

## Key Insights

1. **Consistency is critical**: Features must be comparable across drugs with different concentration ranges and experimental conditions

2. **Event handling is essential**: Media changes are present in most experiments and must be explicitly handled

3. **Hierarchical structure matters**: Preserve dose-response relationships by maintaining concentration hierarchy

4. **Time windows are important**: Different biological processes occur at different timescales

5. **Robustness over sensitivity**: Better to have stable features across experiments than highly sensitive features that vary with conditions

This comprehensive understanding of the data characteristics should guide the development of meaningful, consistent features for drug comparison and toxicity prediction.
"""
    
    doc_path = project_root / "docs" / "O2_REALTIME_DATA.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    print(f"  Documentation saved to: {doc_path}")
    return doc_path


def main():
    """Main analysis pipeline."""
    print("=== Comprehensive Oxygen Data Analysis ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "data_characteristics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_oxygen_data()
    
    # Analyze different aspects
    temporal_stats = analyze_temporal_characteristics(data)
    conc_stats, conc_patterns = analyze_concentration_patterns(data)
    event_stats = analyze_events(data)
    quality_stats = analyze_data_quality(data)
    control_stats = analyze_control_patterns(data)
    
    # Create visualizations
    viz_path = create_visualizations(data, output_dir)
    
    # Generate documentation
    doc_path = generate_documentation(
        temporal_stats, conc_stats, conc_patterns, 
        event_stats, quality_stats, control_stats, output_dir
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Visualization saved to: {viz_path}")
    print(f"📄 Documentation saved to: {doc_path}")


if __name__ == "__main__":
    main()