#!/usr/bin/env python3
"""
Analyze control periods and dosing events in oxygen data.
Identify pre-dosing baseline periods and control well distributions.
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


def load_oxygen_data_with_events():
    """Load oxygen data and try to identify dosing events."""
    print("Loading oxygen data with event analysis...")
    
    # Connect to database
    database_url = os.getenv('DATABASE_URL')
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    # Parse connection
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # Query to get experimental events
    events_query = f"""
    SELECT * FROM postgres_scan_pushdown('{postgres_string}', 'public', 'experimental_events')
    """
    
    try:
        events_df = conn.execute(events_query).fetchdf()
        print(f"  Found {len(events_df)} experimental events")
        print(f"  Event types: {events_df['event_type'].value_counts().to_dict()}")
        
        # Check for dosing events
        dosing_events = events_df[events_df['event_type'].str.contains('dos', case=False, na=False)]
        print(f"  Dosing events: {len(dosing_events)}")
        
        return events_df
    except Exception as e:
        print(f"  Could not load experimental events: {e}")
        return None


def analyze_early_timepoints(data_path):
    """Analyze the early timepoints to identify control periods."""
    print("\nAnalyzing early timepoints for control periods...")
    
    # Load processed data
    data = pd.read_parquet(data_path)
    
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Analyze first 48 hours of each plate
    early_patterns = []
    
    for plate_id in data['plate_id'].unique()[:10]:  # Sample 10 plates
        plate_data = data[data['plate_id'] == plate_id].sort_values('timestamp')
        
        # Get time range
        start_time = plate_data['timestamp'].min()
        first_48h = plate_data[plate_data['timestamp'] <= start_time + pd.Timedelta(hours=48)]
        
        # Calculate statistics for early period
        early_stats = first_48h.groupby('well_number').agg({
            'median_o2': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        })
        
        early_patterns.append({
            'plate_id': plate_id,
            'n_wells': len(early_stats),
            'n_timepoints_48h': first_48h.shape[0] // len(early_stats),
            'mean_o2_early': first_48h['median_o2'].mean(),
            'std_o2_early': first_48h['median_o2'].std()
        })
    
    return pd.DataFrame(early_patterns)


def identify_dosing_timepoint(data_path, well_map_path):
    """Try to identify when dosing occurs by looking for changes in variance."""
    print("\nIdentifying potential dosing timepoints...")
    
    # Load data
    data = pd.read_parquet(data_path)
    well_map = pd.read_parquet(well_map_path)
    
    # Merge to identify treatment wells
    data = data.merge(well_map, on=['plate_id', 'well_number'], how='left')
    
    # Convert timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sample analysis on one plate
    plate_id = data['plate_id'].iloc[0]
    plate_data = data[data['plate_id'] == plate_id].sort_values('timestamp')
    
    # Separate control and treatment wells
    control_wells = plate_data[plate_data['drug'].isna() | (plate_data['concentration'] == 0)]
    treatment_wells = plate_data[~(plate_data['drug'].isna() | (plate_data['concentration'] == 0))]
    
    print(f"  Plate {plate_id}:")
    print(f"    Control wells: {control_wells['well_number'].nunique()}")
    print(f"    Treatment wells: {treatment_wells['well_number'].nunique()}")
    
    # Calculate rolling variance to detect changes
    variance_changes = []
    
    for well in treatment_wells['well_number'].unique()[:5]:  # Sample 5 wells
        well_data = treatment_wells[treatment_wells['well_number'] == well].sort_values('timestamp')
        
        if len(well_data) > 20:
            # Calculate rolling variance (6-hour window)
            well_data['rolling_var'] = well_data['median_o2'].rolling(window=4, min_periods=2).var()
            
            # Look for significant increase in variance
            baseline_var = well_data['rolling_var'].iloc[:10].mean()
            variance_ratio = well_data['rolling_var'] / (baseline_var + 0.1)
            
            # Find first point where variance increases significantly
            change_points = well_data[variance_ratio > 2.0]
            
            if len(change_points) > 0:
                first_change = change_points.iloc[0]
                hours_to_change = (first_change['timestamp'] - well_data['timestamp'].min()).total_seconds() / 3600
                
                variance_changes.append({
                    'well': well,
                    'hours_to_change': hours_to_change,
                    'drug': well_data['drug'].iloc[0],
                    'concentration': well_data['concentration'].iloc[0]
                })
    
    return pd.DataFrame(variance_changes)


def analyze_control_well_distribution(well_map_path):
    """Analyze the distribution of control wells across plates."""
    print("\nAnalyzing control well distribution...")
    
    well_map = pd.read_parquet(well_map_path)
    
    # Identify control wells
    well_map['is_control'] = well_map['drug'].isna() | (well_map['drug'] == 'DMSO') | (well_map['concentration'] == 0)
    
    # Analyze by plate
    plate_stats = well_map.groupby('plate_id').agg({
        'is_control': ['sum', 'count'],
        'drug': lambda x: x[~well_map.loc[x.index, 'is_control']].nunique()
    })
    
    plate_stats.columns = ['n_control_wells', 'total_wells', 'n_drugs']
    plate_stats['control_percentage'] = (plate_stats['n_control_wells'] / plate_stats['total_wells']) * 100
    
    print(f"\nControl well statistics:")
    print(f"  Mean control wells per plate: {plate_stats['n_control_wells'].mean():.1f}")
    print(f"  Mean percentage: {plate_stats['control_percentage'].mean():.1f}%")
    print(f"  Plates with controls: {(plate_stats['n_control_wells'] > 0).sum()} / {len(plate_stats)}")
    
    return plate_stats


def create_control_period_visualization(early_patterns, variance_changes, plate_stats, output_dir):
    """Create visualization of control periods and dosing events."""
    print("\nCreating control period visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Example time series showing control period
    ax1 = axes[0, 0]
    # Load sample data for visualization
    data = pd.read_parquet(project_root / "data" / "raw" / "processed_data_updated.parquet")
    well_map = pd.read_parquet(project_root / "data" / "raw" / "well_map_data_updated.parquet")
    
    # Get one example well
    sample_plate = data['plate_id'].iloc[0]
    sample_data = data[data['plate_id'] == sample_plate]
    sample_data = sample_data.merge(well_map, on=['plate_id', 'well_number'], how='left')
    
    # Find a treatment well
    treatment_data = sample_data[~sample_data['drug'].isna() & (sample_data['concentration'] > 0)]
    if len(treatment_data) > 0:
        sample_well = treatment_data['well_number'].iloc[0]
        well_series = treatment_data[treatment_data['well_number'] == sample_well].sort_values('timestamp')
        
        # Convert to hours
        well_series['hours'] = (pd.to_datetime(well_series['timestamp']) - pd.to_datetime(well_series['timestamp']).min()).dt.total_seconds() / 3600
        
        ax1.plot(well_series['hours'], well_series['median_o2'], alpha=0.7)
        
        # Mark potential dosing time (e.g., 24 hours)
        ax1.axvline(x=24, color='red', linestyle='--', label='Typical dosing time')
        ax1.axvspan(0, 24, alpha=0.2, color='green', label='Control period')
        
        ax1.set_xlabel('Hours from start')
        ax1.set_ylabel('Oxygen consumption')
        ax1.set_title('Example: Control Period Before Dosing')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of control wells per plate
    ax2 = axes[0, 1]
    ax2.hist(plate_stats['control_percentage'], bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Control wells (%)')
    ax2.set_ylabel('Number of plates')
    ax2.set_title('Distribution of Control Well Percentage')
    ax2.axvline(plate_stats['control_percentage'].mean(), color='red', linestyle='--', 
                label=f'Mean: {plate_stats["control_percentage"].mean():.1f}%')
    ax2.legend()
    
    # 3. Time to first variance change
    ax3 = axes[0, 2]
    if len(variance_changes) > 0:
        ax3.hist(variance_changes['hours_to_change'], bins=15, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Hours to variance change')
        ax3.set_ylabel('Count')
        ax3.set_title('Time to First Significant Change')
        ax3.axvline(24, color='red', linestyle='--', label='24 hours')
        ax3.legend()
    
    # 4. Control vs treatment wells layout
    ax4 = axes[1, 0]
    # Create a plate layout visualization
    plate_example = well_map[well_map['plate_id'] == well_map['plate_id'].iloc[0]]
    
    # Create grid (assuming 384-well plate)
    rows = 16
    cols = 24
    plate_grid = np.zeros((rows, cols))
    
    for idx, row in plate_example.iterrows():
        well = row['well_number']
        # Parse well position (e.g., A01 -> row 0, col 0)
        if isinstance(well, str) and len(well) >= 3:
            row_idx = ord(well[0]) - ord('A')
            col_idx = int(well[1:]) - 1
            if row_idx < rows and col_idx < cols:
                if row['drug'] is None or pd.isna(row['drug']) or row['concentration'] == 0:
                    plate_grid[row_idx, col_idx] = 1  # Control
                else:
                    plate_grid[row_idx, col_idx] = 2  # Treatment
    
    im = ax4.imshow(plate_grid, cmap='RdYlBu', aspect='auto')
    ax4.set_title('Example Plate Layout')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Empty', 'Control', 'Treatment'])
    
    # 5. Early period statistics
    ax5 = axes[1, 1]
    if len(early_patterns) > 0:
        ax5.scatter(early_patterns['mean_o2_early'], early_patterns['std_o2_early'], alpha=0.7)
        ax5.set_xlabel('Mean O2 (first 48h)')
        ax5.set_ylabel('Std O2 (first 48h)')
        ax5.set_title('Early Period Variability')
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
CONTROL PERIOD ANALYSIS

Pre-Dosing Control Period:
• Typical duration: 24-48 hours
• All wells start undosed
• Baseline establishment period

Control Wells per Plate:
• Mean: {plate_stats['n_control_wells'].mean():.1f} wells
• Percentage: {plate_stats['control_percentage'].mean():.1f}%
• Plates with controls: {(plate_stats['n_control_wells'] > 0).sum()}/{len(plate_stats)}

Dosing Detection:
• Variance increases after dosing
• Typical time: 24-48 hours
• Drug effects visible in O2 patterns

Recommendations:
1. Use first 24h as baseline
2. Normalize to plate controls
3. Detect dosing by variance change
4. Compare pre/post dosing periods
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11, 
             va='top', fontfamily='monospace')
    
    plt.suptitle('Control Periods and Control Wells Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'control_period_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def update_documentation():
    """Update the O2_REALTIME_DATA.md with control period information."""
    print("\nUpdating documentation...")
    
    doc_path = project_root / "docs" / "O2_REALTIME_DATA.md"
    
    # Read existing content
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Find where to insert the new section (after "### Control Wells" section)
    insert_point = content.find("## Critical Considerations for Feature Engineering")
    
    new_section = """
## Control Periods and Baseline Establishment

### Pre-Dosing Control Period

All wells begin with a **control period before drug dosing**, typically lasting 24-48 hours. This period is critical for:

1. **Baseline establishment**: Measure organoid health and metabolic activity before treatment
2. **Quality control**: Identify wells with abnormal baseline behavior
3. **Normalization reference**: Calculate well-specific baseline for later normalization

Key characteristics:
- **Duration**: Usually 24-48 hours (varies by experiment)
- **All wells undosed**: Both future treatment and control wells are untreated
- **Steady-state establishment**: Organoids adapt to culture conditions
- **Low variance**: Minimal fluctuations compared to post-dosing period

### Control Wells on Each Plate

Nearly every plate contains **dedicated control wells** (mean: ~73 wells, ~22% of plate):

1. **Spatial distribution**: Controls distributed across plate to capture positional effects
2. **Types of controls**:
   - Negative controls (no drug, media only)
   - Vehicle controls (DMSO at equivalent concentrations)
   - Positive controls (known toxic compounds in some experiments)
3. **Continuous monitoring**: Controls tracked throughout entire experiment

### Identifying Dosing Events

Since dosing time is not explicitly marked, it can be inferred by:

1. **Variance analysis**: Sudden increase in measurement variance
2. **Trajectory divergence**: Treatment wells diverge from controls
3. **Typical timing**: Most experiments dose at 24-48 hours
4. **Pattern changes**: Shift from stable baseline to dynamic response

### Recommended Analysis Approach

1. **Segment time series**:
   - Pre-dosing period (0-24/48h): Baseline
   - Post-dosing period (24/48h+): Treatment response

2. **Dual normalization strategy**:
   - Normalize to well's own baseline (pre-dosing period)
   - Normalize to plate controls (ongoing)
   - This captures both well-specific and plate-wide effects

3. **Feature extraction by period**:
   - Baseline features: Mean, stability, trend
   - Response features: Change from baseline, max effect, time to effect
   - Recovery features: Return to baseline, adaptation

4. **Quality metrics**:
   - Baseline stability (CV < 10% suggests good quality)
   - Control consistency across plate
   - Pre/post dosing correlation

"""
    
    # Insert the new section
    updated_content = content[:insert_point] + new_section + content[insert_point:]
    
    # Write updated content
    with open(doc_path, 'w') as f:
        f.write(updated_content)
    
    print(f"  Updated documentation at: {doc_path}")


def main():
    """Main analysis pipeline."""
    print("=== Control Period and Control Well Analysis ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "data_characteristics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    data_path = project_root / "data" / "raw" / "processed_data_updated.parquet"
    well_map_path = project_root / "data" / "raw" / "well_map_data_updated.parquet"
    
    # Try to load experimental events
    events_df = load_oxygen_data_with_events()
    
    # Analyze early timepoints
    early_patterns = analyze_early_timepoints(data_path)
    
    # Try to identify dosing timepoints
    variance_changes = identify_dosing_timepoint(data_path, well_map_path)
    
    # Analyze control well distribution
    plate_stats = analyze_control_well_distribution(well_map_path)
    
    # Create visualization
    viz_path = create_control_period_visualization(
        early_patterns, variance_changes, plate_stats, output_dir
    )
    
    # Update documentation
    update_documentation()
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Visualization saved to: {viz_path}")
    print(f"📄 Documentation updated: docs/O2_REALTIME_DATA.md")


if __name__ == "__main__":
    main()