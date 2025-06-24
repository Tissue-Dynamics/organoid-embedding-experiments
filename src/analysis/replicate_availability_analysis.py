#!/usr/bin/env python3
"""
Analyze replicate availability patterns in the oxygen consumption dataset.
Creates visualizations showing distribution of replicate counts per drug/concentration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
import os
from collections import defaultdict

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

def connect_to_database():
    """Connect to the PostgreSQL database via DuckDB."""
    # Use the database URL directly
    DATABASE_URL = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    # Parse connection string
    conn.execute(f"ATTACH '{DATABASE_URL}' AS supabase (TYPE POSTGRES, READ_ONLY);")
    return conn

def analyze_replicate_patterns():
    """Analyze how many replicates are available for each drug/concentration combination."""
    
    print("Connecting to database...")
    conn = connect_to_database()
    
    # Query to get replicate counts per drug/concentration
    query = """
    WITH well_data AS (
        SELECT 
            w.drug,
            w.concentration,
            w.well_number,
            w.plate_id,
            w.is_excluded,
            COUNT(p.timestamp) as n_timepoints
        FROM supabase.public.well_map_data w
        LEFT JOIN supabase.public.processed_data p
            ON w.plate_id = p.plate_id AND w.well_number = p.well_number
        WHERE w.drug != '' 
        AND w.drug IS NOT NULL
        AND w.concentration > 0  -- Exclude controls for this analysis
        GROUP BY w.drug, w.concentration, w.well_number, w.plate_id, w.is_excluded
    ),
    replicate_counts AS (
        SELECT 
            drug,
            concentration,
            COUNT(CASE WHEN NOT is_excluded AND n_timepoints >= 100 THEN 1 END) as valid_replicates,
            COUNT(*) as total_replicates
        FROM well_data
        GROUP BY drug, concentration
    )
    SELECT * FROM replicate_counts
    ORDER BY drug, concentration
    """
    
    print("Fetching replicate data...")
    df = conn.execute(query).df()
    
    # Also get overall drug summary
    drug_summary_query = """
    SELECT 
        drug,
        COUNT(DISTINCT concentration) as n_concentrations,
        AVG(valid_replicates) as avg_replicates,
        MIN(valid_replicates) as min_replicates,
        MAX(valid_replicates) as max_replicates
    FROM (
        SELECT 
            w.drug,
            w.concentration,
            COUNT(CASE WHEN NOT w.is_excluded THEN 1 END) as valid_replicates
        FROM supabase.public.well_map_data w
        WHERE w.drug != '' 
        AND w.drug IS NOT NULL
        AND w.concentration > 0
        GROUP BY w.drug, w.concentration
    ) t
    GROUP BY drug
    HAVING COUNT(DISTINCT concentration) >= 4
    """
    
    drug_summary = conn.execute(drug_summary_query).df()
    
    conn.close()
    
    return df, drug_summary

def create_replicate_visualizations(df, drug_summary):
    """Create comprehensive visualizations of replicate availability."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall distribution of replicate counts
    ax1 = fig.add_subplot(gs[0, :])
    replicate_dist = df['valid_replicates'].value_counts().sort_index()
    bars = ax1.bar(replicate_dist.index, replicate_dist.values, 
                    color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'][:len(replicate_dist)], 
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    ax1.set_xlabel('Number of Valid Replicates', fontsize=14)
    ax1.set_ylabel('Number of Drug/Concentration Combinations', fontsize=14)
    ax1.set_title('Distribution of Replicate Counts Across All Drug/Concentration Combinations', 
                  fontsize=16, fontweight='bold')
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add statistics box
    total_combos = len(df)
    complete_sets = (df['valid_replicates'] == 4).sum()
    incomplete_sets = (df['valid_replicates'] < 4).sum()
    single_replicate = (df['valid_replicates'] == 1).sum()
    
    stats_text = f"Total combinations: {total_combos}\n"
    stats_text += f"Complete (4 reps): {complete_sets} ({complete_sets/total_combos*100:.1f}%)\n"
    stats_text += f"Incomplete (<4 reps): {incomplete_sets} ({incomplete_sets/total_combos*100:.1f}%)\n"
    stats_text += f"Single replicate only: {single_replicate} ({single_replicate/total_combos*100:.1f}%)"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=12)
    
    # 2. Heatmap of replicate counts by drug (top 30 drugs by data volume)
    ax2 = fig.add_subplot(gs[1:3, :])
    
    # Pivot data for heatmap
    top_drugs = drug_summary.nlargest(30, 'n_concentrations')['drug'].tolist()
    df_top = df[df['drug'].isin(top_drugs)]
    
    heatmap_data = df_top.pivot_table(
        index='drug', 
        columns='concentration', 
        values='valid_replicates',
        aggfunc='first'
    )
    
    # Sort by average replicate count
    drug_avg_reps = heatmap_data.mean(axis=1).sort_values(ascending=False)
    heatmap_data = heatmap_data.loc[drug_avg_reps.index]
    
    # Create custom colormap (red=0, orange=1, yellow=2, lightgreen=3, darkgreen=4)
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen']
    n_bins = 5
    cmap = plt.matplotlib.colors.ListedColormap(colors[:n_bins])
    
    im = ax2.imshow(heatmap_data.values, cmap=cmap, aspect='auto', vmin=0, vmax=4)
    
    # Set ticks
    ax2.set_xticks(np.arange(len(heatmap_data.columns)))
    ax2.set_yticks(np.arange(len(heatmap_data.index)))
    ax2.set_xticklabels([f'{c:.2f}' if c < 1 else f'{c:.0f}' for c in heatmap_data.columns], 
                        rotation=45, ha='right')
    ax2.set_yticklabels(heatmap_data.index)
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            if not np.isnan(val):
                text = ax2.text(j, i, f'{int(val)}',
                               ha="center", va="center", color="black" if val > 2 else "white",
                               fontsize=8)
    
    ax2.set_xlabel('Concentration (μM)', fontsize=14)
    ax2.set_ylabel('Drug', fontsize=14)
    ax2.set_title('Replicate Counts by Drug and Concentration (Top 30 Drugs)', 
                  fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2, 3, 4])
    cbar.set_label('Number of Valid Replicates', fontsize=12)
    
    # 3. Drug-level summary statistics
    ax3 = fig.add_subplot(gs[3, 0])
    
    # Distribution of average replicates per drug
    avg_rep_bins = [0, 1, 2, 3, 3.5, 4.1]  # Adjusted upper bound
    drug_summary['avg_rep_binned'] = pd.cut(drug_summary['avg_replicates'], 
                                            bins=avg_rep_bins, 
                                            labels=['0-1', '1-2', '2-3', '3-3.5', '3.5-4'],
                                            include_lowest=True)
    
    avg_dist = drug_summary['avg_rep_binned'].value_counts()
    bars = ax3.bar(range(len(avg_dist)), avg_dist.values, 
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(avg_dist))))
    ax3.set_xticks(range(len(avg_dist)))
    ax3.set_xticklabels(avg_dist.index, rotation=45)
    ax3.set_xlabel('Average Replicates per Drug', fontsize=12)
    ax3.set_ylabel('Number of Drugs', fontsize=12)
    ax3.set_title('Distribution of Average Replicate Counts by Drug', fontsize=14)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. Concentration vs replicate availability
    ax4 = fig.add_subplot(gs[3, 1])
    
    # Group by concentration ranges
    conc_ranges = [(0, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1000)]
    conc_labels = ['0-0.1', '0.1-1', '1-10', '10-100', '100-1000']
    
    avg_reps_by_conc = []
    for low, high in conc_ranges:
        mask = (df['concentration'] > low) & (df['concentration'] <= high)
        avg_reps = df[mask]['valid_replicates'].mean() if mask.any() else 0
        avg_reps_by_conc.append(avg_reps)
    
    bars = ax4.bar(conc_labels, avg_reps_by_conc, 
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(conc_labels))))
    ax4.set_xlabel('Concentration Range (μM)', fontsize=12)
    ax4.set_ylabel('Average Valid Replicates', fontsize=12)
    ax4.set_title('Replicate Availability by Concentration Range', fontsize=14)
    ax4.set_ylim(0, 4.5)
    
    # Add horizontal line at 4
    ax4.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Complete set')
    ax4.legend()
    
    # 5. Drugs with incomplete replicate sets
    ax5 = fig.add_subplot(gs[3, 2])
    
    # Find drugs with the most incomplete sets
    incomplete_drugs = df[df['valid_replicates'] < 4].groupby('drug').size()
    incomplete_drugs = incomplete_drugs.sort_values(ascending=False).head(15)
    
    ax5.barh(range(len(incomplete_drugs)), incomplete_drugs.values, 
             color='coral', edgecolor='black')
    ax5.set_yticks(range(len(incomplete_drugs)))
    ax5.set_yticklabels(incomplete_drugs.index, fontsize=10)
    ax5.set_xlabel('Number of Incomplete Concentration Sets', fontsize=12)
    ax5.set_title('Top 15 Drugs with Incomplete Replicate Sets', fontsize=14)
    ax5.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(incomplete_drugs.values):
        ax5.text(v + 0.1, i, str(v), va='center')
    
    plt.suptitle('Replicate Availability Analysis for Oxygen Consumption Dataset', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = results_dir / "replicate_availability_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Also save data summary
    summary_path = results_dir / "replicate_summary_stats.txt"
    with open(summary_path, 'w') as f:
        f.write("REPLICATE AVAILABILITY SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total drug/concentration combinations: {len(df)}\n")
        f.write(f"Complete replicate sets (4): {(df['valid_replicates'] == 4).sum()} ({(df['valid_replicates'] == 4).sum()/len(df)*100:.1f}%)\n")
        f.write(f"3 replicates: {(df['valid_replicates'] == 3).sum()} ({(df['valid_replicates'] == 3).sum()/len(df)*100:.1f}%)\n")
        f.write(f"2 replicates: {(df['valid_replicates'] == 2).sum()} ({(df['valid_replicates'] == 2).sum()/len(df)*100:.1f}%)\n")
        f.write(f"1 replicate: {(df['valid_replicates'] == 1).sum()} ({(df['valid_replicates'] == 1).sum()/len(df)*100:.1f}%)\n")
        f.write(f"0 replicates: {(df['valid_replicates'] == 0).sum()} ({(df['valid_replicates'] == 0).sum()/len(df)*100:.1f}%)\n")
        f.write(f"\nTotal drugs analyzed: {len(drug_summary)}\n")
        f.write(f"Average replicates per drug: {drug_summary['avg_replicates'].mean():.2f}\n")
    
    print(f"Saved summary statistics to: {summary_path}")
    
    return output_path

def main():
    """Run the replicate availability analysis."""
    print("="*80)
    print("REPLICATE AVAILABILITY ANALYSIS")
    print("="*80)
    
    # Analyze replicate patterns
    df, drug_summary = analyze_replicate_patterns()
    
    print(f"\nAnalyzed {len(df)} drug/concentration combinations")
    print(f"Across {len(drug_summary)} drugs")
    
    # Create visualizations
    viz_path = create_replicate_visualizations(df, drug_summary)
    
    print("\n✅ Analysis complete!")
    print(f"Check the visualization at: {viz_path}")

if __name__ == "__main__":
    main()