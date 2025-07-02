#!/usr/bin/env python3
"""
Data Quality Verification - Check the discrepancy in DILI distribution
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import DataLoader

def verify_data_quality_and_dili():
    """Verify data quality and DILI distribution to resolve discrepancy."""
    
    print("🔍 DATA QUALITY VERIFICATION")
    print("=" * 80)
    print("Investigating discrepancy in DILI distribution reporting")
    print("=" * 80)
    
    # Load raw data
    with DataLoader() as loader:
        oxygen_data = loader.load_oxygen_data()
    
    drug_metadata = pd.read_csv('data/database/drug_rows.csv')
    
    print(f"\n📊 RAW DATA OVERVIEW:")
    print(f"Total oxygen measurements: {len(oxygen_data):,}")
    print(f"Unique drugs in oxygen data: {oxygen_data['drug'].nunique()}")
    print(f"Unique drugs in metadata: {drug_metadata['drug'].nunique()}")
    
    # Analyze data points per drug
    data_points_per_drug = oxygen_data['drug'].value_counts().sort_values(ascending=False)
    
    print(f"\n📈 DATA POINTS PER DRUG:")
    print(f"Mean: {data_points_per_drug.mean():.0f}")
    print(f"Median: {data_points_per_drug.median():.0f}")
    print(f"Min: {data_points_per_drug.min()}")
    print(f"Max: {data_points_per_drug.max():,}")
    
    # Quality categories
    excellent = data_points_per_drug[data_points_per_drug >= 10000]
    very_good = data_points_per_drug[(data_points_per_drug >= 5000) & (data_points_per_drug < 10000)]
    good = data_points_per_drug[(data_points_per_drug >= 1000) & (data_points_per_drug < 5000)]
    poor = data_points_per_drug[data_points_per_drug < 1000]
    
    print(f"\n🎯 DATA QUALITY BREAKDOWN:")
    print(f"Excellent (≥10k points): {len(excellent)} drugs")
    print(f"Very Good (5k-10k): {len(very_good)} drugs")
    print(f"Good (1k-5k): {len(good)} drugs")
    print(f"Poor (<1k): {len(poor)} drugs")
    
    # Check DILI distribution for each quality level
    print(f"\n🚨 DETAILED DILI ANALYSIS BY DATA QUALITY:")
    
    for quality_name, drug_list in [
        ("Excellent (≥10k)", excellent.index),
        ("Very Good (5k-10k)", very_good.index),
        ("Good (1k-5k)", good.index),
        ("Poor (<1k)", poor.index)
    ]:
        print(f"\n{quality_name}: {len(drug_list)} drugs")
        
        if len(drug_list) == 0:
            continue
        
        # Get DILI info for these drugs
        quality_metadata = drug_metadata[drug_metadata['drug'].isin(drug_list)]
        
        print(f"  Drugs with metadata: {len(quality_metadata)}")
        
        # DILI distribution
        dili_counts = quality_metadata['dili'].value_counts()
        print(f"  DILI distribution:")
        
        dili_positive = 0
        dili_negative = 0
        
        for dili_cat, count in dili_counts.items():
            print(f"    {dili_cat}: {count} drugs")
            
            if dili_cat in ['vMost-DILI-Concern', 'vLess-DILI-Concern']:
                dili_positive += count
            elif dili_cat in ['vNo-DILI-Concern']:
                dili_negative += count
        
        total_with_dili = dili_positive + dili_negative
        if total_with_dili > 0:
            print(f"  DILI Summary:")
            print(f"    DILI Positive (Most/Less): {dili_positive} ({dili_positive/total_with_dili*100:.1f}%)")
            print(f"    DILI Negative (No): {dili_negative} ({dili_negative/total_with_dili*100:.1f}%)")
        
        # Unknown/missing DILI
        unknown_dili = len(quality_metadata) - total_with_dili
        if unknown_dili > 0:
            print(f"    Unknown/Missing DILI: {unknown_dili}")
    
    # Detailed analysis of individual drugs
    print(f"\n📋 TOP 20 DRUGS BY DATA VOLUME:")
    top_drugs = data_points_per_drug.head(20)
    
    for drug, count in top_drugs.items():
        drug_meta = drug_metadata[drug_metadata['drug'] == drug]
        
        if len(drug_meta) > 0:
            dili = drug_meta.iloc[0]['dili']
        else:
            dili = 'No Metadata'
        
        print(f"  {drug}: {count:,} points, DILI: {dili}")
    
    # Check for data loading issues
    print(f"\n🔧 DATA LOADING VERIFICATION:")
    
    # Load features from our previous analysis
    try:
        features_df = pd.read_csv('results/data/pk_oxygen_features_comprehensive.csv')
        print(f"✓ Previous analysis features: {len(features_df)} drugs")
        
        # Check DILI distribution in features
        feature_dili = features_df['dili'].value_counts()
        print(f"  DILI in features data:")
        for dili_cat, count in feature_dili.items():
            print(f"    {dili_cat}: {count}")
    except:
        print("✗ Could not load previous features")
    
    # Create visualization
    create_data_quality_visualization(data_points_per_drug, drug_metadata)
    
    # Generate corrected analysis
    return analyze_corrected_dili_distribution(oxygen_data, drug_metadata)

def analyze_corrected_dili_distribution(oxygen_data, drug_metadata):
    """Analyze the correct DILI distribution."""
    
    print(f"\n🎯 CORRECTED DILI ANALYSIS")
    print("=" * 60)
    
    # Get all drugs with oxygen data
    oxygen_drugs = oxygen_data['drug'].unique()
    
    # Merge with metadata
    merged_data = drug_metadata[drug_metadata['drug'].isin(oxygen_drugs)].copy()
    
    # Add data point counts
    data_counts = oxygen_data['drug'].value_counts()
    merged_data['data_points'] = merged_data['drug'].map(data_counts)
    
    print(f"Total drugs with both oxygen data and metadata: {len(merged_data)}")
    
    # DILI distribution
    print(f"\nOverall DILI Distribution:")
    dili_counts = merged_data['dili'].value_counts()
    total_with_dili = 0
    
    for dili_cat, count in dili_counts.items():
        if pd.notna(dili_cat):
            print(f"  {dili_cat}: {count} drugs ({count/len(merged_data)*100:.1f}%)")
            total_with_dili += count
    
    unknown_count = len(merged_data) - total_with_dili
    print(f"  Unknown/NaN: {unknown_count} drugs ({unknown_count/len(merged_data)*100:.1f}%)")
    
    # Binary DILI classification
    print(f"\nBinary DILI Classification:")
    
    # Map to binary
    binary_map = {
        'vNo-DILI-Concern': 'No DILI',
        'vLess-DILI-Concern': 'DILI Positive',
        'vMost-DILI-Concern': 'DILI Positive',
        'Ambiguous DILI-concern': 'Ambiguous'
    }
    
    merged_data['dili_binary'] = merged_data['dili'].map(binary_map)
    binary_counts = merged_data['dili_binary'].value_counts()
    
    for category, count in binary_counts.items():
        if pd.notna(category):
            print(f"  {category}: {count} drugs ({count/len(merged_data)*100:.1f}%)")
    
    # Quality-stratified analysis
    print(f"\nQuality-Stratified DILI Analysis:")
    
    # Define quality tiers
    merged_data['quality_tier'] = pd.cut(
        merged_data['data_points'],
        bins=[0, 1000, 5000, 10000, float('inf')],
        labels=['Poor (<1k)', 'Good (1k-5k)', 'Very Good (5k-10k)', 'Excellent (≥10k)']
    )
    
    for tier in ['Excellent (≥10k)', 'Very Good (5k-10k)', 'Good (1k-5k)', 'Poor (<1k)']:
        tier_data = merged_data[merged_data['quality_tier'] == tier]
        
        if len(tier_data) == 0:
            continue
        
        print(f"\n  {tier}: {len(tier_data)} drugs")
        
        tier_dili = tier_data['dili_binary'].value_counts()
        tier_total = tier_dili.sum()
        
        for dili_cat, count in tier_dili.items():
            if pd.notna(dili_cat):
                print(f"    {dili_cat}: {count} ({count/tier_total*100:.1f}%)")
        
        # Average data points
        avg_points = tier_data['data_points'].mean()
        print(f"    Average data points: {avg_points:,.0f}")
    
    return merged_data

def create_data_quality_visualization(data_points_per_drug, drug_metadata):
    """Create visualization of data quality issues."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Verification Analysis', fontsize=16, fontweight='bold')
    
    # 1. Distribution of data points per drug
    ax1 = axes[0, 0]
    ax1.hist(data_points_per_drug.values, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Data Points per Drug')
    ax1.set_ylabel('Number of Drugs')
    ax1.set_title('Distribution of Data Points per Drug', fontweight='bold')
    ax1.axvline(data_points_per_drug.mean(), color='red', linestyle='--', 
               label=f'Mean: {data_points_per_drug.mean():.0f}')
    ax1.axvline(data_points_per_drug.median(), color='orange', linestyle='--', 
               label=f'Median: {data_points_per_drug.median():.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Quality tiers
    ax2 = axes[0, 1]
    
    excellent = (data_points_per_drug >= 10000).sum()
    very_good = ((data_points_per_drug >= 5000) & (data_points_per_drug < 10000)).sum()
    good = ((data_points_per_drug >= 1000) & (data_points_per_drug < 5000)).sum()
    poor = (data_points_per_drug < 1000).sum()
    
    categories = ['Excellent\n(≥10k)', 'Very Good\n(5k-10k)', 'Good\n(1k-5k)', 'Poor\n(<1k)']
    counts = [excellent, very_good, good, poor]
    colors = ['green', 'lightgreen', 'yellow', 'red']
    
    ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Drugs')
    ax2.set_title('Data Quality Distribution', fontweight='bold')
    
    # Add count labels
    for i, count in enumerate(counts):
        ax2.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. Top drugs by data volume
    ax3 = axes[1, 0]
    
    top_20 = data_points_per_drug.head(20)
    
    # Get DILI info for top drugs
    colors = []
    for drug in top_20.index:
        drug_meta = drug_metadata[drug_metadata['drug'] == drug]
        if len(drug_meta) > 0:
            dili = drug_meta.iloc[0]['dili']
            if dili == 'vMost-DILI-Concern':
                colors.append('red')
            elif dili == 'vLess-DILI-Concern':
                colors.append('orange')
            elif dili == 'vNo-DILI-Concern':
                colors.append('green')
            else:
                colors.append('gray')
        else:
            colors.append('black')
    
    y_pos = np.arange(len(top_20))
    ax3.barh(y_pos, top_20.values, color=colors, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_20.index, fontsize=8)
    ax3.set_xlabel('Data Points')
    ax3.set_title('Top 20 Drugs by Data Volume\n(Colors: Red=Most DILI, Orange=Less DILI, Green=No DILI)', 
                  fontweight='bold', fontsize=10)
    
    # 4. DILI distribution summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate DILI summary
    drugs_with_data = data_points_per_drug.index
    metadata_subset = drug_metadata[drug_metadata['drug'].isin(drugs_with_data)]
    
    dili_summary = metadata_subset['dili'].value_counts()
    
    summary_text = f"""
    DATA QUALITY SUMMARY
    ══════════════════════════════════════
    
    Total Drugs: {len(data_points_per_drug)}
    Total Data Points: {data_points_per_drug.sum():,}
    
    Quality Breakdown:
    • Excellent (≥10k): {excellent} drugs
    • Very Good (5k-10k): {very_good} drugs  
    • Good (1k-5k): {good} drugs
    • Poor (<1k): {poor} drugs
    
    DILI Distribution:
    """
    
    for dili_cat, count in dili_summary.items():
        if pd.notna(dili_cat):
            summary_text += f"• {dili_cat}: {count}\n    "
    
    unknown = len(metadata_subset) - dili_summary.sum()
    summary_text += f"• Unknown/Missing: {unknown}"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/figures/data_quality_verification.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ Data quality visualization saved to: {output_path}")

if __name__ == "__main__":
    verify_data_quality_and_dili()