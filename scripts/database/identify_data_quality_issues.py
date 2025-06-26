#!/usr/bin/env python
"""
Identify data quality issues in the database for exclusion or review.

This script identifies:
1. Wells with drug names that don't match the drug metadata table
2. Plates without proper media change events
3. Plates with late drug start events (>100 hours)

The output is a report of data that should be excluded or reviewed.

Usage:
    python scripts/database/identify_data_quality_issues.py
    
Output:
    - results/data/data_quality_issues.csv
    - Console report with statistics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.utils.data_loader import DataLoader


def identify_drug_consistency_issues(loader):
    """Find wells with drug names not in metadata."""
    print("\n=== DRUG CONSISTENCY ISSUES ===")
    
    # Load data
    wells = loader.load_well_metadata()
    drugs = loader.load_drug_metadata()
    
    # Get unique drug names from both sources
    well_drugs = set(wells['drug'].dropna().unique())
    metadata_drugs = set(drugs['drug'].unique())
    
    # Find missing drugs
    missing_drugs = well_drugs - metadata_drugs
    
    # Find affected wells
    affected_wells = wells[wells['drug'].isin(missing_drugs)].copy()
    affected_wells['issue_type'] = 'drug_not_in_metadata'
    affected_wells['issue_detail'] = affected_wells['drug']
    
    print(f"Found {len(missing_drugs)} drugs not in metadata:")
    for drug in sorted(missing_drugs)[:20]:  # Show first 20
        count = len(wells[wells['drug'] == drug])
        print(f"  - {drug}: {count} wells")
    
    if len(missing_drugs) > 20:
        print(f"  ... and {len(missing_drugs) - 20} more")
    
    print(f"\nTotal affected wells: {len(affected_wells)}")
    print(f"Affected plates: {affected_wells['plate_id'].nunique()}")
    
    return affected_wells[['plate_id', 'well_id', 'well_number', 'drug', 'issue_type', 'issue_detail']]


def identify_media_change_issues(loader):
    """Find plates without proper media change events."""
    print("\n=== MEDIA CHANGE EVENT ISSUES ===")
    
    # Load data
    all_events = loader.load_all_events()
    summary = loader.load_processed_data_summary()
    
    # Get experimental plates
    experimental_plates = set(summary['plate_id'].unique())
    
    # Find plates with media changes
    media_change_plates = set(
        all_events[all_events['title'] == 'Medium Change']['plate_id'].unique()
    )
    
    # Find plates without media changes
    plates_without_media_changes = experimental_plates - media_change_plates
    
    # Create dataframe of issues
    issues = []
    for plate_id in plates_without_media_changes:
        plate_info = summary[summary['plate_id'] == plate_id].iloc[0]
        issues.append({
            'plate_id': plate_id,
            'issue_type': 'no_media_changes',
            'issue_detail': f"No media change events found",
            'duration_hours': plate_info['duration_hours'],
            'unique_wells': plate_info['unique_wells']
        })
    
    issues_df = pd.DataFrame(issues)
    
    print(f"Plates without media changes: {len(plates_without_media_changes)} / {len(experimental_plates)} ({len(plates_without_media_changes)/len(experimental_plates)*100:.1f}%)")
    print("\nAffected plates:")
    for _, row in issues_df.head(10).iterrows():
        print(f"  - {row['plate_id']}: {row['duration_hours']:.1f}h duration, {row['unique_wells']} wells")
    
    if len(issues_df) > 10:
        print(f"  ... and {len(issues_df) - 10} more")
    
    return issues_df


def identify_drug_start_timing_issues(loader):
    """Find plates with late drug start events."""
    print("\n=== DRUG START TIMING ISSUES ===")
    
    # Load event timeline
    timeline = loader.load_event_timeline()
    
    issues = []
    
    # Check each plate
    for plate_id, plate_events in timeline.groupby('plate_id'):
        # Find drug start events
        drug_starts = plate_events[plate_events['event_type'] == 'Drugs Start']
        
        if not drug_starts.empty:
            first_drug_start = drug_starts.iloc[0]
            hours_since_start = first_drug_start['hours_since_start']
            
            if hours_since_start > 100:  # Late drug start
                issues.append({
                    'plate_id': plate_id,
                    'issue_type': 'late_drug_start',
                    'issue_detail': f"Drug start at {hours_since_start:.1f}h (>100h)",
                    'hours_since_start': hours_since_start
                })
    
    issues_df = pd.DataFrame(issues)
    
    print(f"Plates with late drug start: {len(issues_df)}")
    print("\nAffected plates:")
    for _, row in issues_df.head(10).iterrows():
        print(f"  - {row['plate_id']}: Drug start at {row['hours_since_start']:.1f}h")
    
    if len(issues_df) > 10:
        print(f"  ... and {len(issues_df) - 10} more")
    
    return issues_df


def identify_short_experiments(loader):
    """Find plates with unusually short duration."""
    print("\n=== SHORT EXPERIMENT ISSUES ===")
    
    summary = loader.load_processed_data_summary()
    
    # Find short experiments (< 300 hours)
    short_experiments = summary[summary['duration_hours'] < 300].copy()
    short_experiments['issue_type'] = 'short_experiment'
    short_experiments['issue_detail'] = short_experiments['duration_hours'].apply(
        lambda x: f"Duration {x:.1f}h (<300h)"
    )
    
    print(f"Plates with short duration: {len(short_experiments)}")
    print("\nAffected plates:")
    for _, row in short_experiments.head(10).iterrows():
        print(f"  - {row['plate_id']}: {row['duration_hours']:.1f}h duration, {row['unique_wells']} wells")
    
    if len(short_experiments) > 10:
        print(f"  ... and {len(short_experiments) - 10} more")
    
    return short_experiments[['plate_id', 'issue_type', 'issue_detail', 'duration_hours', 'unique_wells']]


def main():
    """Run all quality checks and generate report."""
    print("=== DATA QUALITY ISSUE IDENTIFICATION ===")
    print("This script identifies data that should be excluded or reviewed")
    
    # Create output directory
    output_dir = Path("results/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with DataLoader() as loader:
        # Run all checks
        drug_issues = identify_drug_consistency_issues(loader)
        media_issues = identify_media_change_issues(loader)
        timing_issues = identify_drug_start_timing_issues(loader)
        short_exp_issues = identify_short_experiments(loader)
        
        # Combine all issues
        all_issues = []
        
        # Add well-level drug issues
        for _, row in drug_issues.iterrows():
            all_issues.append({
                'level': 'well',
                'plate_id': row['plate_id'],
                'well_id': row.get('well_id', ''),
                'well_number': row.get('well_number', ''),
                'issue_type': row['issue_type'],
                'issue_detail': row['issue_detail'],
                'recommendation': 'exclude_well'
            })
        
        # Add plate-level issues
        for df, level, recommendation in [
            (media_issues, 'plate', 'review_or_exclude'),
            (timing_issues, 'plate', 'review'),
            (short_exp_issues, 'plate', 'exclude_plate')
        ]:
            for _, row in df.iterrows():
                all_issues.append({
                    'level': level,
                    'plate_id': row['plate_id'],
                    'well_id': '',
                    'well_number': '',
                    'issue_type': row['issue_type'],
                    'issue_detail': row['issue_detail'],
                    'recommendation': recommendation
                })
        
        # Create final report
        report_df = pd.DataFrame(all_issues)
        
        # Save report
        output_file = output_dir / "data_quality_issues.csv"
        report_df.to_csv(output_file, index=False)
        
        print(f"\n=== SUMMARY ===")
        print(f"Total issues found: {len(report_df)}")
        print(f"\nIssue breakdown:")
        print(report_df['issue_type'].value_counts())
        print(f"\nRecommendation breakdown:")
        print(report_df['recommendation'].value_counts())
        print(f"\nReport saved to: {output_file}")
        
        # Create summary for plates to exclude
        plates_to_exclude = report_df[
            report_df['recommendation'].isin(['exclude_plate', 'review_or_exclude'])
        ]['plate_id'].unique()
        
        wells_to_exclude = report_df[
            report_df['recommendation'] == 'exclude_well'
        ][['plate_id', 'well_id', 'well_number']]
        
        print(f"\n=== EXCLUSION SUMMARY ===")
        print(f"Plates to exclude/review: {len(plates_to_exclude)}")
        print(f"Wells to exclude: {len(wells_to_exclude)}")
        
        # Save exclusion lists
        pd.DataFrame({'plate_id': plates_to_exclude}).to_csv(
            output_dir / "plates_to_exclude.csv", index=False
        )
        wells_to_exclude.to_csv(
            output_dir / "wells_to_exclude.csv", index=False
        )


if __name__ == "__main__":
    main()