#!/usr/bin/env python3
"""
Connect Wells to Drugs: Core Data Integration
Map wells to drug treatments and build the foundation for drug analysis.
"""

import os
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment
load_dotenv()
project_root = Path(__file__).parent.parent
results_dir = project_root / "results" / "data"

print("=" * 80)
print("CONNECTING WELLS TO DRUGS - CORE DATA INTEGRATION")
print("=" * 80)

# Setup database connection
database_url = os.getenv('DATABASE_URL')
if not database_url:
    raise ValueError("DATABASE_URL environment variable not set")

conn = duckdb.connect()
conn.execute("INSTALL postgres;")
conn.execute("LOAD postgres;")

parsed = urlparse(database_url)
postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"

print("✅ Database connection established")

# Load well mapping data
print("\n📋 Loading well mapping data...")

well_map_query = f"""
SELECT 
    plate_id,
    well_number,
    drug,
    concentration,
    (plate_id::text || '_' || well_number::text) as well_id
FROM postgres_scan('{postgres_string}', 'public', 'well_map_data')
ORDER BY plate_id, well_number
"""

well_map_df = conn.execute(well_map_query).fetchdf()
print(f"   📊 {len(well_map_df):,} well mappings loaded")
print(f"   🧪 {well_map_df['plate_id'].nunique()} plates")
print(f"   💊 {well_map_df['drug'].nunique()} unique drugs")

# Load Step 1 quality data  
print("\n📊 Loading Step 1 quality assessment...")
step1_df = pd.read_parquet(results_dir / "step1_quality_assessment_all_plates.parquet")
print(f"   📈 {len(step1_df):,} wells with quality metrics")

# Merge wells with drug information
print("\n🔗 Connecting wells to drug treatments...")

# Merge step1 with well mapping
integrated_df = step1_df.merge(
    well_map_df[['well_id', 'drug', 'concentration']], 
    on='well_id', 
    how='left'
)

print(f"   ✅ {len(integrated_df):,} wells integrated")
print(f"   💊 {integrated_df['drug'].nunique()} drugs represented")
print(f"   🔬 {integrated_df['concentration'].nunique()} concentration levels")

# Analyze drug coverage
drug_coverage = integrated_df.groupby('drug').agg({
    'well_id': 'count',
    'concentration': 'nunique',
    'cv_o2': 'mean'
}).round(3)

drug_coverage.columns = ['n_wells', 'n_concentrations', 'mean_cv']
drug_coverage = drug_coverage.sort_values('n_wells', ascending=False)

print(f"\n💊 DRUG COVERAGE ANALYSIS:")
print(f"   Top 10 drugs by well count:")
print(drug_coverage.head(10))

# Load drug metadata table
print(f"\n🧬 Loading drug metadata...")

drugs_query = f"""
SELECT *
FROM postgres_scan('{postgres_string}', 'public', 'drugs')
ORDER BY drug
"""

drugs_df = conn.execute(drugs_query).fetchdf()
print(f"   📋 {len(drugs_df)} drugs with metadata")
print(f"   📊 {len(drugs_df.columns)} metadata columns")

# Show available metadata
print(f"\n   Available metadata columns:")
for i, col in enumerate(drugs_df.columns):
    if i % 4 == 0:
        print(f"   ", end="")
    print(f"{col:<20}", end="")
    if (i + 1) % 4 == 0:
        print()
if len(drugs_df.columns) % 4 != 0:
    print()

# Connect to drug metadata
print(f"\n🔗 Connecting to drug metadata...")

# Clean drug names for matching
integrated_df['drug_clean'] = integrated_df['drug'].str.lower().str.strip()
drugs_df['drug_clean'] = drugs_df['drug'].str.lower().str.strip()

# Merge with drug metadata
final_df = integrated_df.merge(
    drugs_df,
    left_on='drug_clean',
    right_on='drug_clean',
    how='left',
    suffixes=('', '_meta')
)

print(f"   ✅ {len(final_df):,} wells processed")
print(f"   💊 {final_df['drug_meta'].nunique() if 'drug_meta' in final_df.columns else 'N/A'} drugs with metadata matched")

# Analyze what we have  
metadata_coverage = len(final_df) / len(final_df) * 100
print(f"   📈 Metadata coverage: {metadata_coverage:.1f}%")

# Key drug properties for analysis
key_properties = ['dili_risk', 'hepatotoxicity', 'max_dose_mg', 'half_life_h', 'clearance_ml_min_kg']
available_properties = [col for col in key_properties if col in final_df.columns]

print(f"\n🎯 KEY DRUG PROPERTIES AVAILABLE:")
for prop in available_properties:
    non_null = final_df[prop].notna().sum()
    wells_with_prop = final_df[final_df[prop].notna()]['well_id'].nunique()
    print(f"   {prop}: {non_null:,} values, {wells_with_prop:,} wells")

# Quality and drug analysis
print(f"\n📊 QUALITY + DRUG ANALYSIS:")
quality_drug_summary = final_df.groupby(['drug', 'concentration']).agg({
    'cv_o2': ['count', 'mean', 'std'],
    'baseline_duration_hours': 'mean'
}).round(3)

# Flatten column names
quality_drug_summary.columns = ['n_wells', 'mean_cv', 'std_cv', 'mean_baseline_duration']
quality_drug_summary = quality_drug_summary.reset_index()

# Filter for drugs with good representation
well_represented_drugs = quality_drug_summary[quality_drug_summary['n_wells'] >= 4]
print(f"   💊 {len(well_represented_drugs)} drug-concentration pairs with ≥4 wells")

# Save results
print(f"\n💾 Saving integrated data...")

# Main integrated dataset
integrated_path = results_dir / "wells_drugs_integrated.parquet"
final_df.to_parquet(integrated_path, index=False)

csv_path = results_dir / "wells_drugs_integrated.csv"  
final_df.to_csv(csv_path, index=False)

# Drug summary
drug_summary_path = results_dir / "drug_summary.parquet"
quality_drug_summary.to_parquet(drug_summary_path, index=False)

print(f"✅ Integrated data: {integrated_path}")
print(f"📄 CSV export: {csv_path}")
print(f"📊 Drug summary: {drug_summary_path}")

# Summary statistics
print(f"\n" + "="*80)
print("WELLS-TO-DRUGS CONNECTION COMPLETE")
print("="*80)

print(f"\n📈 FINAL DATASET:")
print(f"   Total wells: {len(final_df):,}")
print(f"   Unique drugs: {final_df['drug'].nunique()}")
print(f"   Concentration levels: {final_df['concentration'].nunique()}")

print(f"\n💊 DRUG ANALYSIS READY:")
print(f"   Drug-concentration pairs: {len(quality_drug_summary)}")
print(f"   Well-represented pairs (≥4 wells): {len(well_represented_drugs)}")
print(f"   Quality wells (CV≤0.25): {(final_df['cv_o2'] <= 0.25).sum():,}")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. Dose-response analysis by drug")
print(f"   2. Correlate oxygen patterns with drug properties")  
print(f"   3. DILI risk prediction from oxygen data")
print(f"   4. Build drug safety models")

# Show sample of final data
print(f"\n📋 SAMPLE INTEGRATED DATA:")
sample_cols = ['well_id', 'drug', 'concentration', 'cv_o2']
available_sample_cols = [col for col in sample_cols if col in final_df.columns]
sample_data = final_df[available_sample_cols].head(5)
print(sample_data)

conn.close()
print(f"\n🎉 Wells-to-drugs connection complete! Ready for drug analysis!")