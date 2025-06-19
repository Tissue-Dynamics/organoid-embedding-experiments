#!/usr/bin/env python3
"""
Investigate Well Exclusions: Why so few wells in analysis?
Track what's excluding wells at each step.
"""

import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
results_dir = project_root / "results" / "data"

print("=" * 80)
print("INVESTIGATING WELL EXCLUSIONS")
print("=" * 80)

# Load the integrated data
df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")

print(f"📊 STEP-BY-STEP EXCLUSION ANALYSIS:")
print(f"   Starting wells: {len(df):,}")

# Step 1: Quality filter
print(f"\n🔍 STEP 1: Quality Filter (CV ≤ 0.25)")
quality_mask = df['cv_o2'] <= 0.25
print(f"   Before: {len(df):,} wells")
print(f"   After:  {quality_mask.sum():,} wells ({quality_mask.mean()*100:.1f}%)")
print(f"   Excluded: {(~quality_mask).sum():,} wells with high CV")

quality_df = df[quality_mask].copy()

# Step 2: DILI data availability
print(f"\n🔍 STEP 2: DILI Data Availability")
dili_mask = quality_df['dili'].notna()
print(f"   Before: {len(quality_df):,} quality wells")
print(f"   After:  {dili_mask.sum():,} wells ({dili_mask.mean()*100:.1f}%)")
print(f"   Excluded: {(~dili_mask).sum():,} wells missing DILI data")

# Investigate what drugs are missing DILI data
missing_dili = quality_df[~dili_mask]
print(f"\n   🔬 Drugs missing DILI data:")
missing_drugs = missing_dili['drug'].value_counts().head(10)
print(missing_drugs)

# Step 3: Drug representation filter (≥8 wells)
dili_df = quality_df[dili_mask].copy()
print(f"\n🔍 STEP 3: Drug Representation Filter (≥8 wells per drug)")

drug_counts = dili_df['drug'].value_counts()
well_represented_drugs = drug_counts[drug_counts >= 8].index
represented_mask = dili_df['drug'].isin(well_represented_drugs)

print(f"   Before: {len(dili_df):,} wells with DILI data")
print(f"   After:  {represented_mask.sum():,} wells ({represented_mask.mean()*100:.1f}%)")
print(f"   Excluded: {(~represented_mask).sum():,} wells from under-represented drugs")

print(f"\n   📊 Drug representation breakdown:")
print(f"   Drugs with ≥8 wells: {(drug_counts >= 8).sum()}")
print(f"   Drugs with 4-7 wells: {((drug_counts >= 4) & (drug_counts < 8)).sum()}")
print(f"   Drugs with 1-3 wells: {(drug_counts < 4).sum()}")

# Check what's in Phase 2 hierarchical embeddings
print(f"\n" + "="*80)
print("PHASE 2 HIERARCHICAL EMBEDDINGS CHECK")
print("="*80)

# Try to load Phase 2 results
try:
    import joblib
    phase2_results = joblib.load(results_dir / "hierarchical_embedding_results.joblib")
    
    print(f"✅ Phase 2 results found!")
    print(f"   Keys: {list(phase2_results.keys())}")
    
    if 'drug_embeddings' in phase2_results:
        drug_embeddings = phase2_results['drug_embeddings']
        print(f"   📊 Drug embeddings: {len(drug_embeddings)} drugs")
        print(f"   🔬 Embedding dimensions: {drug_embeddings.shape if hasattr(drug_embeddings, 'shape') else 'Unknown'}")
        
        # Check if these drugs have DILI data
        if isinstance(drug_embeddings, pd.DataFrame):
            phase2_drugs = set(drug_embeddings.index if hasattr(drug_embeddings, 'index') else [])
        else:
            phase2_drugs = set()
        
        current_dili_drugs = set(dili_df['drug'].unique())
        overlap = phase2_drugs.intersection(current_dili_drugs)
        
        print(f"   🔗 Overlap with DILI drugs: {len(overlap)} drugs")
        print(f"   📈 Coverage: {len(overlap)/len(phase2_drugs)*100:.1f}% of Phase 2 drugs have DILI data")
        
except Exception as e:
    print(f"❌ Phase 2 results not found or corrupted: {e}")

# Detailed breakdown by exclusion reason
print(f"\n" + "="*80)
print("DETAILED EXCLUSION BREAKDOWN")
print("="*80)

# Create exclusion tracking
exclusion_analysis = pd.DataFrame({
    'total_wells': len(df),
    'high_cv': (~quality_mask).sum(),
    'missing_dili': (~dili_mask).sum(),
    'under_represented': (~represented_mask).sum(),
    'final_analysis': represented_mask.sum()
}, index=[0])

print(f"📊 EXCLUSION FUNNEL:")
print(f"   Total wells in database: {len(df):,}")
print(f"   └─ Excluded for high CV (>0.25): {(~quality_mask).sum():,}")
print(f"      └─ Quality wells remaining: {quality_mask.sum():,}")
print(f"         └─ Excluded for missing DILI data: {(~dili_mask).sum():,}")
print(f"            └─ Wells with DILI data: {dili_mask.sum():,}")
print(f"               └─ Excluded from under-represented drugs: {(~represented_mask).sum():,}")
print(f"                  └─ Final analysis wells: {represented_mask.sum():,}")

# Check concentration distribution
print(f"\n🧪 CONCENTRATION ANALYSIS:")
conc_analysis = dili_df.groupby('drug')['concentration'].nunique().describe()
print(f"   Concentrations per drug:")
print(conc_analysis.round(1))

# Save exclusion analysis
exclusion_details = {
    'total_wells': len(df),
    'quality_wells': quality_mask.sum(),
    'dili_wells': dili_mask.sum(),
    'final_wells': represented_mask.sum(),
    'quality_pct': quality_mask.mean() * 100,
    'dili_coverage_pct': dili_mask.mean() * 100,
    'representation_pct': represented_mask.mean() * 100,
    'drugs_total': df['drug'].nunique(),
    'drugs_with_dili': dili_df['drug'].nunique(),
    'drugs_well_represented': len(well_represented_drugs)
}

print(f"\n📋 SUMMARY STATISTICS:")
for key, value in exclusion_details.items():
    if 'pct' in key:
        print(f"   {key}: {value:.1f}%")
    else:
        print(f"   {key}: {value:,}")

# Recommendations
print(f"\n" + "="*80)
print("RECOMMENDATIONS TO INCREASE WELL COUNT")
print("="*80)

print(f"🎯 OPTIONS TO GET MORE WELLS:")
print(f"   1. Relax CV threshold (0.25 → 0.5): +{((df['cv_o2'] <= 0.5) & (df['cv_o2'] > 0.25)).sum():,} wells")
print(f"   2. Include drugs with ≥4 wells: +{((drug_counts >= 4) & (drug_counts < 8)).sum():,} drugs")
print(f"   3. Use Phase 2 hierarchical approach: Drug-level instead of well-level")
print(f"   4. Include partial DILI data: Use available subset")

print(f"\n💡 RECOMMENDED APPROACH:")
print(f"   • Use Phase 2 hierarchical drug embeddings (240 drugs)")
print(f"   • Correlate with drug properties at drug level")
print(f"   • Much better statistical power than well-level analysis")

print(f"\n🔍 Phase 2 vs Current Comparison:")
print(f"   Phase 2: ~240 drugs with embeddings")
print(f"   Current: {len(well_represented_drugs)} drugs with ≥8 wells")
print(f"   → Phase 2 approach gives 2x more drugs for analysis!")