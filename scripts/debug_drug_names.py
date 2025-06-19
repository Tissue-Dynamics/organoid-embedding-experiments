#!/usr/bin/env python3
"""
Debug Drug Names: Find why embeddings and DILI drugs don't overlap
"""

import pandas as pd
import joblib
from pathlib import Path

results_dir = Path("results/data")

# Load Phase 2 results
phase2 = joblib.load(results_dir / "hierarchical_embedding_results.joblib")
drug_metadata = phase2['drug_metadata']

# Load DILI data
wells_df = pd.read_parquet(results_dir / "wells_drugs_integrated.parquet")

print("=" * 80)
print("DRUG NAME INVESTIGATION")
print("=" * 80)

print(f"\n📊 PHASE 2 DRUG EMBEDDINGS:")
print(f"Number of drugs: {len(drug_metadata)}")
print(f"Sample drug names from embeddings:")
if hasattr(drug_metadata, 'index'):
    embed_drugs = list(drug_metadata.index)[:20]
else:
    embed_drugs = list(drug_metadata['drug'])[:20] if 'drug' in drug_metadata.columns else []
print(embed_drugs)

print(f"\n💊 DILI DRUGS:")
dili_drugs = wells_df[wells_df['dili'].notna()]['drug'].unique()
print(f"Number of drugs with DILI: {len(dili_drugs)}")
print(f"Sample drug names with DILI:")
print(list(dili_drugs)[:20])

print(f"\n🔍 CHECKING FOR OVERLAP:")
embed_drug_set = set(drug_metadata.index if hasattr(drug_metadata, 'index') else [])
dili_drug_set = set(dili_drugs)

# Check exact matches
exact_overlap = embed_drug_set.intersection(dili_drug_set)
print(f"Exact matches: {len(exact_overlap)}")

# Check case differences
embed_lower = {d.lower(): d for d in embed_drug_set}
dili_lower = {d.lower(): d for d in dili_drug_set}
case_matches = set(embed_lower.keys()).intersection(set(dili_lower.keys()))
print(f"Case-insensitive matches: {len(case_matches)}")

if case_matches:
    print(f"\nCase differences found:")
    for match in list(case_matches)[:10]:
        print(f"  Embedding: '{embed_lower[match]}' vs DILI: '{dili_lower[match]}'")

# Check for Sanofi drugs
print(f"\n🧪 SANOFI DRUG ANALYSIS:")
sanofi_embed = [d for d in embed_drug_set if 'Sanofi' in d]
sanofi_dili = [d for d in dili_drug_set if 'Sanofi' in d]
print(f"Sanofi drugs in embeddings: {len(sanofi_embed)}")
print(f"Sanofi drugs in DILI: {len(sanofi_dili)}")
print(f"Sample Sanofi embed: {sanofi_embed[:5]}")
print(f"Sample Sanofi DILI: {sanofi_dili[:5]}")

# Check what metadata contains
print(f"\n📋 DRUG METADATA STRUCTURE:")
print(f"Type: {type(drug_metadata)}")
if hasattr(drug_metadata, 'columns'):
    print(f"Columns: {drug_metadata.columns.tolist()}")
if hasattr(drug_metadata, 'shape'):
    print(f"Shape: {drug_metadata.shape}")
if hasattr(drug_metadata, 'head'):
    print(f"First few rows:")
    print(drug_metadata.head())

# Check all drugs in wells data
print(f"\n🔍 ALL DRUGS IN WELLS DATA:")
all_well_drugs = wells_df['drug'].unique()
print(f"Total unique drugs: {len(all_well_drugs)}")
print(f"Drugs starting with numbers: {sum(1 for d in all_well_drugs if d[0].isdigit())}")
print(f"Drugs containing 'Sanofi': {sum(1 for d in all_well_drugs if 'Sanofi' in d)}")

# Find pattern
print(f"\n🎯 PATTERN ANALYSIS:")
# Check if embeddings use Sanofi drugs
if len(sanofi_embed) > 100:
    print("Embeddings appear to be mostly Sanofi compounds")
    print("DILI data appears to be mostly named drugs")
    print("→ This explains the lack of overlap!")
    
    # Count how many wells have Sanofi drugs with DILI data
    sanofi_with_dili = wells_df[(wells_df['drug'].str.contains('Sanofi')) & (wells_df['dili'].notna())]
    print(f"\nSanofi drugs with DILI data: {sanofi_with_dili['drug'].nunique()}")
    if len(sanofi_with_dili) > 0:
        print(f"Sample Sanofi+DILI drugs: {sanofi_with_dili['drug'].unique()[:10].tolist()}")