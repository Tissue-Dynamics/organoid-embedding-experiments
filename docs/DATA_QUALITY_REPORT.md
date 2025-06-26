# Data Quality Report

Generated from `scripts/database/identify_data_quality_issues.py`

## Summary

Total issues identified: **35,017**

### Issue Breakdown:
- **Drug name inconsistencies**: 34,992 wells (86.1% of all wells)
- **Missing media change events**: 9 plates (26.5% of experimental plates)
- **Short experiments**: 9 plates (< 300 hours duration)
- **Late drug start**: 7 plates (drug start > 100 hours)

## Key Findings

### 1. Drug Name Issues (34,992 wells affected)

**109 drug names** in well metadata don't match the drug reference table:

- **Empty drug names**: 29,524 wells (largest issue)
- **Drug variants with .1 suffix**: e.g., "Tamoxifen.1", "Gemcitabine.1" (32 wells each)
- **Control conditions**: "Ctrl" (192 wells), "DMSO" (64 wells), "Full media"
- **Drug names with parentheses**: "Bortezomib (A09)", "Cobimetinib (E03)"
- **Different formulations**: "Copanlisib tris-HCl", "Aminolevulinic acid hydrochloride"
- **Media conditions**: "Hormones to Media", "Full Hormones", various cytokine conditions

### 2. Plates Without Media Changes (9 plates)

Plates with no "Medium Change" events recorded:
- Plate durations range from 0.6h to 628.8h
- Some may be test plates or abbreviated experiments
- One plate (5157acd5) only ran for 0.6 hours

### 3. Short Experiments (9 plates)

Plates with < 300 hour duration:
- **Extremely short**: 5157acd5 (0.6h), 10354e0a (26.8h)
- **Moderately short**: Several plates with 150-290 hour durations
- May represent pilot experiments or technical issues

### 4. Late Drug Start Events (7 plates)

Plates where drugs were added > 100 hours after start:
- **Extreme case**: Plate 10354e0a - drugs added at 5,425 hours (226 days!)
- Others range from 117-262 hours

## Recommendations

### Immediate Actions:
1. **Exclude wells with unmatched drug names** (34,992 wells)
2. **Review/exclude plates without media changes** (9 plates)
3. **Exclude short experiments** < 50 hours (2 plates)

### For Review:
1. Plates with late drug starts (may be valid delayed treatment experiments)
2. Drug name standardization (remove .1 suffixes, standardize formulations)
3. Control condition naming (standardize "Ctrl", "DMSO", media conditions)

## Data Files Generated

- `results/data/data_quality_issues.csv` - Full issue report
- `results/data/plates_to_exclude.csv` - List of problematic plates
- `results/data/wells_to_exclude.csv` - List of wells with drug name issues

## Next Steps

1. Create exclusion flags in the database
2. Standardize drug naming conventions
3. Document experimental protocols for unusual timings
4. Update data validation tests to reflect actual data patterns