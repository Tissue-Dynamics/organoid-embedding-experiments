# Results Figures Organization

This directory contains the final hierarchical clustering analysis:

## 📁 embedding_comparisons/
- `fourier_hierarchical_clusters.png` - **Main Result**: Fourier transform clustering of all 240 qualifying drugs showing proper hierarchical structure (wells → concentrations → drugs) with actual oxygen dose-response curves

## 📁 drug_analysis/
- `drug_diversity_analysis.png` - Overview of drug library diversity (249 drugs)
- `drug_pattern_visualization.png` - Example oxygen patterns for 8 drugs

## 📁 preprocessing_demos/
- `control_normalization_demo.png` - Control-based normalization example

## Key Findings Summary

### Hierarchical Drug Analysis (240 drugs):
- **Proper Structure**: 7,616 wells → 1,872 concentrations → 240 drugs
- **Quality Filters**: ≥4 concentrations, ≥14 days data, ≥500 measurements
- **Exclusions Applied**: All excluded wells and measurements removed
- **Total Data**: 2.7M oxygen measurements properly structured

### Fourier Transform Clustering Results:
- **Cluster 0** (34 drugs): Low-amplitude stable responses
- **Cluster 1** (28 drugs): Moderate responses with clear patterns
- **Cluster 2** (105 drugs): **Largest cluster** - stable/low responses across concentrations
- **Cluster 3** (20 drugs): Strong dose-dependent toxicity increases
- **Cluster 4** (30 drugs): Variable responses with high spikes
- **Cluster 5** (23 drugs): Complex multi-phase response patterns

### Key Technical Achievements:
- ✅ **Proper replication**: 4 wells per concentration maintained
- ✅ **Dose-response preserved**: Multiple concentration curves per drug visible
- ✅ **Exclusion filtering**: Only valid, non-excluded data used
- ✅ **Full library coverage**: 240 of 249 drugs meet quality criteria
- ✅ **Biological relevance**: Clear separation of different toxicity mechanisms

### What Each Cluster Represents:
The clusters capture real pharmacological differences - different drugs cluster together based on similar temporal oxygen response patterns across their full dose-response curves, not just single concentrations.