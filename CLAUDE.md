# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for comparing embedding methods on liver organoid oxygen time series data. The project evaluates traditional methods (DTW, Fourier, SAX), feature-based approaches (TSFresh, catch22), and deep learning methods (autoencoders, transformers, triplet networks) on ~7,680 time series from 240 drug treatments.

**CURRENT STATUS: Hierarchical drug embeddings COMPLETE. Ready for drug correlation analysis.**

### Recent Accomplishments
- ✅ Implemented hierarchical drug embeddings: 7,616 wells → 1,872 concentrations → 240 drugs
- ✅ Generated 5 embedding method visualizations with concentration-based coloring
- ✅ Fixed data aggregation to preserve dose-response patterns (no improper averaging)
- ✅ Applied proper data filtering (is_excluded=false, ≥4 concentrations, ≥14 days)
- ✅ Connected to drugs table with 198 drugs and 71 metadata columns for correlation analysis

### Next Priority
Cross embeddings with drug properties to correlate embedding components with DILI risk, hepatotoxicity, pharmacokinetics, and chemical descriptors.

## Critical Lessons Learned

### ⚠️ Single-Drug Outlier Correlation Problem
**DO NOT attempt structure-function correlation analysis using molecular fingerprints vs oxygen embeddings.** 

**Why this fails:**
- With 155 drugs, any single outlier drug will appear to "correlate" with some molecular feature by chance
- Creates spurious r=0.9+ correlations that look impressive but are meaningless
- The "1 vs 154" comparison problem: when only one drug has extreme values, correlations are guaranteed
- Synthetic controls confirm that random single outliers generate similar correlation strengths

**Evidence from failed analysis:**
- 17/20 "top correlations" were single-drug outliers (Fulvestrant, Sitaxentan, Alectinib)
- When outliers removed, correlations collapsed (r=0.933 → r=-1.000)
- No proper negative controls or biological replicability
- Permutation tests passed but preserved the same flawed 1-vs-154 structure

**Correct approach for structure-function:** Need multiple drugs with similar structures showing consistent patterns, proper negative controls, and validation across drug families - not individual outlier correlations.

## Development Commands

```bash
# Install dependencies with uv (USER PREFERENCE: Always use uv!)
uv pip install -r requirements.txt
uv pip install "numpy<2.0"  # Required for TSFresh compatibility

# Database credentials (already configured)
export DATABASE_URL="postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"

# Current analysis scripts (READY TO USE)
uv run python scripts/analysis/hierarchical_cluster_oxygen_visualization.py  # Main embeddings script
uv run python scripts/analysis/explore_drugs_table.py  # Drug metadata exploration
uv run python scripts/analysis/drug_embedding_correlation_analysis.py  # Drug correlations
uv run python scripts/analysis/louvain_clustering_all_embeddings.py  # Clustering analysis
```

## Project Structure

```
├── data/                      # Data processing modules
│   ├── preprocessing/        # Data cleaning and normalization
│   └── raw/                 # Raw data files (parquet)
├── embeddings/               # All embedding implementations
│   ├── traditional/         # DTW, Fourier, SAX
│   ├── features/           # TSFresh, catch22, custom
│   └── deep_learning/      # Autoencoders, transformers
├── evaluation/              # Metrics and visualization tools
├── scripts/                # Analysis scripts
│   ├── analysis/          # Data analysis and visualization
│   └── experiments/       # Experiment runners
├── results/                # Generated outputs
│   ├── data/             # Processed embeddings and results
│   ├── embeddings/       # Specialized embedding outputs
│   └── figures/          # All visualizations
└── docs/                  # Documentation and summaries
```

## Architecture and Key Components

### Data Layer
- **`data/raw/`** - Raw data files (processed_data_updated.parquet, well_map_data_updated.parquet)
- **`data/preprocessing/`** - Data cleaning and preprocessing utilities
  - Database connections handled directly via DuckDB with PostgreSQL backend
  - Analysis scripts load data directly from Supabase using environment variables

### Embedding Modules
- **`embeddings/traditional/`** - DTW, Fourier Transform, SAX implementations
- **`embeddings/features/`** - TSFresh, catch22, custom organoid-specific features
- **`embeddings/deep_learning/`** - Autoencoders (LSTM/CNN), Transformers, Triplet Networks

### Data Preprocessing
- **`data/preprocessing/`** - Comprehensive preprocessing utilities
  - `cleaner.py` - Quality assessment, duplicate removal, NaN handling
  - `normalizer.py` - Various normalization strategies including control-based
  - `interpolation.py` - Missing value interpolation with multiple methods
  - `outlier_detection.py` - Statistical and ML-based outlier detection
  - `event_correction.py` - Media change artifact correction

### Data Characteristics
- ~7,680 time series with irregular hourly sampling
- Requires handling missing data and media change artifacts
- Control organoids used for baseline correction
- Data spans ~2 weeks with drug treatments at 8 concentrations, 4 replicates each

### Current Analysis Scripts
- **`scripts/analysis/hierarchical_cluster_oxygen_visualization.py`** - Main hierarchical embedding pipeline
- **`scripts/analysis/drug_embedding_correlation_analysis.py`** - Drug property correlations
- **`scripts/analysis/louvain_clustering_all_embeddings.py`** - Community detection clustering
- **`scripts/analysis/explore_drugs_table.py`** - Drug metadata exploration

### Evaluation and Visualization
- **`evaluation/metrics/`** - Comprehensive embedding quality assessment
  - Clustering metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Neighborhood preservation (trustworthiness, continuity)
  - Dimensionality analysis and stability measures
- **`evaluation/visualization/`** - Rich plotting utilities for embeddings and comparisons

### Configuration Management
- Environment variables for Supabase credentials via `python-dotenv`
- Direct configuration within analysis scripts
- Results saved as joblib files and CSV data

### Current State
This is a **fully implemented research codebase** with comprehensive embedding methods, preprocessing utilities, experiment management, and evaluation tools. All major components are functional and ready for organoid time series analysis.

**LATEST PROGRESS:** Hierarchical drug embeddings are complete with 5 methods (Fourier, SAX, catch22, TSFresh, Custom). Generated cluster visualizations showing concentration-based dose-response patterns. Drug metadata table explored with 71 properties including DILI risk, hepatotoxicity flags, and pharmacokinetic data.

### Key Results Generated
- `results/figures/embedding_comparisons/fourier_hierarchical_clusters.png`
- `results/figures/embedding_comparisons/tsfresh_hierarchical_clusters.png`
- `results/figures/embedding_comparisons/catch22_hierarchical_clusters.png`
- `results/figures/embedding_comparisons/sax_hierarchical_clusters.png`
- `results/figures/embedding_comparisons/custom_hierarchical_clusters.png`

### Drug Dataset Summary
- **240 qualifying drugs** (≥4 concentrations, ≥14 days data)
- **1,872 concentration-level embeddings** (~7.8 concentrations per drug)
- **7,616 well-level time series** (4 replicates per concentration)
- **198 drugs in metadata table** with 71 properties including DILI, hepatotoxicity, PK data

### Critical Technical Notes
- **Use uv for all package management** (user preference)
- **numpy<2.0 required** for TSFresh compatibility
- **Database connection via DuckDB** with PostgreSQL backend to Supabase
- **Hierarchical structure preserved**: wells → concentrations → drugs (no improper averaging)
- **Data filtering applied**: is_excluded=false throughout pipeline
- **Concentration-based coloring** in visualizations (not drug-based) for dose-response

## Technology Stack
- **Time Series**: tslearn, tsfresh, catch22, pyts, stumpy
- **Deep Learning**: PyTorch, TensorFlow, tsai (Time Series AI)
- **Data Processing**: pandas, numpy, scipy, scikit-learn
- **Database**: Supabase client
- **Visualization**: matplotlib, seaborn, plotly, umap-learn