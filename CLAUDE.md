# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for comparing embedding methods on liver organoid oxygen time series data. The project evaluates traditional methods (DTW, Fourier, SAX), feature-based approaches (TSFresh, catch22), and deep learning methods (autoencoders, transformers, triplet networks) on ~7,680 time series from 240 drug treatments.

**CURRENT STATUS: Step 3 Media Change Event Detection COMPLETE. Ready for Step 4 dose-response normalization.**

### Recent Accomplishments
- ✅ **Step 1 Complete**: Enhanced data loader with quality flags and baseline detection
  - Quality flag implementation: 6 flags (low_points, high_noise, sensor_drift, baseline_unstable, replicate_discord, media_change_outlier)
  - Event-relative baseline detection: 88.3% wells using pre-dosing periods
  - All 33 plates processed: 10,930 wells with comprehensive quality assessment
- ✅ **Step 2 Complete**: Multi-timescale feature extraction pipeline
  - Rolling window implementation: 24h, 48h, 96h windows with 50% overlap
  - catch22 features: 22 time series features per window (25 total including metadata)
  - Hierarchical SAX features: Coarse/medium/fine levels (43 features total)
  - Comprehensive feature set: 80 features per window (12 basic + 25 catch22 + 43 SAX)
  - Successfully tested on 10 wells: 485 feature records generated
- ✅ **Step 3 Complete**: Media change event detection and event-aware features
  - Event detection from database: 103 media change events across 29 plates
  - Spike characterization: Peak height, recovery time, baseline shifts
  - Inter-event feature extraction: 244 feature records from periods between media changes
  - Event-indexed catch22/SAX features avoiding media change artifacts

### Next Priority
**Step 4**: Dose-response normalization with Hill curve fitting for cross-drug comparability.

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

# Step 1: Data Pipeline Foundation (COMPLETE)
uv run python data/step1_duckdb_foundation.py  # Complete Step 1 with DuckDB

# Step 2: Multi-Timescale Feature Extraction (COMPLETE)
uv run python data/step2_multiscale_features.py  # Extract features at 24h/48h/96h scales

# Step 3: Media Change Event Detection (COMPLETE)
uv run python data/step3_event_detection.py  # Detect and characterize media change events

# Event data pipeline
uv run python scripts/database/download_event_data.py  # Download event data from Supabase

# Legacy analysis scripts
uv run python scripts/analysis/hierarchical_cluster_oxygen_visualization.py  # Main embeddings script
uv run python scripts/analysis/explore_drugs_table.py  # Drug metadata exploration
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
│   ├── database/          # Event data download and integration
│   └── experiments/       # Experiment runners
├── results/                # Generated outputs
│   ├── data/             # Processed embeddings and results
│   ├── embeddings/       # Specialized embedding outputs
│   └── figures/          # All visualizations
└── docs/                  # Documentation and summaries
```

## Architecture and Key Components

### Data Layer
- **`data/raw/`** - Raw data files:
  - `processed_data_updated.parquet` - Main oxygen time series data
  - `well_map_data_updated.parquet` - Well metadata and drug information  
  - `event_data.parquet` - **NEW**: Real experimental events (dosing, media changes) downloaded from Supabase
  - `plate_event_summary.parquet` - **NEW**: Plate-level event summary for quick access
  - `event_enhanced_sample.parquet` - **NEW**: Sample data with event features for testing
- **`data/preprocessing/`** - Data cleaning and preprocessing utilities
- **`scripts/database/`** - **NEW**: Event data pipeline using DuckDB with PostgreSQL backend to Supabase

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
- **`scripts/analysis/comprehensive_oxygen_data_analysis.py`** - **NEW**: Complete data characterization and documentation
- **`scripts/analysis/analyze_control_periods.py`** - **NEW**: Control period and baseline analysis
- **`scripts/database/download_event_data.py`** - **NEW**: Downloads experimental events from Supabase
- **`scripts/database/quick_event_integration.py`** - **NEW**: Integrates events with oxygen data

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

**LATEST PROGRESS:** Event-aware feature engineering foundation complete. Downloaded real experimental events (dosing, media changes) from Supabase database. Created integrated data pipeline combining oxygen time series with event timing. Comprehensive data analysis and documentation generated. Ready for advanced multi-timescale feature extraction with pharmacological grounding.

### Key Results Generated
- `results/figures/data_characteristics/oxygen_data_characteristics.png` - **NEW**: Comprehensive data overview
- `results/figures/data_characteristics/control_period_analysis.png` - **NEW**: Control period analysis
- `docs/O2_REALTIME_DATA.md` - **NEW**: Complete data documentation with feature engineering guidance
- `docs/ADVANCED_FEATURE_ENGINEERING_PLAN.md` - **NEW**: Detailed implementation plan for pharmacological features
- `data/raw/event_data.parquet` - **NEW**: Real experimental events from database
- `data/raw/plate_event_summary.parquet` - **NEW**: Event summary by plate

### Dataset Summary
- **240 qualifying drugs** (≥4 concentrations, ≥14 days data)
- **7,616 well-level time series** (4 replicates per concentration)
- **3.15M oxygen measurements** across 10,973 wells and 33 plates
- **652 experimental events**: 35 dosing events, 103 media changes across 31 plates
- **Event timing**: Media changes occur 95.4 ± 130.0 hours after dosing
- **Control periods**: 24-48h pre-dosing baseline available for all wells

### Event Data Status
- ✅ **Downloaded**: Real event data from Supabase `event_table` 
- ✅ **Integrated**: Event timing with oxygen measurements
- ⚠️ **Partial**: Only sample integration complete (3 plates)
- 🔄 **TODO**: Full dataset integration pending (computationally intensive)

### Critical Technical Notes
- **Use uv for all package management** (user preference)
- **numpy<2.0 required** for TSFresh compatibility
- **Event data pipeline**: Downloaded via Supabase MCP, integrated with DuckDB
- **Timezone handling**: All timestamps converted to timezone-naive for comparison
- **Memory optimization**: Full dataset integration requires chunked processing
- **Real experimental structure**: Dosing events, media changes, control periods now available
- **Feature engineering foundation**: Event-aware temporal windows and dose-response normalization ready

## Technology Stack
- **Time Series**: tslearn, tsfresh, catch22, pyts, stumpy
- **Deep Learning**: PyTorch, TensorFlow, tsai (Time Series AI)
- **Data Processing**: pandas, numpy, scipy, scikit-learn
- **Database**: DuckDB with PostgreSQL backend to Supabase
- **Visualization**: matplotlib, seaborn, plotly, umap-learn