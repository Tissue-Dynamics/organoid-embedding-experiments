# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for comparing embedding methods on liver organoid oxygen time series data. The project evaluates traditional methods (DTW, Fourier, SAX), feature-based approaches (TSFresh, catch22), and deep learning methods (autoencoders, transformers, triplet networks) on ~7,680 time series from 240 drug treatments.

## Development Commands

```bash
# Install dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Supabase credentials

# Run drug embedding analysis
python scripts/drug_embeddings/fast_drug_analysis.py  # Quick analysis with summary features
python scripts/drug_embeddings/demo_drug_embedding.py  # Single drug demo

# Run embedding experiments
python experiments/run_experiment.py  # Uses default config
python experiments/run_experiment.py --config-name quick_test  # Quick test
python experiments/run_experiment.py --config-name deep_learning_only  # DL only
```

## Project Structure

```
├── data/                      # Core data processing modules
├── embeddings/               # All embedding implementations
├── evaluation/              # Metrics and visualization tools
├── experiments/            # Experiment configs and runners
├── scripts/               # Analysis and utility scripts
│   ├── analysis/         # Data analysis scripts
│   ├── database/         # Database utilities
│   ├── drug_embeddings/  # Drug embedding pipeline
│   └── data_exploration/ # Exploratory scripts
├── results/              # Generated outputs
│   ├── figures/         # Visualizations
│   └── data/           # Processed data
└── docs/               # Documentation
```

## Architecture and Key Components

### Data Layer
- **`data/loaders/supabase_loader.py`** - Central data access through `SupabaseDataLoader` class
  - Handles all database operations with batch processing and parallel loading
  - Key methods: `get_time_series_data()`, `prepare_time_series_matrix()`, `load_experiment_batch()`
  - Returns pandas DataFrames and numpy arrays ready for ML workflows

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

### Experiment Management
- **`experiments/run_experiment.py`** - Main experiment runner with Hydra configuration
- **`experiments/configs/`** - YAML configuration files for different experiment types
  - `config.yaml` - Full comparison of all methods
  - `quick_test.yaml` - Fast subset for testing
  - `deep_learning_only.yaml` - Focus on neural approaches

### Evaluation and Visualization
- **`evaluation/metrics/`** - Comprehensive embedding quality assessment
  - Clustering metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Neighborhood preservation (trustworthiness, continuity)
  - Dimensionality analysis and stability measures
- **`evaluation/visualization/`** - Rich plotting utilities for embeddings and comparisons

### Configuration Management
- Uses `hydra-core` for experiment configuration management
- Environment variables for Supabase credentials via `python-dotenv`
- Flexible YAML-based configs supporting method-specific parameters

### Current State
This is a **fully implemented research codebase** with comprehensive embedding methods, preprocessing utilities, experiment management, and evaluation tools. All major components are functional and ready for organoid time series analysis.

## Technology Stack
- **Time Series**: tslearn, tsfresh, catch22, pyts, stumpy
- **Deep Learning**: PyTorch, TensorFlow, tsai (Time Series AI)
- **Data Processing**: pandas, numpy, scipy, scikit-learn
- **Database**: Supabase client
- **Visualization**: matplotlib, seaborn, plotly, umap-learn