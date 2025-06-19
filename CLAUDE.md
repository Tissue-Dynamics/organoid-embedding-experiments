# CLAUDE.md

## Project Overview

Clean, refactored organoid DILI prediction codebase. This project analyzes liver organoid oxygen consumption patterns to predict drug-induced liver injury (DILI) risk.

## Key Results

- **Phase 2 Hierarchical Embeddings**: r=0.260 correlation with DILI
- **Event-Aware Features**: r=0.435 correlation with DILI (67% improvement)
- **Best Predictors**: Consumption ratio and temporal progression features

## Project Structure

```
├── src/                    # Core analysis modules
│   ├── analysis/          # DILI correlation and prediction
│   ├── features/          # Feature extraction (event-aware, embeddings)
│   ├── visualization/     # Plotting and validation
│   └── utils/            # Utilities
├── results/              # Generated outputs
│   ├── data/            # Processed datasets and models
│   └── figures/         # Key visualizations
├── config/              # Configuration files
├── tests/               # Unit tests
└── run_analysis.py      # Main analysis runner

```

## Usage

```bash
# Install dependencies
uv pip install -r requirements.txt

# Set database credentials
export DATABASE_URL="postgresql://..."

# Run complete analysis
python run_analysis.py
```

## Key Files

- `src/analysis/phase2_embeddings.py` - Hierarchical drug embeddings
- `src/features/event_aware_extraction.py` - Event-aware feature extraction  
- `src/analysis/dili_correlation.py` - DILI correlation analysis
- `results/data/hierarchical_embedding_results.joblib` - Phase 2 results
- `results/figures/event_aware_final_summary.png` - Key findings

## Development Commands

```bash
# Use uv for package management (user preference)
uv pip install -r requirements.txt

# Database connection
export DATABASE_URL="postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
```
