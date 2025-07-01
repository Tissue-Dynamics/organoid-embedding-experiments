# CLAUDE.md

## Project Overview

Clean, refactored organoid DILI prediction codebase. This project analyzes liver organoid oxygen consumption patterns to predict drug-induced liver injury (DILI) risk.

## Key Results (Latest Analysis)

- **Dynamic Variability Features**: r=0.473 correlation with DILI likelihood (p=0.003)
- **Event-Normalized Features**: r=0.477 on 34 drugs, AUROC=0.832
- **Best Predictors**: Rolling variance features (`rolling_50_mean_std_std`)
- **Key Insight**: Drug-induced liver injury manifests as increased variability in oxygen consumption patterns

## Lessons Learned

See `docs/COMPREHENSIVE_DILI_ANALYSIS_LESSONS_LEARNED.md` for detailed documentation of:
- Feature engineering discoveries (dynamic variability > central tendency)
- Technical best practices (modular design, robust error handling)
- Experimental design insights (plate effects, randomization issues)
- Future recommendations (expand rolling variance analysis)

## Project Structure

```
├── src/                    # Reusable modules and libraries
│   └── utils/             # Utility functions (data loading, etc.)
├── scripts/                # Executable scripts (run these!)
│   ├── analysis/          # DILI correlation and prediction scripts
│   ├── features/          # Feature extraction scripts
│   ├── visualization/     # Plotting and validation scripts
│   ├── database/          # Database interaction scripts
│   └── experiments/       # Experimental analysis scripts
├── results/              # Generated outputs
│   ├── data/            # Processed datasets and models
│   └── figures/         # Visualizations
├── archived/             # Previous analysis iterations
├── docs/                # Documentation and guides
├── embeddings/           # Embedding method implementations
├── config/              # Configuration files
└── tests/               # Unit tests

```

**Key distinction**:
- `src/` = Import from here (reusable modules, utilities, classes)
- `scripts/` = Run these (executable analysis scripts with main blocks)

## Database Access

The project supports both remote PostgreSQL and local DuckDB databases:

```bash
# Export database to local DuckDB file (recommended for faster analysis)
python scripts/database/export_db_with_progress.py

# Use local database (auto-detected if organoid_data.duckdb exists)
with DataLoader() as loader:
    data = loader.load_oxygen_data()

# Force remote database
with DataLoader(use_local=False) as loader:
    data = loader.load_drug_metadata()
```

### Exported Tables
- **Core Tables**: drugs, event_table, well_map_data, plate_table, processed_data
- **Imaging Data**: well_image_data (organoid counts and measurements)
- **Gene Expression**: gene_samples, gene_biomarkers, gene_drug_keys, gene_expression

## Usage

```bash
# Install dependencies
uv pip install -r requirements.txt

# Set database credentials (for remote access or initial export)
export DATABASE_URL="postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"

# Export database to local file (one-time setup, takes ~10-15 minutes)
python scripts/database/export_db_with_progress.py

# Run feature extraction
python scripts/features/event_aware_extraction.py
python scripts/features/multi_timescale_catch22.py
python scripts/features/dose_response_normalization.py

# Run analysis
python scripts/analysis/dili_correlation.py
python scripts/analysis/enhanced_dili_prediction.py

# Create visualizations
python scripts/visualization/event_verification.py
```

## Development Commands

```bash
# Use uv for package management (user preference)
uv pip install -r requirements.txt

# Database connection (set in environment or .env file)
export DATABASE_URL="postgresql://..."

# Run tests
python -m pytest tests/

# Check code style
ruff check .
```

## Important Guidelines

**ALWAYS USE UV FOR PACKAGE MANAGEMENT** - This is critical:
- **Installing packages**: `uv pip install package-name` (NOT `pip install`)
- **Installing requirements**: `uv pip install -r requirements.txt`
- **Running Python scripts**: Just use `python script.py` (uv manages the environment)
- **Checking installed packages**: `uv pip list`
- **Why uv**: It's much faster than pip and handles dependencies better
- The project uses a `.venv` managed by uv - never use pip directly

**NEVER CREATE OVERVIEW/SUMMARY FIGURES** - They always suck and are not useful. Focus on:
- Actual data visualizations (time series, scatter plots, heatmaps)
- Method-specific results (embedding plots, correlation matrices)
- Validation plots (event detection, model performance)
- Real scientific figures that show data, not promotional graphics

**DO NOT MENTION "IMPROVEMENT OVER BASELINE"** - We are still in exploration phase:
- Avoid comparing to previous results as "improvements"
- Don't claim X% better performance
- Focus on absolute metrics and findings
- This is research exploration, not optimization

**ALWAYS GENERATE VERIFICATION FIGURES IN THE SAME SCRIPT** - Every script must produce its own figures:
- Never create separate visualization scripts
- Include all visualizations directly in the feature extraction/analysis script
- This ensures immediate verification of results
- If a script doesn't produce figures, it cannot be verified and should be deleted

**NEVER PUT FILES IN THE ROOT DIRECTORY** - This is absolutely critical:
- The root directory is for navigation and organization ONLY
- NO Python scripts should ever be placed in the root directory
- ALL scripts must go in their proper subdirectories:
  - `scripts/analysis/` for analysis scripts
  - `scripts/features/` for feature extraction
  - `scripts/visualization/` for visualization
  - `Sandbox/` for temporary/experimental work
- If you create a file in root by mistake, DELETE IT IMMEDIATELY
- Root should only contain: README.md, CLAUDE.md, requirements.txt, pyproject.toml, and directories

**File Management Guidelines**:
- Always delete temporary files or throw-away scripts after you are done using them
- Keep the project structure clean and organized

## CRITICAL DATA INTERPRETATION NOTE

**OXYGEN DATA INTERPRETATION** - This is absolutely critical for all analysis:
- The 'o2' column represents **OXYGEN PRESENCE/CONCENTRATION** in the medium
- **Lower O2 values = Higher oxygen consumption** (cells are consuming more oxygen)
- **Higher O2 values = Lower oxygen consumption** (cells are consuming less oxygen)
- **DO NOT confuse this**: It's measuring remaining oxygen, not consumption rate directly
- **Toxicity interpretation**: Toxic drugs may DECREASE oxygen consumption (higher O2 remaining)
- **Metabolic activation**: Some drugs may INCREASE oxygen consumption (lower O2 remaining)
- **Always verify the directionality** when interpreting oxygen consumption changes