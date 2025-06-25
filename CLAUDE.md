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

## Usage

```bash
# Install dependencies
uv pip install -r requirements.txt

# Set database credentials
export DATABASE_URL="postgresql://..."

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

# Database connection
export DATABASE_URL="postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
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
