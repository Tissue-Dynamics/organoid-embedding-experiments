# Organoid DILI Prediction

Machine learning analysis of liver organoid oxygen consumption patterns to predict drug-induced liver injury (DILI) risk.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set database credentials
export DATABASE_URL="postgresql://..."

# Run analysis
python run_analysis.py
```

## Key Results

- Event-aware features achieve **r=0.435** correlation with DILI risk
- 67% improvement over traditional embedding methods
- Identifies consumption ratio and temporal progression as key predictors

## Structure

- `src/` - Core analysis modules
- `results/` - Generated outputs and figures
- `run_analysis.py` - Main analysis runner

See `CLAUDE.md` for detailed development information.
