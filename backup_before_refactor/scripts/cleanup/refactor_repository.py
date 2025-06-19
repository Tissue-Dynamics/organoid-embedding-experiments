#!/usr/bin/env python3
"""
Refactor Repository - Clean up cruft and organize properly
"""

import shutil
from pathlib import Path
import os

project_root = Path(__file__).parent.parent.parent

print("=" * 80)
print("REPOSITORY REFACTORING AND CLEANUP")
print("=" * 80)

# Define clean structure
new_structure = {
    'src/': {
        'data/': ['loaders/', 'preprocessing/', 'features/'],
        'embeddings/': ['traditional/', 'deep_learning/', 'features/'],
        'analysis/': ['dili/', 'correlations/', 'prediction/'],
        'visualization/': ['plots/', 'figures/', 'validation/']
    },
    'scripts/': {
        'experiments/': [],
        'utilities/': []
    },
    'results/': {
        'data/': ['processed/', 'features/', 'models/'],
        'figures/': ['analysis/', 'validation/', 'final/']
    },
    'docs/': [],
    'tests/': []
}

# Files to keep (core analysis)
keep_files = {
    'scripts/analysis/': [
        'hierarchical_cluster_oxygen_visualization.py',  # Main Phase 2 analysis
        'explore_drugs_table.py',  # Drug metadata exploration
    ],
    'scripts/preprocessing/': [
        'extract_event_aware_features.py',  # Event-aware features
        'correlate_event_features_dili.py',  # DILI correlation
    ],
    'scripts/visualization/': [
        'verify_media_change_events.py',  # Event verification
    ],
    'results/data/': [
        'hierarchical_embedding_results.joblib',  # Phase 2 results
        'event_aware_features_drugs.parquet',  # Event features
        'media_change_events.parquet',  # Media events
        'wells_drugs_integrated.parquet',  # Core dataset
        'event_aware_dili_correlations.csv',  # Key results
    ],
    'results/figures/': [
        'embedding_comparisons/',  # Phase 2 visualizations
        'event_verification/',  # Event validation
        'event_aware_final_summary.png',  # Key result
    ]
}

# Create backup
print("\n📦 Creating backup...")
backup_dir = project_root / "backup_before_refactor"
if backup_dir.exists():
    shutil.rmtree(backup_dir)
shutil.copytree(project_root, backup_dir, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.venv'))
print(f"   Backup created: {backup_dir}")

# Files/dirs to remove (cruft)
remove_patterns = [
    'scripts/analysis/detect_*',
    'scripts/analysis/quick_*',
    'scripts/analysis/calibrate_*',
    'scripts/analysis/find_*',
    'scripts/visualization/step*',
    'scripts/visualization/strict_*',
    'scripts/visualization/plate_*',
    'scripts/visualization/comprehensive_*',
    'scripts/visualization/control_wells_analysis.py',
    'scripts/preprocessing/check_*',
    'scripts/preprocessing/combine_*',
    'scripts/preprocessing/extract_media_change_events.py',
    'scripts/preprocessing/extract_media_changes_simple.py',
    'scripts/preprocessing/generate_final_summary.py',
    'scripts/connect_wells_to_drugs.py',
    'scripts/debug_*',
    'scripts/drug_*',
    'scripts/event_*',
    'scripts/investigate_*',
    'scripts/phase2_*',
    'results/figures/step3_validation/',
    'results/figures/combined_analysis/',
    'results/figures/drug_analysis/',
    'results/figures/drug_embeddings_dili/',
    'results/figures/enhanced_prediction/',
    'results/figures/event_aware_dili/',
    'results/figures/event_aware_features/',
    'results/figures/simple_dili_analysis/',
    'results/data/*spike*',
    'results/data/*step*',
    'results/data/combined_*',
    'results/data/drug_dili_*',
    'results/data/drug_oxygen_*',
    'results/data/phase2_*',
    'results/data/*dili_correlations.csv',
    'results/data/untagged_*',
    'results/data/wells_drugs_integrated.csv',
    'results/data/dili_predictions_*',
    'results/data/best_dili_*',
]

print("\n🧹 Removing cruft...")
removed_count = 0

for pattern in remove_patterns:
    # Handle glob patterns
    if '*' in pattern:
        from glob import glob
        matches = glob(str(project_root / pattern))
        for match in matches:
            path = Path(match)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_count += 1
                print(f"   Removed: {path.relative_to(project_root)}")
    else:
        path = project_root / pattern
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed_count += 1
            print(f"   Removed: {path.relative_to(project_root)}")

print(f"\n   Total items removed: {removed_count}")

# Create clean structure
print("\n🏗️  Creating clean structure...")

# Move core files to new locations
moves = [
    ('scripts/analysis/hierarchical_cluster_oxygen_visualization.py', 'src/analysis/phase2_embeddings.py'),
    ('scripts/preprocessing/extract_event_aware_features.py', 'src/features/event_aware_extraction.py'),
    ('scripts/preprocessing/correlate_event_features_dili.py', 'src/analysis/dili_correlation.py'),
    ('scripts/visualization/verify_media_change_events.py', 'src/visualization/event_verification.py'),
]

for old_path, new_path in moves:
    old_file = project_root / old_path
    new_file = project_root / new_path
    
    if old_file.exists():
        new_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_file), str(new_file))
        print(f"   Moved: {old_path} → {new_path}")

# Create clean directories
clean_dirs = [
    'src/data',
    'src/embeddings', 
    'src/analysis',
    'src/visualization',
    'src/utils',
    'notebooks',
    'tests',
    'docs',
    'config'
]

for dir_path in clean_dirs:
    (project_root / dir_path).mkdir(parents=True, exist_ok=True)

# Create main module files
print("\n📝 Creating main module files...")

# src/__init__.py
(project_root / "src/__init__.py").write_text('"""Organoid DILI Prediction Library"""')

# Main analysis runner
main_runner = '''#!/usr/bin/env python3
"""
Main Analysis Runner
Reproduce key results from the organoid DILI prediction study
"""

from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def run_phase2_embeddings():
    """Run Phase 2 hierarchical embeddings analysis"""
    from analysis.phase2_embeddings import main
    print("🔬 Running Phase 2 embeddings analysis...")
    main()

def run_event_aware_features():
    """Extract event-aware features"""
    from features.event_aware_extraction import main
    print("⚡ Extracting event-aware features...")
    main()

def run_dili_correlation():
    """Correlate features with DILI risk"""
    from analysis.dili_correlation import main
    print("🎯 Running DILI correlation analysis...")
    main()

def run_all():
    """Run complete analysis pipeline"""
    print("=" * 60)
    print("ORGANOID DILI PREDICTION ANALYSIS")
    print("=" * 60)
    
    try:
        run_phase2_embeddings()
        run_event_aware_features() 
        run_dili_correlation()
        
        print("\\n✅ Analysis complete!")
        print("   Check results/ directory for outputs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_all()
'''

(project_root / "run_analysis.py").write_text(main_runner)

# Create requirements.txt
requirements = '''# Core dependencies
pandas>=1.5.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Time series analysis
tslearn>=0.5.0
tsfresh>=0.19.0
catch22>=0.4.0

# Database
duckdb>=0.8.0
psycopg2-binary>=2.9.0

# Deep learning
torch>=1.12.0

# Visualization
plotly>=5.0.0
umap-learn>=0.5.0

# Utilities
joblib>=1.1.0
tqdm>=4.64.0
pathlib2>=2.3.0
'''

(project_root / "requirements.txt").write_text(requirements)

# Update CLAUDE.md with clean structure
claude_md = '''# CLAUDE.md

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
'''

(project_root / "CLAUDE.md").write_text(claude_md)

# Create simple README
readme = '''# Organoid DILI Prediction

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
'''

(project_root / "README.md").write_text(readme)

# Clean up empty directories
print("\n🧹 Cleaning up empty directories...")
for root, dirs, files in os.walk(project_root, topdown=False):
    for dirname in dirs:
        dir_path = Path(root) / dirname
        try:
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                print(f"   Removed empty: {dir_path.relative_to(project_root)}")
        except:
            pass

# Summary
print("\n" + "=" * 80)
print("REFACTORING COMPLETE")
print("=" * 80)

print(f"\n📊 SUMMARY:")
print(f"   Items removed: {removed_count}")
print(f"   Backup created: {backup_dir}")
print(f"   Clean structure implemented")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. Test: python run_analysis.py")
print(f"   2. Review: Check results/ for key outputs")
print(f"   3. Commit: git add -A && git commit -m 'Refactor and clean repository'")

print(f"\n📁 KEY FILES PRESERVED:")
print(f"   • src/analysis/phase2_embeddings.py")
print(f"   • src/features/event_aware_extraction.py") 
print(f"   • src/analysis/dili_correlation.py")
print(f"   • results/data/hierarchical_embedding_results.joblib")
print(f"   • results/figures/embedding_comparisons/")

print("\n✅ Repository is now clean and organized!")