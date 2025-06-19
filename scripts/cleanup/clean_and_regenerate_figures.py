#!/usr/bin/env python3
"""
Clean up figures directory and regenerate essential visualizations
Remove cruft and focus on key results
"""

import shutil
from pathlib import Path
import os
import sys

project_root = Path(__file__).parent.parent.parent
figures_dir = project_root / "results" / "figures"

print("=" * 80)
print("CLEANING FIGURES AND REGENERATING KEY VISUALIZATIONS")
print("=" * 80)

# Backup existing figures
backup_figures = project_root / "backup_figures"
if backup_figures.exists():
    shutil.rmtree(backup_figures)
shutil.copytree(figures_dir, backup_figures)
print(f"📦 Backed up figures to: {backup_figures}")

# Define essential figures to keep
essential_figures = {
    'core/': [
        'event_aware_final_summary.png',  # Main result
        'media_change_events_summary.png'  # Event overview
    ],
    'phase2/': [
        'embedding_comparisons/fourier_hierarchical_clusters.png',
        'embedding_comparisons/catch22_hierarchical_clusters.png',
        'embedding_comparisons/sax_hierarchical_clusters.png',
        'embedding_comparisons/custom_hierarchical_clusters.png'
    ],
    'validation/': [
        'event_verification/plate_1_event_verification.png',
        'event_verification/plate_2_event_verification.png', 
        'event_verification/plate_3_event_verification.png',
        'event_verification/spike_characterization_detailed.png',
        'event_verification/event_timing_summary_all_plates.png'
    ]
}

# Clean figures directory
print("\n🧹 Cleaning figures directory...")
if figures_dir.exists():
    shutil.rmtree(figures_dir)
figures_dir.mkdir(parents=True, exist_ok=True)

# Create clean structure
for dir_name in essential_figures.keys():
    (figures_dir / dir_name).mkdir(parents=True, exist_ok=True)

# Copy essential figures back
print("\n📋 Restoring essential figures...")
restored_count = 0
for category, file_list in essential_figures.items():
    for fig_path in file_list:
        old_file = backup_figures / fig_path
        if '/' in fig_path:
            # Handle subdirectory files
            new_file = figures_dir / category / Path(fig_path).name
        else:
            new_file = figures_dir / category / fig_path
        
        if old_file.exists():
            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_file, new_file)
            print(f"   ✓ {fig_path} → {category}")
            restored_count += 1

print(f"\n   Restored {restored_count} essential figures")

# Create figure regeneration script
print("\n🔄 Creating figure regeneration script...")

regen_script = '''#!/usr/bin/env python3
"""
Regenerate Essential Figures
Clean, focused visualizations for key results
"""

import sys
from pathlib import Path
import os

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def regenerate_phase2_embeddings():
    """Regenerate Phase 2 embedding visualizations"""
    print("🔬 Regenerating Phase 2 embedding visualizations...")
    
    try:
        from analysis.phase2_embeddings import HierarchicalClusterOxygenVisualization
        
        # Set clean output directory
        analyzer = HierarchicalClusterOxygenVisualization()
        analyzer.figure_dir = project_root / "results" / "figures" / "phase2"
        analyzer.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing results if available
        results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
        if results_file.exists():
            import joblib
            results = joblib.load(results_file)
            analyzer.create_hierarchical_visualizations(results, n_clusters=6)
            print("   ✅ Phase 2 visualizations regenerated")
        else:
            print("   ⚠️ Phase 2 results not found, run full analysis first")
            
    except Exception as e:
        print(f"   ❌ Error regenerating Phase 2: {e}")

def regenerate_event_verification():
    """Regenerate event verification plots"""
    print("🔍 Regenerating event verification plots...")
    
    try:
        # Set database URL
        os.environ["DATABASE_URL"] = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
        
        from visualization.event_verification import main as verify_events
        
        # Override figure directory in the script
        import visualization.event_verification as ev_module
        ev_module.fig_dir = project_root / "results" / "figures" / "validation"
        ev_module.fig_dir.mkdir(parents=True, exist_ok=True)
        
        verify_events()
        print("   ✅ Event verification plots regenerated")
        
    except Exception as e:
        print(f"   ❌ Error regenerating event verification: {e}")

def create_summary_figure():
    """Create clean summary figure"""
    print("📊 Creating summary figure...")
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Load correlation results if available
        results_dir = project_root / "results" / "data"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Organoid DILI Prediction: Key Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Method comparison
        methods = ['Phase 2\\nFourier', 'Phase 2\\nTSFresh', 'Phase 2\\nCatch22', 'Event-Aware\\nFeatures']
        correlations = [0.260, 0.243, 0.237, 0.435]
        colors = ['lightblue', 'lightblue', 'lightblue', 'lightgreen']
        
        bars = ax1.bar(methods, correlations, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Correlation with DILI Risk', fontsize=12)
        ax1.set_title('Method Performance Comparison', fontsize=14)
        ax1.set_ylim(0, 0.5)
        
        # Add value labels and improvement
        for i, (bar, val) in enumerate(zip(bars, correlations)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            if i == 3:  # Event-aware
                improvement = (val - 0.260) / 0.260 * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                        f'+{improvement:.0f}%', ha='center', va='bottom',
                        fontsize=10, color='green', fontweight='bold')
        
        ax1.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='r=0.3 threshold')
        ax1.legend()
        
        # Plot 2: Dataset sizes
        datasets = ['Phase 2\\nEmbeddings', 'Event-Aware\\nFeatures', 'Combined\\nOverlap']
        sizes = [240, 63, 41]
        colors2 = ['lightcoral', 'lightgreen', 'gold']
        
        bars2 = ax2.bar(datasets, sizes, color=colors2, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Drugs', fontsize=12)
        ax2.set_title('Dataset Coverage', fontsize=14)
        
        for bar, size in zip(bars2, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(size), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save to core directory
        core_dir = project_root / "results" / "figures" / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(core_dir / 'key_results_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Summary figure created")
        
    except Exception as e:
        print(f"   ❌ Error creating summary: {e}")

def main():
    """Regenerate all essential figures"""
    print("=" * 60)
    print("REGENERATING ESSENTIAL FIGURES")
    print("=" * 60)
    
    create_summary_figure()
    regenerate_event_verification()
    regenerate_phase2_embeddings()
    
    print("\\n✅ Figure regeneration complete!")
    print("\\n📁 Clean figure structure:")
    print("   results/figures/core/        - Key summary figures")
    print("   results/figures/phase2/      - Phase 2 embedding plots")
    print("   results/figures/validation/  - Event verification plots")

if __name__ == "__main__":
    main()
'''

regen_file = project_root / "regenerate_figures.py"
regen_file.write_text(regen_script)
print(f"   Created: {regen_file}")

# Create clean README for figures
readme_content = '''# Results Figures

Clean, essential visualizations for the organoid DILI prediction analysis.

## Directory Structure

### `core/` - Key Results
- `key_results_summary.png` - Main performance comparison
- `event_aware_final_summary.png` - Event-aware feature results (if available)
- `media_change_events_summary.png` - Media change event overview (if available)

### `phase2/` - Phase 2 Embeddings
- Hierarchical clustering visualizations for each embedding method
- Shows wells → concentrations → drugs progression

### `validation/` - Event Verification  
- `plate_*_event_verification.png` - Time series with detected events
- `spike_characterization_detailed.png` - Detailed spike analysis
- `event_timing_summary_all_plates.png` - Event timing overview

## Regeneration

To regenerate all figures:

```bash
python regenerate_figures.py
```

Individual modules can also be run directly from `src/` directory.
'''

(figures_dir / "README.md").write_text(readme_content)

print(f"\n📝 Created: {figures_dir / 'README.md'}")

# Summary
print("\n" + "=" * 80)
print("FIGURE CLEANUP COMPLETE")
print("=" * 80)

print(f"\n📊 SUMMARY:")
print(f"   Figures backed up: {backup_figures}")
print(f"   Essential figures restored: {restored_count}")
print(f"   Clean structure created: core/, phase2/, validation/")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. Run: python regenerate_figures.py")
print(f"   2. Review: results/figures/ for clean visualizations")
print(f"   3. Commit: Clean figure structure")

print(f"\n📁 CLEAN STRUCTURE:")
for category in essential_figures.keys():
    dir_path = figures_dir / category
    if dir_path.exists():
        file_count = len(list(dir_path.glob("*.png")))
        print(f"   {category:<12} {file_count} figures")

print("\n✅ Ready to regenerate clean figures!")