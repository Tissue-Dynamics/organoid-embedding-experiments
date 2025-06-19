#!/usr/bin/env python3
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
        
        print("\n✅ Analysis complete!")
        print("   Check results/ directory for outputs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_all()
