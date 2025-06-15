#!/usr/bin/env python3
"""Map out drug-based embedding strategy for organoid experiments."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_plate_drug_structure():
    """Explore how drugs and concentrations are organized in the data."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    
    # Connect to database
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    from urllib.parse import urlparse
    parsed = urlparse(database_url)
    
    attach_query = f"""
    ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
    user={parsed.username} password={parsed.password}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    
    conn.execute(attach_query)
    
    logger.info("🔍 Exploring plate and drug structure...")
    
    # First, check if there's a well_map or plate_layout table
    tables_query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    AND (table_name LIKE '%well%' OR table_name LIKE '%map%' OR table_name LIKE '%layout%'
         OR table_name LIKE '%drug%' OR table_name LIKE '%treatment%' OR table_name LIKE '%compound%')
    """
    
    tables = conn.execute(tables_query).df()
    logger.info(f"Found relevant tables: {tables['table_name'].tolist()}")
    
    # Check drugs table structure (we know this exists)
    drugs_schema = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'drugs'
    ORDER BY ordinal_position
    LIMIT 20
    """
    
    drugs_cols = conn.execute(drugs_schema).df()
    logger.info("\nDrugs table columns:")
    for _, row in drugs_cols.iterrows():
        logger.info(f"  {row['column_name']}: {row['data_type']}")
    
    # Get sample drug entries
    drugs_sample = conn.execute("SELECT id, drug, cmax_dose FROM supabase.public.drugs LIMIT 5").df()
    logger.info(f"\nSample drugs:\n{drugs_sample}")
    
    # Check for plate metadata
    plate_info_query = """
    SELECT DISTINCT plate_id, COUNT(DISTINCT well_number) as n_wells
    FROM supabase.public.processed_data
    GROUP BY plate_id
    ORDER BY n_wells DESC
    LIMIT 5
    """
    
    plate_info = conn.execute(plate_info_query).df()
    logger.info(f"\nPlate well counts:\n{plate_info}")
    
    # Check well_map_data structure - THIS IS KEY!
    logger.info("\n🎯 Exploring well_map_data table (drug-concentration mapping)...")
    
    well_map_schema = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'well_map_data'
    ORDER BY ordinal_position
    """
    
    well_map_cols = conn.execute(well_map_schema).df()
    logger.info("\nwell_map_data columns:")
    for _, row in well_map_cols.iterrows():
        logger.info(f"  {row['column_name']}: {row['data_type']}")
    
    # Get sample well mappings
    well_map_sample = """
    SELECT *
    FROM supabase.public.well_map_data
    LIMIT 10
    """
    
    well_map_df = conn.execute(well_map_sample).df()
    logger.info(f"\nSample well mappings:\n{well_map_df}")
    
    # Analyze drug distribution in a plate
    drug_distribution_query = """
    SELECT 
        plate_id,
        COUNT(DISTINCT drug_name) as n_drugs,
        COUNT(DISTINCT concentration_um) as n_concentrations,
        COUNT(*) as n_wells
    FROM supabase.public.well_map_data
    GROUP BY plate_id
    LIMIT 5
    """
    
    try:
        drug_dist = conn.execute(drug_distribution_query).df()
        logger.info(f"\nDrug distribution by plate:\n{drug_dist}")
    except Exception as e:
        logger.error(f"Could not get drug distribution: {e}")
    
    conn.close()
    
    return plate_info

def design_drug_embedding_strategy():
    """Design the strategy for drug-based embeddings."""
    
    logger.info("\n🎯 DRUG EMBEDDING STRATEGY DESIGN")
    
    strategy = {
        "overview": """
        Goal: Create a comprehensive embedding for each drug that captures its 
        effects across all concentrations, time points, and replicates.
        """,
        
        "data_hierarchy": {
            "level_1": "Drug (e.g., Paracetamol, Rifampicin)",
            "level_2": "Concentration (e.g., 10μM, 100μM, 1mM)",
            "level_3": "Replicate (e.g., 4 replicates per concentration)",
            "level_4": "Time series (e.g., 2800 time points per well)"
        },
        
        "embedding_levels": {
            "well_level": {
                "description": "Individual well time series → embedding",
                "methods": ["Fourier", "SAX", "Custom features", "Autoencoder"],
                "output": "One embedding vector per well"
            },
            "concentration_level": {
                "description": "Aggregate embeddings across replicates",
                "methods": ["Mean pooling", "Attention pooling", "Distribution features"],
                "output": "One embedding per concentration"
            },
            "drug_level": {
                "description": "Combine all concentrations into drug signature",
                "methods": ["Dose-response modeling", "Multi-scale features", "Hierarchical attention"],
                "output": "One comprehensive embedding per drug"
            }
        },
        
        "key_features": {
            "dose_response": [
                "EC50/IC50 estimation",
                "Hill coefficient",
                "Maximum effect (Emax)",
                "Area under dose-response curve"
            ],
            "temporal_dynamics": [
                "Early response (0-24h)",
                "Sustained effect (24-168h)",
                "Recovery pattern",
                "Oscillation characteristics"
            ],
            "variability_metrics": [
                "Inter-replicate consistency",
                "Concentration-dependent variance",
                "Temporal stability"
            ]
        },
        
        "aggregation_approaches": {
            "approach_1": {
                "name": "Hierarchical Mean Pooling",
                "description": "Average embeddings at each level",
                "pros": "Simple, robust to outliers",
                "cons": "Loses distributional information"
            },
            "approach_2": {
                "name": "Attention-Based Aggregation",
                "description": "Learn importance weights for each concentration",
                "pros": "Adaptive, can emphasize critical concentrations",
                "cons": "Requires training"
            },
            "approach_3": {
                "name": "Multi-Resolution Features",
                "description": "Extract features at multiple scales and concatenate",
                "pros": "Preserves information at all levels",
                "cons": "High dimensional"
            },
            "approach_4": {
                "name": "Dose-Response Parameterization",
                "description": "Fit dose-response curves and use parameters as features",
                "pros": "Biologically interpretable",
                "cons": "Assumes specific functional form"
            }
        }
    }
    
    # Print strategy
    logger.info("\n📋 Data Hierarchy:")
    for level, desc in strategy["data_hierarchy"].items():
        logger.info(f"  {level}: {desc}")
    
    logger.info("\n📊 Embedding Levels:")
    for level, details in strategy["embedding_levels"].items():
        logger.info(f"\n  {level.upper()}:")
        logger.info(f"    {details['description']}")
        logger.info(f"    Methods: {', '.join(details['methods'])}")
        logger.info(f"    Output: {details['output']}")
    
    logger.info("\n🔧 Aggregation Approaches:")
    for approach_id, approach in strategy["aggregation_approaches"].items():
        logger.info(f"\n  {approach['name']}:")
        logger.info(f"    {approach['description']}")
        logger.info(f"    ✅ {approach['pros']}")
        logger.info(f"    ⚠️  {approach['cons']}")
    
    return strategy

def create_visual_strategy_diagram():
    """Create a visual diagram of the embedding strategy."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Data hierarchy
    ax1.text(0.5, 0.95, 'Data Hierarchy', ha='center', va='top', fontsize=16, weight='bold')
    
    # Drug level
    ax1.add_patch(plt.Rectangle((0.1, 0.8), 0.8, 0.1, fill=True, color='lightcoral', alpha=0.7))
    ax1.text(0.5, 0.85, 'Drug Level\n(e.g., Paracetamol)', ha='center', va='center', fontsize=12)
    
    # Concentration level
    for i, conc in enumerate(['10μM', '100μM', '1mM']):
        x = 0.15 + i * 0.25
        ax1.add_patch(plt.Rectangle((x, 0.6), 0.2, 0.08, fill=True, color='lightskyblue', alpha=0.7))
        ax1.text(x + 0.1, 0.64, conc, ha='center', va='center', fontsize=10)
        ax1.arrow(0.5, 0.8, x + 0.1 - 0.5, -0.12, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Replicate level
    for i in range(3):
        for j in range(4):
            x = 0.15 + i * 0.25 + j * 0.04
            ax1.add_patch(plt.Rectangle((x, 0.4), 0.03, 0.06, fill=True, color='lightgreen', alpha=0.7))
        ax1.text(0.15 + i * 0.25 + 0.08, 0.35, 'Replicates', ha='center', va='top', fontsize=8)
    
    # Time series level
    ax1.text(0.5, 0.25, 'Time Series Data\n(~2800 time points per well)', ha='center', va='center', fontsize=10)
    
    # Sample time series
    t = np.linspace(0, 1, 100)
    for i in range(3):
        x_offset = 0.15 + i * 0.25
        y_data = 0.15 + 0.05 * np.sin(10 * t + i) * np.exp(-2 * t)
        ax1.plot(x_offset + t * 0.2, y_data, 'b-', alpha=0.5, linewidth=1)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right: Embedding process
    ax2.text(0.5, 0.95, 'Embedding Process', ha='center', va='top', fontsize=16, weight='bold')
    
    # Process flow
    steps = [
        (0.85, 'Raw Time Series'),
        (0.70, 'Well Embeddings\n(Fourier/SAX/Custom)'),
        (0.55, 'Concentration Aggregation\n(Mean/Attention)'),
        (0.40, 'Drug Signature\n(Dose-Response + Temporal)'),
        (0.25, 'Final Drug Embedding')
    ]
    
    for i, (y, label) in enumerate(steps):
        ax2.add_patch(plt.Rectangle((0.2, y-0.05), 0.6, 0.08, fill=True, 
                                   color=['lightgray', 'lightgreen', 'lightskyblue', 'lightcoral', 'gold'][i], 
                                   alpha=0.7))
        ax2.text(0.5, y-0.01, label, ha='center', va='center', fontsize=10)
        
        if i < len(steps) - 1:
            ax2.arrow(0.5, y-0.05, 0, -0.08, head_width=0.03, head_length=0.02, fc='black', ec='black')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('drug_embedding_strategy.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Saved strategy diagram to: drug_embedding_strategy.png")
    plt.close()

def propose_implementation_plan():
    """Propose implementation plan for drug embeddings."""
    
    plan = """
    🚀 IMPLEMENTATION PLAN FOR DRUG EMBEDDINGS
    
    Phase 1: Data Organization (Week 1)
    - [ ] Create well_map table linking wells to drugs/concentrations
    - [ ] Build plate layout parser (96/384 well format)
    - [ ] Implement drug-concentration-replicate mapping
    - [ ] Validate data completeness across experiments
    
    Phase 2: Well-Level Embeddings (Week 2)
    - [ ] Optimize embedding methods for full plates (384 wells)
    - [ ] Implement parallel processing for large-scale analysis
    - [ ] Add quality control metrics per well
    - [ ] Store embeddings efficiently (HDF5/Parquet)
    
    Phase 3: Hierarchical Aggregation (Week 3)
    - [ ] Implement replicate aggregation methods
    - [ ] Build dose-response feature extractors
    - [ ] Create temporal dynamics analyzers
    - [ ] Design attention-based pooling mechanism
    
    Phase 4: Drug Signatures (Week 4)
    - [ ] Combine multi-concentration features
    - [ ] Extract interpretable drug response parameters
    - [ ] Implement signature comparison metrics
    - [ ] Create visualization tools
    
    Phase 5: Validation & Analysis (Week 5)
    - [ ] Compare drug signatures to known toxicity
    - [ ] Cluster drugs by response patterns
    - [ ] Validate against independent test set
    - [ ] Generate publication-ready figures
    """
    
    logger.info(plan)
    
    return plan

def example_drug_embedding_pipeline():
    """Show example code structure for drug embedding pipeline."""
    
    code_structure = '''
    # Example Drug Embedding Pipeline Structure
    
    class DrugEmbeddingPipeline:
        """Complete pipeline for drug-based embeddings."""
        
        def __init__(self, embedding_method='multi_scale'):
            self.well_embedder = WellEmbedder()
            self.concentration_aggregator = ConcentrationAggregator()
            self.drug_signature_extractor = DrugSignatureExtractor()
        
        def process_drug(self, drug_name: str) -> np.ndarray:
            """Generate embedding for a single drug."""
            
            # 1. Get all wells for this drug
            wells_data = self.load_drug_wells(drug_name)
            # Structure: {concentration: {replicate: time_series}}
            
            # 2. Generate well-level embeddings
            well_embeddings = {}
            for conc, replicates in wells_data.items():
                well_embeddings[conc] = []
                for rep_id, time_series in replicates.items():
                    embedding = self.well_embedder.embed(time_series)
                    well_embeddings[conc].append(embedding)
            
            # 3. Aggregate across replicates
            conc_embeddings = {}
            for conc, rep_embeddings in well_embeddings.items():
                conc_embedding = self.concentration_aggregator.aggregate(
                    rep_embeddings, 
                    method='attention'
                )
                conc_embeddings[conc] = conc_embedding
            
            # 4. Extract drug signature
            drug_embedding = self.drug_signature_extractor.extract(
                conc_embeddings,
                include_dose_response=True,
                include_temporal_dynamics=True
            )
            
            return drug_embedding
    
    class MultiScaleDrugEmbedder:
        """Multi-scale approach preserving information at all levels."""
        
        def embed(self, drug_data):
            features = []
            
            # Well-level statistics
            well_features = self.extract_well_features(drug_data)
            features.append(well_features)
            
            # Concentration-level patterns  
            conc_features = self.extract_concentration_features(drug_data)
            features.append(conc_features)
            
            # Dose-response characteristics
            dr_features = self.extract_dose_response_features(drug_data)
            features.append(dr_features)
            
            # Temporal dynamics
            temporal_features = self.extract_temporal_features(drug_data)
            features.append(temporal_features)
            
            # Concatenate all features
            return np.concatenate(features)
    '''
    
    logger.info("\n📝 Example Pipeline Structure:")
    logger.info(code_structure)

if __name__ == "__main__":
    logger.info("🎯 Mapping Drug-Based Embedding Strategy...")
    
    try:
        # Explore current data structure
        plate_info = explore_plate_drug_structure()
        
        # Design embedding strategy
        strategy = design_drug_embedding_strategy()
        
        # Create visual diagram
        create_visual_strategy_diagram()
        
        # Propose implementation plan
        plan = propose_implementation_plan()
        
        # Show example code structure
        example_drug_embedding_pipeline()
        
        logger.info("\n✅ Strategy mapping complete!")
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Create well-to-drug mapping table")
        logger.info("2. Implement hierarchical embedding pipeline")
        logger.info("3. Test on subset of drugs")
        logger.info("4. Scale to full dataset")
        
    except Exception as e:
        logger.error(f"Strategy mapping failed: {e}")
        raise