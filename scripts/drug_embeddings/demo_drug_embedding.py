#!/usr/bin/env python3
"""Quick demo of drug embedding pipeline with one drug."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_drug_embedding():
    """Demo drug embedding with minimal data."""
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
    
    # Pick one drug with reasonable data
    drug_name = 'Sanofi-1'
    logger.info(f"\n🎯 Demo: Processing drug '{drug_name}'")
    
    # Get well mappings for this drug - limit concentrations
    well_query = f"""
    SELECT DISTINCT
        w.plate_id,
        w.well_number,
        w.concentration,
        w.units
    FROM supabase.public.well_map_data w
    WHERE w.drug = '{drug_name}'
    AND w.is_excluded = false
    AND w.concentration IN (
        SELECT DISTINCT concentration 
        FROM supabase.public.well_map_data 
        WHERE drug = '{drug_name}'
        ORDER BY concentration
        LIMIT 5  -- Only 5 concentrations for demo
    )
    ORDER BY w.concentration, w.well_number
    LIMIT 20  -- Max 20 wells total
    """
    
    wells_df = conn.execute(well_query).df()
    logger.info(f"Found {len(wells_df)} wells across {wells_df['concentration'].nunique()} concentrations")
    logger.info(f"Concentrations: {sorted(wells_df['concentration'].unique())}")
    
    # Load time series data
    drug_data = {}
    
    for concentration in sorted(wells_df['concentration'].unique()):
        drug_data[concentration] = {}
        
        conc_wells = wells_df[wells_df['concentration'] == concentration]
        logger.info(f"\nConcentration {concentration}: {len(conc_wells)} wells")
        
        for _, well in conc_wells.iterrows():
            # Get downsampled time series
            ts_query = f"""
            SELECT 
                AVG(median_o2) as median_o2,
                MIN(timestamp) as timestamp
            FROM supabase.public.processed_data
            WHERE plate_id = '{well['plate_id']}'
            AND well_number = {well['well_number']}
            AND is_excluded = false
            GROUP BY DATE_TRUNC('hour', timestamp)  -- Group by hour
            ORDER BY timestamp
            LIMIT 200  -- Max 200 time points
            """
            
            ts_df = conn.execute(ts_query).df()
            
            if len(ts_df) > 10:
                ts_array = ts_df['median_o2'].values
                drug_data[concentration][well['well_number']] = ts_array
                logger.info(f"  Well {well['well_number']}: {len(ts_array)} time points")
    
    # Generate embeddings
    from embeddings import CustomFeaturesEmbedder
    embedder = CustomFeaturesEmbedder()
    
    logger.info("\n📊 Generating embeddings...")
    well_embeddings = {}
    
    for concentration, wells in drug_data.items():
        if wells:
            # Create matrix
            time_series_list = list(wells.values())
            max_len = max(len(ts) for ts in time_series_list)
            X = np.zeros((len(time_series_list), max_len))
            
            for i, ts in enumerate(time_series_list):
                X[i, :len(ts)] = ts
            
            # Generate embeddings
            embeddings = embedder.fit_transform(X)
            well_embeddings[concentration] = embeddings
            logger.info(f"Concentration {concentration}: {embeddings.shape}")
    
    # Aggregate by concentration
    logger.info("\n🔄 Aggregating embeddings...")
    conc_embeddings = {}
    
    for concentration, embeddings in well_embeddings.items():
        if len(embeddings) > 0:
            mean_embedding = np.mean(embeddings, axis=0)
            std_embedding = np.std(embeddings, axis=0)
            conc_embeddings[concentration] = np.concatenate([mean_embedding, std_embedding])
            logger.info(f"Concentration {concentration}: {conc_embeddings[concentration].shape}")
    
    # Create drug signature
    logger.info("\n🧬 Creating drug signature...")
    sorted_concs = sorted(conc_embeddings.keys())
    
    # Simple dose-response features
    dr_features = []
    for i in range(min(5, len(list(conc_embeddings.values())[0]))):
        response = [conc_embeddings[c][i] for c in sorted_concs]
        if len(sorted_concs) > 1:
            slope = np.polyfit(range(len(sorted_concs)), response, 1)[0]
        else:
            slope = 0
        dr_features.append(slope)
    
    # Concatenate all features
    all_embeddings = [conc_embeddings[c] for c in sorted_concs]
    drug_signature = np.concatenate([
        np.array(dr_features),
        np.concatenate(all_embeddings),
        np.mean(all_embeddings, axis=0),
        np.std(all_embeddings, axis=0)
    ])
    
    logger.info(f"\n✅ Drug signature for {drug_name}: {drug_signature.shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sample time series
    ax = axes[0, 0]
    for conc in sorted_concs[:3]:  # First 3 concentrations
        if conc in drug_data and drug_data[conc]:
            ts = list(drug_data[conc].values())[0]
            ax.plot(ts, label=f'{conc} μM', alpha=0.7)
    ax.set_title(f'Sample Time Series - {drug_name}')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Median O2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Embedding space
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_concs)))
    for i, (conc, emb) in enumerate(conc_embeddings.items()):
        ax.scatter([i], [np.mean(emb[:10])], c=[colors[i]], s=100, label=f'{conc} μM')
    ax.set_title('Mean Feature Value by Concentration')
    ax.set_xlabel('Concentration Index')
    ax.set_ylabel('Mean Feature Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Drug signature visualization
    ax = axes[1, 0]
    ax.bar(range(len(drug_signature[:20])), drug_signature[:20])
    ax.set_title('Drug Signature (First 20 Features)')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Value')
    
    # 4. Dose response
    ax = axes[1, 1]
    if len(sorted_concs) > 1:
        feature_idx = 0  # First feature
        response = [conc_embeddings[c][feature_idx] for c in sorted_concs]
        ax.plot(range(len(sorted_concs)), response, 'o-')
        ax.set_xticks(range(len(sorted_concs)))
        ax.set_xticklabels([f'{c:.1f}' for c in sorted_concs], rotation=45)
        ax.set_title('Dose-Response (Feature 1)')
        ax.set_xlabel('Concentration (μM)')
        ax.set_ylabel('Feature Value')
    
    plt.tight_layout()
    plt.savefig('drug_embedding_demo.png', dpi=150, bbox_inches='tight')
    logger.info(f"💾 Saved visualization to: drug_embedding_demo.png")
    
    conn.close()
    
    return drug_signature, drug_data

if __name__ == "__main__":
    logger.info("🧬 Running Drug Embedding Demo...")
    
    try:
        signature, data = demo_drug_embedding()
        
        logger.info("\n📊 DEMO COMPLETE!")
        logger.info(f"Generated drug signature with {len(signature)} features")
        logger.info(f"Processed {sum(len(wells) for wells in data.values())} time series")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise