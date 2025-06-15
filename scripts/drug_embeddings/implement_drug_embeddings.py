#!/usr/bin/env python3
"""Implement drug-based embeddings using real organoid data structure."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugEmbeddingPipeline:
    """Pipeline for creating drug-based embeddings from organoid data."""
    
    def __init__(self):
        self.conn = self._connect_db()
        self.well_embedder = None
        self.results = {}
        
    def _connect_db(self):
        """Connect to database."""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        database_url = os.getenv('DATABASE_URL')
        
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
        return conn
    
    def get_drug_overview(self):
        """Get overview of drugs and their experimental coverage."""
        query = """
        SELECT 
            w.drug,
            COUNT(DISTINCT w.plate_id) as n_plates,
            COUNT(DISTINCT w.concentration) as n_concentrations,
            COUNT(DISTINCT w.well_number) as n_wells,
            COUNT(DISTINCT p.timestamp) as n_timepoints,
            STRING_AGG(DISTINCT CAST(w.concentration AS VARCHAR) || w.units, ', ') as concentrations
        FROM supabase.public.well_map_data w
        JOIN supabase.public.processed_data p ON w.plate_id = p.plate_id AND w.well_number = p.well_number
        WHERE w.drug IS NOT NULL 
        AND w.drug != 'control'
        AND p.is_excluded = false
        GROUP BY w.drug
        ORDER BY n_wells DESC
        LIMIT 20
        """
        
        df = self.conn.execute(query).df()
        logger.info(f"\\n📊 Drug Overview:\\n{df}")
        return df
    
    def load_drug_data(self, drug_name: str) -> Dict[float, Dict[int, np.ndarray]]:
        """Load all time series data for a specific drug."""
        logger.info(f"\\n📥 Loading data for drug: {drug_name}")
        
        # Get well mappings for this drug
        well_query = f"""
        SELECT DISTINCT
            w.plate_id,
            w.well_number,
            w.concentration,
            w.units
        FROM supabase.public.well_map_data w
        WHERE w.drug = '{drug_name}'
        AND w.is_excluded = false
        ORDER BY w.concentration, w.well_number
        """
        
        wells_df = self.conn.execute(well_query).df()
        logger.info(f"Found {len(wells_df)} wells for {drug_name}")
        logger.info(f"Concentrations: {sorted(wells_df['concentration'].unique())}")
        
        # Load time series data
        drug_data = {}
        
        for concentration in sorted(wells_df['concentration'].unique()):
            drug_data[concentration] = {}
            
            conc_wells = wells_df[wells_df['concentration'] == concentration]
            
            # Limit to first 4 wells per concentration for speed
            for _, well in conc_wells.head(4).iterrows():
                # Get time series for this well - sample every 10th point for speed
                ts_query = f"""
                SELECT timestamp, median_o2
                FROM (
                    SELECT timestamp, median_o2, 
                           ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                    FROM supabase.public.processed_data
                    WHERE plate_id = '{well['plate_id']}'
                    AND well_number = {well['well_number']}
                    AND is_excluded = false
                ) t
                WHERE rn % 10 = 0
                ORDER BY timestamp
                """
                
                ts_df = self.conn.execute(ts_query).df()
                
                if len(ts_df) > 0:
                    # Convert to time series array
                    ts_array = ts_df['median_o2'].values
                    drug_data[concentration][well['well_number']] = ts_array
        
        logger.info(f"Loaded data structure: {concentration} -> {len(drug_data[concentration])} wells")
        
        return drug_data
    
    def generate_well_embeddings(self, drug_data: Dict[float, Dict[int, np.ndarray]]) -> Dict[float, List[np.ndarray]]:
        """Generate embeddings for each well."""
        from embeddings import FourierEmbedder, CustomFeaturesEmbedder
        
        # Use fast embedders for now
        self.well_embedder = CustomFeaturesEmbedder()
        
        well_embeddings = {}
        
        for concentration, wells in drug_data.items():
            well_embeddings[concentration] = []
            
            # Create matrix of time series for this concentration
            time_series_list = list(wells.values())
            
            if time_series_list:
                # Pad to same length
                max_len = max(len(ts) for ts in time_series_list)
                X = np.zeros((len(time_series_list), max_len))
                
                for i, ts in enumerate(time_series_list):
                    X[i, :len(ts)] = ts
                
                # Generate embeddings
                embeddings = self.well_embedder.fit_transform(X)
                well_embeddings[concentration] = embeddings
                
                logger.info(f"Concentration {concentration}: {embeddings.shape}")
        
        return well_embeddings
    
    def aggregate_concentration_embeddings(self, well_embeddings: Dict[float, List[np.ndarray]]) -> Dict[float, np.ndarray]:
        """Aggregate embeddings across replicates for each concentration."""
        conc_embeddings = {}
        
        for concentration, embeddings in well_embeddings.items():
            if len(embeddings) > 0:
                # Simple mean pooling for now
                mean_embedding = np.mean(embeddings, axis=0)
                std_embedding = np.std(embeddings, axis=0)
                
                # Combine mean and std for richer representation
                conc_embeddings[concentration] = np.concatenate([mean_embedding, std_embedding])
                
                logger.info(f"Concentration {concentration}: {conc_embeddings[concentration].shape}")
        
        return conc_embeddings
    
    def extract_drug_signature(self, conc_embeddings: Dict[float, np.ndarray], drug_name: str) -> np.ndarray:
        """Extract final drug signature from concentration embeddings."""
        if not conc_embeddings:
            raise ValueError(f"No embeddings found for {drug_name}")
        
        # Sort by concentration
        sorted_concs = sorted(conc_embeddings.keys())
        
        # Extract dose-response features
        dr_features = self._extract_dose_response_features(conc_embeddings, sorted_concs)
        
        # Concatenate all concentration embeddings
        all_embeddings = [conc_embeddings[c] for c in sorted_concs]
        concat_features = np.concatenate(all_embeddings)
        
        # Global statistics
        global_mean = np.mean(all_embeddings, axis=0)
        global_std = np.std(all_embeddings, axis=0)
        
        # Final drug signature
        drug_signature = np.concatenate([
            dr_features,
            concat_features,
            global_mean,
            global_std
        ])
        
        logger.info(f"Drug signature for {drug_name}: {drug_signature.shape}")
        
        return drug_signature
    
    def _extract_dose_response_features(self, conc_embeddings: Dict[float, np.ndarray], sorted_concs: List[float]) -> np.ndarray:
        """Extract dose-response curve features."""
        features = []
        
        # Get first principal component across concentrations
        embeddings_matrix = np.array([conc_embeddings[c][:10] for c in sorted_concs])  # Use first 10 features
        
        # Concentration-response slopes
        for i in range(min(10, embeddings_matrix.shape[1])):
            response = embeddings_matrix[:, i]
            log_concs = np.log10(sorted_concs)
            
            # Fit linear trend
            slope = np.polyfit(log_concs, response, 1)[0]
            features.append(slope)
        
        # Hill equation parameters would go here
        # For now, just use simple features
        features.extend([
            len(sorted_concs),  # Number of concentrations tested
            np.log10(max(sorted_concs) / min(sorted_concs)),  # Concentration range
        ])
        
        return np.array(features)
    
    def process_drug(self, drug_name: str) -> np.ndarray:
        """Complete pipeline for one drug."""
        # Load data
        drug_data = self.load_drug_data(drug_name)
        
        if not drug_data:
            raise ValueError(f"No data found for drug: {drug_name}")
        
        # Generate well embeddings
        well_embeddings = self.generate_well_embeddings(drug_data)
        
        # Aggregate by concentration
        conc_embeddings = self.aggregate_concentration_embeddings(well_embeddings)
        
        # Extract drug signature
        drug_signature = self.extract_drug_signature(conc_embeddings, drug_name)
        
        # Store results
        self.results[drug_name] = {
            'signature': drug_signature,
            'concentrations': sorted(drug_data.keys()),
            'n_wells': sum(len(wells) for wells in drug_data.values())
        }
        
        return drug_signature
    
    def process_multiple_drugs(self, drug_list: List[str]):
        """Process multiple drugs and create comparison."""
        signatures = {}
        
        for drug in tqdm(drug_list, desc="Processing drugs"):
            try:
                signature = self.process_drug(drug)
                signatures[drug] = signature
            except Exception as e:
                logger.error(f"Failed to process {drug}: {e}")
        
        return signatures
    
    def visualize_drug_signatures(self, signatures: Dict[str, np.ndarray]):
        """Visualize drug signatures using dimensionality reduction."""
        if len(signatures) < 2:
            logger.warning("Need at least 2 drugs to visualize")
            return
        
        # Create signature matrix
        drug_names = list(signatures.keys())
        X = np.array([signatures[drug] for drug in drug_names])
        
        # PCA for visualization
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.6)
        
        # Add labels
        for i, drug in enumerate(drug_names):
            plt.annotate(drug, (X_pca[i, 0], X_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('Drug Signatures - PCA Projection')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('drug_signatures_pca.png', dpi=150)
        plt.close()
        
        logger.info("💾 Saved drug signatures visualization to: drug_signatures_pca.png")

def main():
    """Run drug embedding pipeline on real data."""
    logger.info("🧬 Starting Drug Embedding Pipeline...")
    
    # Initialize pipeline
    pipeline = DrugEmbeddingPipeline()
    
    # Get drug overview
    drug_overview = pipeline.get_drug_overview()
    
    # Select top drugs with good data coverage, excluding empty and control
    top_drugs = drug_overview[
        (drug_overview['drug'] != '') & 
        (drug_overview['drug'] != 'Ctrl') &
        (drug_overview['drug'].notna())
    ].head(5)['drug'].tolist()
    logger.info(f"\\n🎯 Processing top {len(top_drugs)} drugs: {top_drugs}")
    
    # Process drugs
    signatures = pipeline.process_multiple_drugs(top_drugs)
    
    # Visualize results
    if signatures:
        pipeline.visualize_drug_signatures(signatures)
        
        # Save results
        output_file = 'drug_signatures.joblib'
        joblib.dump(signatures, output_file)
        logger.info(f"\\n💾 Saved {len(signatures)} drug signatures to: {output_file}")
    
    # Summary statistics
    logger.info("\\n📊 SUMMARY:")
    for drug, result in pipeline.results.items():
        logger.info(f"  {drug}:")
        logger.info(f"    - Signature dimension: {result['signature'].shape}")
        logger.info(f"    - Concentrations tested: {result['concentrations']}")
        logger.info(f"    - Total wells: {result['n_wells']}")
    
    pipeline.conn.close()

if __name__ == "__main__":
    main()