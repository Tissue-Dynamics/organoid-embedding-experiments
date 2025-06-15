#!/usr/bin/env python3
"""Run drug embeddings on multiple drugs with proper concentration handling."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDrugEmbeddingPipeline:
    """Optimized pipeline for drug embeddings."""
    
    def __init__(self):
        self.conn = self._connect_db()
        self.embedder = None
        self.all_signatures = {}
        
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
    
    def get_drugs_with_good_data(self, min_concentrations=3, min_wells=20):
        """Get drugs with sufficient data."""
        query = f"""
        SELECT 
            w.drug,
            COUNT(DISTINCT w.concentration) as n_concentrations,
            COUNT(DISTINCT w.well_number) as n_wells,
            STRING_AGG(DISTINCT CAST(w.concentration AS VARCHAR), ', ') as concentrations
        FROM supabase.public.well_map_data w
        WHERE w.drug IS NOT NULL 
        AND w.drug != ''
        AND w.drug != 'Ctrl'
        AND w.drug NOT LIKE '%media%'
        AND w.drug NOT LIKE '%Media%'
        AND w.drug NOT LIKE '%Hormones%'
        AND w.is_excluded = false
        GROUP BY w.drug
        HAVING COUNT(DISTINCT w.concentration) >= {min_concentrations}
        AND COUNT(DISTINCT w.well_number) >= {min_wells}
        ORDER BY n_wells DESC
        LIMIT 10
        """
        
        df = self.conn.execute(query).df()
        logger.info(f"\n📊 Drugs with good data coverage:\n{df}")
        return df
    
    def load_drug_data_optimized(self, drug_name: str, max_concentrations=6):
        """Load drug data with optimization."""
        logger.info(f"\n📥 Loading data for drug: {drug_name}")
        
        # Get concentrations for this drug
        conc_query = f"""
        SELECT DISTINCT concentration
        FROM supabase.public.well_map_data
        WHERE drug = '{drug_name}'
        AND is_excluded = false
        AND concentration > 0
        ORDER BY concentration
        LIMIT {max_concentrations}
        """
        
        concentrations = self.conn.execute(conc_query).df()['concentration'].tolist()
        
        if len(concentrations) == 0:
            logger.warning(f"No non-zero concentrations found for {drug_name}")
            return {}
        
        logger.info(f"Selected concentrations: {concentrations}")
        
        # Load data for each concentration
        drug_data = {}
        
        for conc in concentrations:
            # Get wells for this concentration (limit to 4 replicates)
            wells_query = f"""
            SELECT DISTINCT plate_id, well_number
            FROM supabase.public.well_map_data
            WHERE drug = '{drug_name}'
            AND concentration = {conc}
            AND is_excluded = false
            LIMIT 4
            """
            
            wells = self.conn.execute(wells_query).df()
            
            if len(wells) == 0:
                continue
                
            drug_data[conc] = {}
            
            for _, well in wells.iterrows():
                # Get hourly averaged time series
                ts_query = f"""
                SELECT 
                    AVG(median_o2) as median_o2
                FROM supabase.public.processed_data
                WHERE plate_id = '{well['plate_id']}'
                AND well_number = {well['well_number']}
                AND is_excluded = false
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY DATE_TRUNC('hour', timestamp)
                LIMIT 168  -- 1 week of hourly data
                """
                
                ts_df = self.conn.execute(ts_query).df()
                
                if len(ts_df) > 20:  # Minimum viable time series
                    drug_data[conc][well['well_number']] = ts_df['median_o2'].values
        
        logger.info(f"Loaded {len(drug_data)} concentrations with data")
        return drug_data
    
    def process_drug(self, drug_name: str):
        """Process a single drug."""
        # Load data
        drug_data = self.load_drug_data_optimized(drug_name)
        
        if len(drug_data) < 2:  # Need at least 2 concentrations
            logger.warning(f"Insufficient data for {drug_name}")
            return None
        
        # Initialize embedder
        if self.embedder is None:
            from embeddings import CustomFeaturesEmbedder
            self.embedder = CustomFeaturesEmbedder()
        
        # Generate embeddings for each concentration
        conc_features = {}
        
        for conc, wells_data in drug_data.items():
            if len(wells_data) == 0:
                continue
                
            # Create matrix
            time_series_list = list(wells_data.values())
            max_len = max(len(ts) for ts in time_series_list)
            X = np.zeros((len(time_series_list), max_len))
            
            for i, ts in enumerate(time_series_list):
                X[i, :len(ts)] = ts
            
            # Generate embeddings
            embeddings = self.embedder.fit_transform(X)
            
            # Aggregate across replicates
            mean_emb = np.mean(embeddings, axis=0)
            std_emb = np.std(embeddings, axis=0) if len(embeddings) > 1 else np.zeros_like(mean_emb)
            
            conc_features[conc] = np.concatenate([mean_emb, std_emb])
        
        # Create drug signature
        sorted_concs = sorted(conc_features.keys())
        
        # Dose-response features
        dr_features = []
        n_features = min(10, len(list(conc_features.values())[0]) // 2)  # Use mean features only
        
        for i in range(n_features):
            response = [conc_features[c][i] for c in sorted_concs]
            if len(sorted_concs) > 2:
                # Fit dose-response curve
                log_concs = np.log10(sorted_concs)
                coef = np.polyfit(log_concs, response, 2)  # Quadratic fit
                dr_features.extend(coef)
            else:
                dr_features.extend([0, 0, 0])
        
        # Combine all features
        all_features = []
        all_features.extend(dr_features)
        
        # Add concentration-specific features
        for conc in sorted_concs:
            all_features.extend(conc_features[conc][:20])  # First 20 features per concentration
        
        # Global statistics
        all_embeddings = list(conc_features.values())
        all_features.extend(np.mean(all_embeddings, axis=0)[:20])
        all_features.extend(np.std(all_embeddings, axis=0)[:20] if len(all_embeddings) > 1 else np.zeros(20))
        
        drug_signature = np.array(all_features)
        
        logger.info(f"✅ {drug_name}: signature shape {drug_signature.shape}")
        
        # Store metadata
        self.all_signatures[drug_name] = {
            'signature': drug_signature,
            'concentrations': sorted_concs,
            'n_concentrations': len(sorted_concs),
            'n_wells': sum(len(wells) for wells in drug_data.values())
        }
        
        return drug_signature
    
    def process_multiple_drugs(self, drug_names):
        """Process multiple drugs."""
        signatures = {}
        
        for drug in tqdm(drug_names, desc="Processing drugs"):
            try:
                sig = self.process_drug(drug)
                if sig is not None:
                    signatures[drug] = sig
            except Exception as e:
                logger.error(f"Failed to process {drug}: {e}")
        
        return signatures
    
    def visualize_drug_space(self, signatures):
        """Visualize drug embedding space."""
        if len(signatures) < 3:
            logger.warning("Need at least 3 drugs for visualization")
            return
        
        # Prepare data
        drug_names = list(signatures.keys())
        X = np.array([signatures[drug] for drug in drug_names])
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=min(3, len(drug_names)-1))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5))
        
        # 2D PCA
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], s=200, alpha=0.7, 
                             c=range(len(drug_names)), cmap='tab10')
        
        for i, drug in enumerate(drug_names):
            ax1.annotate(drug, (X_pca[i, 0], X_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('Drug Embedding Space (2D)')
        ax1.grid(True, alpha=0.3)
        
        # Feature importance
        ax2 = fig.add_subplot(132)
        feature_var = np.var(X_scaled, axis=0)
        top_features_idx = np.argsort(feature_var)[-20:]
        ax2.barh(range(20), feature_var[top_features_idx])
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Feature Index')
        ax2.set_title('Top 20 Most Variable Features')
        
        # Drug metadata
        ax3 = fig.add_subplot(133)
        n_concs = [self.all_signatures[drug]['n_concentrations'] for drug in drug_names]
        n_wells = [self.all_signatures[drug]['n_wells'] for drug in drug_names]
        
        ax3.scatter(n_concs, n_wells, s=100, alpha=0.6)
        for i, drug in enumerate(drug_names):
            ax3.annotate(drug, (n_concs[i], n_wells[i]), 
                        xytext=(2, 2), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Number of Concentrations')
        ax3.set_ylabel('Number of Wells')
        ax3.set_title('Drug Data Coverage')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('drug_embedding_space.png', dpi=150, bbox_inches='tight')
        logger.info("💾 Saved visualization to: drug_embedding_space.png")
        
        # Distance matrix
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(X_scaled, metric='euclidean'))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(distances, xticklabels=drug_names, yticklabels=drug_names, 
                    cmap='viridis', annot=True, fmt='.1f')
        plt.title('Drug Signature Distance Matrix')
        plt.tight_layout()
        plt.savefig('drug_distance_matrix.png', dpi=150, bbox_inches='tight')
        logger.info("💾 Saved distance matrix to: drug_distance_matrix.png")

def main():
    """Run optimized drug embedding pipeline."""
    logger.info("🧬 Starting Optimized Drug Embedding Pipeline...")
    
    # Initialize pipeline
    pipeline = OptimizedDrugEmbeddingPipeline()
    
    # Get drugs with good data
    drugs_df = pipeline.get_drugs_with_good_data(min_concentrations=3, min_wells=20)
    
    if len(drugs_df) == 0:
        logger.error("No drugs found with sufficient data")
        return
    
    # Select drugs to process
    drug_names = drugs_df.head(8)['drug'].tolist()
    logger.info(f"\n🎯 Processing {len(drug_names)} drugs: {drug_names}")
    
    # Process drugs
    signatures = pipeline.process_multiple_drugs(drug_names)
    
    if len(signatures) > 0:
        # Visualize
        pipeline.visualize_drug_space(signatures)
        
        # Save results
        results = {
            'signatures': signatures,
            'metadata': pipeline.all_signatures
        }
        
        joblib.dump(results, 'drug_embeddings_results.joblib')
        logger.info(f"\n💾 Saved {len(signatures)} drug signatures to: drug_embeddings_results.joblib")
        
        # Summary
        logger.info("\n📊 SUMMARY:")
        for drug, meta in pipeline.all_signatures.items():
            logger.info(f"\n{drug}:")
            logger.info(f"  - Signature dimension: {meta['signature'].shape}")
            logger.info(f"  - Concentrations: {meta['concentrations']}")
            logger.info(f"  - Total wells: {meta['n_wells']}")
    
    pipeline.conn.close()

if __name__ == "__main__":
    main()