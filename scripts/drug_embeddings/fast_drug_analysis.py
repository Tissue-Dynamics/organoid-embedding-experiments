#!/usr/bin/env python3
"""Fast drug analysis with minimal data loading for quick results."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastDrugAnalysis:
    """Fast analysis of drug responses."""
    
    def __init__(self):
        self.conn = self._connect_db()
        
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
    
    def get_drug_summary_features(self, limit=20):
        """Get summary features for drugs directly from database."""
        logger.info("📊 Extracting drug summary features...")
        
        query = f"""
        WITH well_time_ranges AS (
            SELECT 
                w.drug,
                w.concentration,
                w.well_number,
                w.plate_id,
                MIN(p.timestamp) as min_time,
                MAX(p.timestamp) as max_time
            FROM supabase.public.well_map_data w
            JOIN supabase.public.processed_data p 
                ON w.plate_id = p.plate_id AND w.well_number = p.well_number
            WHERE w.drug IS NOT NULL 
            AND w.drug != ''
            AND w.drug != 'Ctrl'
            AND w.drug NOT LIKE '%media%'
            AND w.drug NOT LIKE '%Media%'
            AND w.is_excluded = false
            AND p.is_excluded = false
            GROUP BY w.drug, w.concentration, w.well_number, w.plate_id
        ),
        drug_stats AS (
            SELECT 
                wt.drug,
                wt.concentration,
                wt.well_number,
                wt.plate_id,
                AVG(p.median_o2) as mean_o2,
                STDDEV(p.median_o2) as std_o2,
                MIN(p.median_o2) as min_o2,
                MAX(p.median_o2) as max_o2,
                COUNT(*) as n_measurements,
                -- Early response (first 24 hours)
                AVG(CASE WHEN p.timestamp < wt.min_time + INTERVAL '24 hours' 
                    THEN p.median_o2 END) as early_o2,
                -- Late response (after 72 hours)
                AVG(CASE WHEN p.timestamp > wt.min_time + INTERVAL '72 hours' 
                    THEN p.median_o2 END) as late_o2
            FROM well_time_ranges wt
            JOIN supabase.public.processed_data p 
                ON wt.plate_id = p.plate_id AND wt.well_number = p.well_number
            WHERE p.is_excluded = false
            GROUP BY wt.drug, wt.concentration, wt.well_number, wt.plate_id
        ),
        drug_aggregated AS (
            SELECT 
                drug,
                concentration,
                -- Aggregate across wells
                AVG(mean_o2) as avg_mean_o2,
                AVG(std_o2) as avg_std_o2,
                AVG(min_o2) as avg_min_o2,
                AVG(max_o2) as avg_max_o2,
                AVG(early_o2) as avg_early_o2,
                AVG(late_o2) as avg_late_o2,
                AVG(late_o2 - early_o2) as avg_response_change,
                COUNT(DISTINCT well_number) as n_wells
            FROM drug_stats
            WHERE concentration > 0  -- Non-control
            GROUP BY drug, concentration
        ),
        drug_features AS (
            SELECT 
                drug,
                -- Concentration-based features
                COUNT(DISTINCT concentration) as n_concentrations,
                MIN(concentration) as min_concentration,
                MAX(concentration) as max_concentration,
                -- Response features across concentrations
                MIN(avg_mean_o2) as min_response,
                MAX(avg_mean_o2) as max_response,
                MAX(avg_mean_o2) - MIN(avg_mean_o2) as response_range,
                AVG(avg_std_o2) as avg_variability,
                -- Dose-response slope (simple linear approximation)
                CORR(LOG10(concentration + 0.001), avg_mean_o2) as dose_response_correlation,
                -- Early vs late response
                AVG(avg_early_o2) as overall_early_response,
                AVG(avg_late_o2) as overall_late_response,
                AVG(avg_response_change) as overall_response_change,
                -- Data quality
                SUM(n_wells) as total_wells
            FROM drug_aggregated
            GROUP BY drug
            HAVING COUNT(DISTINCT concentration) >= 3  -- At least 3 concentrations
        )
        SELECT *
        FROM drug_features
        ORDER BY total_wells DESC
        LIMIT {limit}
        """
        
        df = self.conn.execute(query).df()
        logger.info(f"✅ Extracted features for {len(df)} drugs")
        return df
    
    def create_drug_embeddings_from_features(self, features_df):
        """Create embeddings from summary features."""
        # Select numeric features for embedding
        feature_cols = [
            'n_concentrations', 'min_concentration', 'max_concentration',
            'min_response', 'max_response', 'response_range', 'avg_variability',
            'dose_response_correlation', 'overall_early_response', 
            'overall_late_response', 'overall_response_change'
        ]
        
        X = features_df[feature_cols].values
        drug_names = features_df['drug'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, drug_names, feature_cols
    
    def analyze_drug_clusters(self, X_scaled, drug_names):
        """Perform clustering analysis on drugs."""
        logger.info("\n🔍 Analyzing drug clusters...")
        
        # Optimal number of clusters
        inertias = []
        K_range = range(2, min(10, len(drug_names)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method - pick k=4 for now
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Create results dataframe
        cluster_df = pd.DataFrame({
            'drug': drug_names,
            'cluster': labels
        })
        
        logger.info(f"\n📊 Drug clusters (k={optimal_k}):")
        for cluster in range(optimal_k):
            drugs_in_cluster = cluster_df[cluster_df['cluster'] == cluster]['drug'].tolist()
            logger.info(f"  Cluster {cluster}: {', '.join(drugs_in_cluster)}")
        
        return labels, kmeans
    
    def visualize_drug_space(self, X_scaled, drug_names, labels, feature_cols):
        """Create comprehensive visualization of drug space."""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. PCA visualization
        ax1 = plt.subplot(3, 3, 1)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=200, alpha=0.7)
        
        for i, drug in enumerate(drug_names):
            ax1.annotate(drug, (X_pca[i, 0], X_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('Drug Space - PCA Projection')
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature importance
        ax2 = plt.subplot(3, 3, 2)
        loadings = pca.components_[0]
        feature_importance = np.abs(loadings)
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        ax2.barh(range(len(feature_cols)), feature_importance[sorted_idx])
        ax2.set_yticks(range(len(feature_cols)))
        ax2.set_yticklabels([feature_cols[i] for i in sorted_idx])
        ax2.set_xlabel('Absolute Loading')
        ax2.set_title('Feature Importance (PC1)')
        
        # 3. Hierarchical clustering dendrogram
        ax3 = plt.subplot(3, 3, 3)
        linkage_matrix = linkage(X_scaled, method='ward')
        dendrogram(linkage_matrix, labels=drug_names, orientation='right', ax=ax3)
        ax3.set_title('Hierarchical Clustering')
        ax3.set_xlabel('Distance')
        
        # 4. Dose-response correlation distribution
        ax4 = plt.subplot(3, 3, 4)
        dr_corr = X_scaled[:, feature_cols.index('dose_response_correlation')]
        ax4.hist(dr_corr, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Dose-Response Correlation')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Dose-Response Patterns')
        ax4.grid(True, alpha=0.3)
        
        # 5. Response range vs variability
        ax5 = plt.subplot(3, 3, 5)
        response_range = X_scaled[:, feature_cols.index('response_range')]
        variability = X_scaled[:, feature_cols.index('avg_variability')]
        
        scatter = ax5.scatter(response_range, variability, c=labels, cmap='tab10', s=100, alpha=0.7)
        for i, drug in enumerate(drug_names):
            ax5.annotate(drug, (response_range[i], variability[i]), 
                        xytext=(2, 2), textcoords='offset points', fontsize=6)
        
        ax5.set_xlabel('Response Range (Standardized)')
        ax5.set_ylabel('Average Variability (Standardized)')
        ax5.set_title('Response Range vs Variability')
        ax5.grid(True, alpha=0.3)
        
        # 6. Early vs Late response
        ax6 = plt.subplot(3, 3, 6)
        early = X_scaled[:, feature_cols.index('overall_early_response')]
        late = X_scaled[:, feature_cols.index('overall_late_response')]
        
        scatter = ax6.scatter(early, late, c=labels, cmap='tab10', s=100, alpha=0.7)
        for i, drug in enumerate(drug_names):
            ax6.annotate(drug, (early[i], late[i]), 
                        xytext=(2, 2), textcoords='offset points', fontsize=6)
        
        ax6.set_xlabel('Early Response (Standardized)')
        ax6.set_ylabel('Late Response (Standardized)')
        ax6.set_title('Early vs Late Response')
        ax6.grid(True, alpha=0.3)
        
        # 7. Cluster characteristics
        ax7 = plt.subplot(3, 3, 7)
        cluster_means = []
        for cluster in range(max(labels) + 1):
            cluster_mask = labels == cluster
            cluster_mean = X_scaled[cluster_mask].mean(axis=0)
            cluster_means.append(cluster_mean)
        
        cluster_means = np.array(cluster_means)
        im = ax7.imshow(cluster_means.T, aspect='auto', cmap='RdBu_r')
        ax7.set_yticks(range(len(feature_cols)))
        ax7.set_yticklabels(feature_cols)
        ax7.set_xticks(range(max(labels) + 1))
        ax7.set_xticklabels([f'Cluster {i}' for i in range(max(labels) + 1)])
        ax7.set_title('Cluster Feature Profiles')
        plt.colorbar(im, ax=ax7)
        
        # 8. Number of concentrations tested
        ax8 = plt.subplot(3, 3, 8)
        n_concs = X_scaled[:, feature_cols.index('n_concentrations')]
        ax8.bar(range(len(drug_names)), n_concs)
        ax8.set_xticks(range(len(drug_names)))
        ax8.set_xticklabels(drug_names, rotation=90)
        ax8.set_ylabel('N Concentrations (Standardized)')
        ax8.set_title('Data Coverage by Drug')
        
        # 9. Response change distribution
        ax9 = plt.subplot(3, 3, 9)
        response_change = X_scaled[:, feature_cols.index('overall_response_change')]
        
        for cluster in range(max(labels) + 1):
            cluster_data = response_change[labels == cluster]
            ax9.hist(cluster_data, bins=10, alpha=0.5, label=f'Cluster {cluster}')
        
        ax9.set_xlabel('Response Change (Late - Early)')
        ax9.set_ylabel('Count')
        ax9.set_title('Response Change by Cluster')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('drug_response_analysis.png', dpi=150, bbox_inches='tight')
        logger.info("💾 Saved comprehensive analysis to: drug_response_analysis.png")
        
    def identify_toxicity_patterns(self, X_scaled, drug_names, feature_cols, features_df):
        """Identify potential toxicity patterns."""
        logger.info("\n⚠️ Identifying potential toxicity patterns...")
        
        # Create toxicity score based on multiple factors
        toxicity_features = {
            'high_response_range': X_scaled[:, feature_cols.index('response_range')],
            'high_variability': X_scaled[:, feature_cols.index('avg_variability')],
            'negative_dose_response': -X_scaled[:, feature_cols.index('dose_response_correlation')],
            'late_deterioration': -X_scaled[:, feature_cols.index('overall_response_change')]
        }
        
        # Combine into toxicity score
        toxicity_score = np.mean(list(toxicity_features.values()), axis=0)
        
        # Rank drugs by toxicity
        toxicity_ranking = pd.DataFrame({
            'drug': drug_names,
            'toxicity_score': toxicity_score,
            'response_range': features_df['response_range'].values,
            'dose_response_corr': features_df['dose_response_correlation'].values,
            'response_change': features_df['overall_response_change'].values
        }).sort_values('toxicity_score', ascending=False)
        
        logger.info("\n🚨 Top 10 potentially toxic drugs:")
        print(toxicity_ranking.head(10).to_string(index=False))
        
        # Create toxicity visualization
        plt.figure(figsize=(12, 8))
        
        # Toxicity score distribution
        plt.subplot(2, 2, 1)
        plt.hist(toxicity_score, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(toxicity_score, 75), color='red', linestyle='--', 
                   label='75th percentile')
        plt.xlabel('Toxicity Score')
        plt.ylabel('Count')
        plt.title('Distribution of Toxicity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Top toxic drugs
        plt.subplot(2, 2, 2)
        top_n = 10
        top_drugs = toxicity_ranking.head(top_n)
        plt.barh(range(top_n), top_drugs['toxicity_score'])
        plt.yticks(range(top_n), top_drugs['drug'])
        plt.xlabel('Toxicity Score')
        plt.title(f'Top {top_n} Drugs by Toxicity Score')
        plt.grid(True, alpha=0.3)
        
        # Toxicity components
        plt.subplot(2, 2, 3)
        components = pd.DataFrame(toxicity_features).T
        components.columns = drug_names
        top_toxic_drugs = toxicity_ranking.head(5)['drug'].values
        
        for drug in top_toxic_drugs:
            if drug in components.columns:
                plt.plot(components[drug], marker='o', label=drug, alpha=0.7)
        
        plt.xticks(range(len(toxicity_features)), list(toxicity_features.keys()), rotation=45)
        plt.ylabel('Standardized Score')
        plt.title('Toxicity Components for Top Drugs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Dose-response vs response change
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(features_df['dose_response_correlation'], 
                            features_df['overall_response_change'],
                            c=toxicity_score, cmap='YlOrRd', s=100, alpha=0.7)
        
        for i, drug in enumerate(drug_names):
            if toxicity_score[i] > np.percentile(toxicity_score, 75):
                plt.annotate(drug, 
                           (features_df['dose_response_correlation'].iloc[i], 
                            features_df['overall_response_change'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Dose-Response Correlation')
        plt.ylabel('Response Change (Late - Early)')
        plt.title('Toxicity Indicators')
        plt.colorbar(scatter, label='Toxicity Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('drug_toxicity_analysis.png', dpi=150, bbox_inches='tight')
        logger.info("💾 Saved toxicity analysis to: drug_toxicity_analysis.png")
        
        return toxicity_ranking

def main():
    """Run fast drug analysis."""
    logger.info("🚀 Starting Fast Drug Response Analysis...")
    
    analyzer = FastDrugAnalysis()
    
    # Get drug features
    features_df = analyzer.get_drug_summary_features(limit=30)
    
    # Create embeddings
    X_scaled, drug_names, feature_cols = analyzer.create_drug_embeddings_from_features(features_df)
    
    # Cluster analysis
    labels, kmeans = analyzer.analyze_drug_clusters(X_scaled, drug_names)
    
    # Visualize
    analyzer.visualize_drug_space(X_scaled, drug_names, labels, feature_cols)
    
    # Toxicity analysis
    toxicity_ranking = analyzer.identify_toxicity_patterns(X_scaled, drug_names, feature_cols, features_df)
    
    # Save results
    results = {
        'features_df': features_df,
        'embeddings': X_scaled,
        'drug_names': drug_names,
        'cluster_labels': labels,
        'toxicity_ranking': toxicity_ranking,
        'feature_cols': feature_cols
    }
    
    joblib.dump(results, 'fast_drug_analysis_results.joblib')
    logger.info("\n💾 Saved all results to: fast_drug_analysis_results.joblib")
    
    analyzer.conn.close()
    
    logger.info("\n✅ Analysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()