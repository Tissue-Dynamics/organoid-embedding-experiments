#!/usr/bin/env python3
"""Hierarchical clustering: wells -> concentrations -> drugs with proper exclusions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import duckdb
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from embeddings import (
    FourierEmbedder, SAXEmbedder, CustomFeaturesEmbedder,
    Catch22Embedder
)

# TSFresh import with error handling
try:
    from embeddings import TSFreshEmbedder
except:
    TSFreshEmbedder = None
    print("TSFresh embedder not available")

class HierarchicalClusterOxygenVisualization:
    """Proper hierarchical clustering: wells -> concentrations -> drugs."""
    
    def __init__(self):
        self.conn = self._connect_db()
        
    def _connect_db(self):
        """Connect to database."""
        load_dotenv()
        database_url = os.getenv('DATABASE_URL')
        
        conn = duckdb.connect()
        conn.execute("INSTALL postgres;")
        conn.execute("LOAD postgres;")
        
        parsed = urlparse(database_url)
        attach_query = f"""
        ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
        user={parsed.username} password={parsed.password}' 
        AS supabase (TYPE POSTGRES, READ_ONLY);
        """
        
        conn.execute(attach_query)
        return conn
    
    def get_hierarchical_data_structure(self):
        """Get properly structured data: wells -> concentrations -> drugs."""
        
        print("🔍 Loading hierarchical data structure...")
        
        # Get all non-excluded data with proper hierarchy
        query = """
        WITH drug_structure AS (
            SELECT 
                w.drug,
                w.concentration,
                w.well_number,
                w.plate_id,
                p.timestamp,
                p.median_o2,
                w.is_excluded as well_excluded,
                p.is_excluded as measurement_excluded,
                -- Standardize time within each drug to hours from start
                EXTRACT(EPOCH FROM (p.timestamp - MIN(p.timestamp) OVER (PARTITION BY w.drug))) / 3600.0 as hours_from_start
            FROM supabase.public.well_map_data w
            JOIN supabase.public.processed_data p
                ON w.plate_id = p.plate_id AND w.well_number = p.well_number
            WHERE w.drug != '' 
            AND w.drug IS NOT NULL
            AND w.is_excluded = false  -- Only non-excluded wells
            AND p.is_excluded = false  -- Only non-excluded measurements
            AND w.concentration >= 0   -- Include control (concentration = 0)
        ),
        drug_summary AS (
            SELECT 
                drug,
                COUNT(DISTINCT concentration) as n_concentrations,
                COUNT(DISTINCT well_number) as n_wells,
                COUNT(DISTINCT CASE WHEN concentration > 0 THEN concentration END) as n_drug_concentrations,
                COUNT(DISTINCT CASE WHEN concentration = 0 THEN well_number END) as n_control_wells,
                MIN(hours_from_start) as min_hours,
                MAX(hours_from_start) as max_hours,
                COUNT(*) as total_measurements
            FROM drug_structure
            GROUP BY drug
        )
        SELECT 
            ds.*,
            summary.n_concentrations,
            summary.n_drug_concentrations,
            summary.n_control_wells,
            summary.max_hours - summary.min_hours as duration_hours
        FROM drug_structure ds
        JOIN drug_summary summary ON ds.drug = summary.drug
        WHERE summary.n_drug_concentrations >= 4  -- At least 4 non-control concentrations
        AND summary.max_hours - summary.min_hours >= 200  -- At least ~8 days
        AND summary.total_measurements >= 500
        ORDER BY ds.drug, ds.concentration, ds.well_number, ds.hours_from_start
        """
        
        df = self.conn.execute(query).df()
        
        # Get drug summary
        drug_summary = df.groupby('drug').agg({
            'n_concentrations': 'first',
            'n_drug_concentrations': 'first', 
            'n_control_wells': 'first',
            'duration_hours': 'first'
        }).reset_index()
        
        # Add total measurements count
        drug_summary['total_measurements'] = df.groupby('drug').size().values
        
        print(f"✅ Loaded data for {len(drug_summary)} qualifying drugs:")
        print(f"   - Total measurements: {len(df):,}")
        print(f"   - Average concentrations per drug: {drug_summary['n_concentrations'].mean():.1f}")
        print(f"   - Average duration: {drug_summary['duration_hours'].mean():.0f} hours")
        
        print("\nTop 10 drugs by data volume:")
        top_drugs = drug_summary.nlargest(10, 'total_measurements')
        print(top_drugs[['drug', 'n_concentrations', 'n_drug_concentrations', 'total_measurements']])
        
        return df, drug_summary
    
    def create_well_time_series(self, df, max_hours=300):
        """Create time series for each individual well."""
        
        print(f"\n📊 Creating individual well time series (max {max_hours} hours)...")
        
        well_data = []
        
        # Group by drug, concentration, well to create individual time series
        for (drug, conc, well, plate), group in df.groupby(['drug', 'concentration', 'well_number', 'plate_id']):
            group = group.sort_values('hours_from_start')
            
            if len(group) >= 50:  # Need sufficient data points
                # Create hourly binned time series
                hours = group['hours_from_start'].values
                o2_values = group['median_o2'].values
                
                # Bin into hours
                time_series = np.full(max_hours, np.nan)
                for i, (hour, o2) in enumerate(zip(hours, o2_values)):
                    hour_idx = int(np.round(hour))
                    if 0 <= hour_idx < max_hours:
                        if np.isnan(time_series[hour_idx]):
                            time_series[hour_idx] = o2
                        else:
                            time_series[hour_idx] = (time_series[hour_idx] + o2) / 2  # Average if multiple
                
                # Interpolate missing values
                series = pd.Series(time_series)
                series = series.interpolate(method='linear', limit_direction='both')
                
                if series.notna().sum() >= 100:  # Need at least 100 valid points
                    well_data.append({
                        'drug': drug,
                        'concentration': conc,
                        'well_number': well,
                        'plate_id': plate,
                        'is_control': conc == 0,
                        'time_series': series.values,
                        'n_measurements': len(group)
                    })
        
        well_df = pd.DataFrame(well_data)
        print(f"✅ Created {len(well_df)} well time series")
        
        # Show structure
        structure = well_df.groupby('drug').agg({
            'concentration': 'nunique',
            'well_number': 'nunique', 
            'is_control': 'sum'
        }).rename(columns={'concentration': 'n_conc', 'well_number': 'n_wells', 'is_control': 'n_controls'})
        
        print("Drug structure (top 10):")
        print(structure.head(10))
        
        return well_df
    
    def hierarchical_embedding_pipeline(self, well_df):
        """Create embeddings following hierarchy: wells -> concentrations -> drugs."""
        
        print("\n🧬 Running hierarchical embedding pipeline...")
        
        # Step 1: Create embeddings for ALL wells
        print("  Step 1: Creating well-level embeddings...")
        well_time_series = np.array([w for w in well_df['time_series']])
        print(f"    Well time series matrix: {well_time_series.shape}")
        
        # Create multiple embedding types for wells
        well_embeddings = {}
        
        # Fourier embeddings
        print("    Computing Fourier embeddings...")
        fourier = FourierEmbedder(n_components=20)
        well_embeddings['fourier'] = fourier.fit_transform(well_time_series)
        
        # SAX embeddings  
        print("    Computing SAX embeddings...")
        sax = SAXEmbedder(n_segments=15, n_symbols=6)
        well_embeddings['sax'] = sax.fit_transform(well_time_series)
        
        # Catch22 embeddings (robust feature-based)
        print("    Computing Catch22 embeddings...")
        try:
            catch22 = Catch22Embedder()
            well_embeddings['catch22'] = catch22.fit_transform(well_time_series)
        except Exception as e:
            print(f"    Catch22 failed: {e}")
            
        # TSFresh embeddings (comprehensive features)
        print("    Computing TSFresh embeddings...")
        try:
            # Downgrade numpy temporarily for TSFresh compatibility
            import warnings
            warnings.filterwarnings('ignore')
            
            # TSFresh on smaller subset due to computational cost and numpy issues
            subset_size = min(500, len(well_time_series))  # Smaller subset
            np.random.seed(42)  # Reproducible
            subset_indices = np.random.choice(len(well_time_series), subset_size, replace=False)
            subset_data = well_time_series[subset_indices]
            
            # Convert to TSFresh format
            tsfresh_data = []
            for i, series in enumerate(subset_data):
                # Sample every 5th point to reduce computation
                for t in range(0, len(series), 5):
                    value = series[t]
                    if not np.isnan(value) and np.isfinite(value):
                        tsfresh_data.append({
                            'id': i,
                            'time': t, 
                            'value': float(value)
                        })
            
            if len(tsfresh_data) > 1000:  # Need sufficient data
                tsfresh_df = pd.DataFrame(tsfresh_data)
                
                # Import with error handling for numpy compatibility
                try:
                    from tsfresh import extract_features
                    from tsfresh.utilities.dataframe_functions import impute
                    from tsfresh.feature_extraction import EfficientFCParameters
                    
                    # Use efficient feature set
                    features = extract_features(tsfresh_df, 
                                               column_id='id', 
                                               column_sort='time',
                                               default_fc_parameters=EfficientFCParameters(),
                                               disable_progressbar=True,
                                               n_jobs=1)  # Single thread to avoid issues
                    
                    # Handle NaN/inf values
                    features = features.replace([np.inf, -np.inf], np.nan)
                    features = impute(features)
                    
                    if len(features) > 0 and features.shape[1] > 0:
                        # Remove constant columns
                        feature_std = features.std()
                        valid_features = features.loc[:, feature_std > 1e-10]
                        
                        if len(valid_features.columns) > 0:
                            # Create full matrix with zeros for non-subset samples
                            full_tsfresh = np.zeros((len(well_time_series), valid_features.shape[1]))
                            full_tsfresh[subset_indices] = valid_features.values
                            well_embeddings['tsfresh'] = full_tsfresh
                            print(f"    TSFresh completed with {valid_features.shape[1]} features")
                        else:
                            print("    TSFresh: All features were constant")
                    else:
                        print("    TSFresh returned empty features")
                        
                except ImportError as ie:
                    print(f"    TSFresh import failed: {ie}")
                except Exception as tse:
                    print(f"    TSFresh extraction failed: {tse}")
            else:
                print(f"    TSFresh: Insufficient data points ({len(tsfresh_data)})")
                
        except Exception as e:
            print(f"    TSFresh failed: {e}")
        
        # Custom features
        features = []
        for series in well_time_series:
            valid_series = series[~np.isnan(series)]
            
            # Calculate features with NaN handling
            mean_val = np.mean(valid_series) if len(valid_series) > 0 else 0
            std_val = np.std(valid_series) if len(valid_series) > 1 else 0
            min_val = np.min(valid_series) if len(valid_series) > 0 else 0
            max_val = np.max(valid_series) if len(valid_series) > 0 else 0
            range_val = max_val - min_val
            
            early_val = np.mean(valid_series[:len(valid_series)//4]) if len(valid_series) >= 4 else mean_val
            late_val = np.mean(valid_series[-len(valid_series)//4:]) if len(valid_series) >= 4 else mean_val
            
            # Handle correlation calculation with NaN check
            if len(valid_series) > 1:
                try:
                    corr_matrix = np.corrcoef(np.arange(len(valid_series)), valid_series)
                    trend_val = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
                except:
                    trend_val = 0
            else:
                trend_val = 0
                
            variability_val = np.mean(np.abs(np.diff(valid_series))) if len(valid_series) > 1 else 0
            
            feat = [mean_val, std_val, min_val, max_val, range_val, early_val, late_val, trend_val, variability_val]
            features.append(feat)
        
        # Handle any remaining NaNs in features
        features_array = np.array(features)
        features_clean = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        well_embeddings['custom'] = features_scaled
        
        # Step 2: Aggregate wells to concentration level
        print("  Step 2: Aggregating to concentration level...")
        concentration_embeddings = {}
        concentration_metadata = []
        
        for method, well_emb in well_embeddings.items():
            conc_emb = []
            
            # Group wells by drug and concentration
            for (drug, conc), group_idx in well_df.groupby(['drug', 'concentration']).groups.items():
                if len(group_idx) >= 2:  # Need at least 2 wells per concentration
                    # Average embeddings across wells for this concentration
                    well_embeddings_for_conc = well_emb[group_idx]
                    avg_embedding = np.mean(well_embeddings_for_conc, axis=0)
                    conc_emb.append(avg_embedding)
                    
                    if method == 'fourier':  # Only add metadata once
                        concentration_metadata.append({
                            'drug': drug,
                            'concentration': conc,
                            'is_control': conc == 0,
                            'n_wells': len(group_idx)
                        })
            
            concentration_embeddings[method] = np.array(conc_emb)
        
        conc_df = pd.DataFrame(concentration_metadata)
        print(f"    Concentration embeddings: {conc_df.shape[0]} concentrations")
        
        # Step 3: Aggregate concentrations to drug level 
        print("  Step 3: Aggregating to drug level...")
        drug_embeddings = {}
        drug_metadata = []
        
        for method, conc_emb in concentration_embeddings.items():
            drug_emb = []
            
            # Group concentrations by drug (excluding controls)
            drug_groups = conc_df[conc_df['concentration'] > 0].groupby('drug')
            
            for drug, group_idx in drug_groups.groups.items():
                if len(group_idx) >= 3:  # Need at least 3 concentrations
                    # Average embeddings across concentrations for this drug
                    conc_embeddings_for_drug = conc_emb[group_idx]
                    avg_embedding = np.mean(conc_embeddings_for_drug, axis=0)
                    drug_emb.append(avg_embedding)
                    
                    if method == 'fourier':  # Only add metadata once
                        drug_info = conc_df[conc_df['drug'] == drug]
                        drug_metadata.append({
                            'drug': drug,
                            'n_concentrations': len(group_idx),
                            'total_wells': drug_info['n_wells'].sum()
                        })
            
            drug_embeddings[method] = np.array(drug_emb)
        
        drug_df = pd.DataFrame(drug_metadata)
        print(f"    Drug embeddings: {drug_df.shape[0]} drugs")
        
        return {
            'well_embeddings': well_embeddings,
            'well_metadata': well_df,
            'concentration_embeddings': concentration_embeddings, 
            'concentration_metadata': conc_df,
            'drug_embeddings': drug_embeddings,
            'drug_metadata': drug_df
        }
    
    def create_hierarchical_visualizations(self, results, n_clusters=6):
        """Create visualizations showing clustering at each level."""
        
        print(f"\n🎨 Creating hierarchical visualizations...")
        
        # Focus on drug-level clustering with concentration examples
        drug_embeddings = results['drug_embeddings']
        drug_metadata = results['drug_metadata']
        concentration_metadata = results['concentration_metadata']
        well_metadata = results['well_metadata']
        
        for method_name, embedding in drug_embeddings.items():
            print(f"\n🔍 Processing {method_name} ({embedding.shape[0]} drugs)...")
            
            # Standardize embeddings for better clustering
            scaler = StandardScaler()
            embedding_scaled = scaler.fit_transform(embedding)
            
            # Cluster drugs
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            drug_labels = kmeans.fit_predict(embedding_scaled)
            
            # t-SNE for visualization (instead of PCA)
            print(f"    Computing t-SNE for {method_name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embedding)//4))
            embedding_2d = tsne.fit_transform(embedding_scaled)
            
            # Create figure
            fig = plt.figure(figsize=(24, 8 * n_clusters))
            
            # Top plot: Drug-level t-SNE
            ax_tsne = plt.subplot(n_clusters + 1, 1, 1)
            
            scatter = ax_tsne.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                     c=drug_labels, cmap='tab10', s=120, alpha=0.8, 
                                     edgecolors='black', linewidth=1)
            
            # Label drugs (only a subset to avoid overcrowding)
            for i, drug in enumerate(drug_metadata['drug']):
                if i % 5 == 0:  # Show every 5th drug label to avoid overcrowding
                    ax_tsne.annotate(drug[:8], (embedding_2d[i, 0], embedding_2d[i, 1]), 
                                    fontsize=8, alpha=0.7, ha='center')
            
            ax_tsne.set_xlabel('t-SNE 1', fontsize=14)
            ax_tsne.set_ylabel('t-SNE 2', fontsize=14)
            ax_tsne.set_title(f'{method_name.upper()} Drug-Level Clustering\n'
                             f'{len(drug_metadata)} drugs (hierarchical: wells → concentrations → drugs)', 
                             fontsize=18, fontweight='bold')
            ax_tsne.grid(True, alpha=0.3)
            
            # For each cluster, show concentration-level oxygen patterns
            for cluster_id in range(n_clusters):
                ax = plt.subplot(n_clusters + 1, 1, cluster_id + 2)
                
                # Get drugs in this cluster
                cluster_mask = drug_labels == cluster_id
                cluster_drugs = drug_metadata[cluster_mask]['drug'].tolist()
                
                if len(cluster_drugs) == 0:
                    ax.text(0.5, 0.5, f'Cluster {cluster_id}: No drugs', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=14)
                    continue
                
                # Get all concentration curves for drugs in this cluster
                # Collect all concentration values across all drugs in cluster
                all_concentrations = set()
                for drug in cluster_drugs:
                    drug_concs = concentration_metadata[
                        (concentration_metadata['drug'] == drug) & 
                        (concentration_metadata['concentration'] > 0)
                    ]['concentration'].values
                    all_concentrations.update(drug_concs)
                
                # Sort concentrations and assign colors
                sorted_concentrations = sorted(list(all_concentrations))
                conc_colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_concentrations)))
                conc_color_map = {conc: conc_colors[i] for i, conc in enumerate(sorted_concentrations)}
                
                # Plot curves colored by concentration
                plotted_concentrations = set()
                
                for drug in cluster_drugs[:12]:  # Max 12 drugs per cluster
                    # Get all concentrations for this drug
                    drug_concentrations = concentration_metadata[
                        (concentration_metadata['drug'] == drug) & 
                        (concentration_metadata['concentration'] > 0)
                    ].sort_values('concentration')
                    
                    # For each concentration, get the well time series and average them
                    for _, conc_row in drug_concentrations.iterrows():
                        conc = conc_row['concentration']
                        
                        # Get wells for this drug/concentration
                        wells_for_conc = well_metadata[
                            (well_metadata['drug'] == drug) & 
                            (well_metadata['concentration'] == conc)
                        ]
                        
                        if len(wells_for_conc) > 0:
                            # Average time series across wells
                            well_series = [w for w in wells_for_conc['time_series']]
                            avg_series = np.nanmean(well_series, axis=0)
                            time_hours = np.arange(len(avg_series))
                            
                            # Plot with color based on concentration
                            color = conc_color_map[conc]
                            alpha = 0.7
                            linewidth = 1.5
                            
                            # Only add label for first occurrence of each concentration
                            if conc not in plotted_concentrations:
                                label = f'{conc:.2f}µM' if conc < 1 else f'{conc:.1f}µM'
                                plotted_concentrations.add(conc)
                            else:
                                label = None
                            
                            ax.plot(time_hours, avg_series, 
                                   color=color, alpha=alpha, linewidth=linewidth,
                                   label=label)
                
                ax.set_xlabel('Time (hours)', fontsize=12)
                ax.set_ylabel('O2 (%)', fontsize=12)
                ax.set_title(f'Cluster {cluster_id} - Concentration Response Curves ({len(cluster_drugs)} drugs)\n'
                           f'Drugs: {", ".join([d[:10] for d in cluster_drugs[:5]])}{"..." if len(cluster_drugs) > 5 else ""}', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                # Set appropriate axis limits to show full data range
                ax.set_xlim(-10, 350)  # Full time range (2+ weeks = ~336 hours)
                ax.set_ylim(-10, 100)  # Extended oxygen range as requested
            
            plt.suptitle(f'{method_name.upper()} Hierarchical Embedding Analysis\n'
                        f'Proper hierarchy: {len(well_metadata)} wells → {len(concentration_metadata)} concentrations → {len(drug_metadata)} drugs\n'
                        f'Each cluster shows drugs with similar dose-response patterns',
                        fontsize=20, fontweight='bold')
            plt.tight_layout()
            
            # Save
            output_path = f'results/figures/embedding_comparisons/{method_name.lower()}_hierarchical_clusters.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved {method_name} hierarchical visualization to: {output_path}")
            plt.close()
            
            # Print cluster details
            print(f"   Cluster sizes: {[sum(drug_labels == i) for i in range(n_clusters)]}")
            for i in range(n_clusters):
                cluster_drugs = drug_metadata[drug_labels == i]['drug'].tolist()
                print(f"   Cluster {i}: {cluster_drugs[:5]}{'...' if len(cluster_drugs) > 5 else ''}")
    
    def run_hierarchical_analysis(self):
        """Run complete hierarchical analysis."""
        
        # Load hierarchical data
        df, drug_summary = self.get_hierarchical_data_structure()
        
        # Create well time series
        well_df = self.create_well_time_series(df)
        
        # Run hierarchical embedding pipeline
        results = self.hierarchical_embedding_pipeline(well_df)
        
        # Create visualizations
        self.create_hierarchical_visualizations(results, n_clusters=6)
        
        # Save results
        output_path = 'results/data/hierarchical_embedding_results.joblib'
        joblib.dump(results, output_path)
        
        print(f"\n✅ Hierarchical analysis complete!")
        print(f"   📊 Processed {len(results['well_metadata'])} wells → {len(results['concentration_metadata'])} concentrations → {len(results['drug_metadata'])} drugs")
        print(f"   💾 Saved results to: {output_path}")
        print(f"   🖼️  Check results/figures/embedding_comparisons/ for hierarchical visualizations")


if __name__ == "__main__":
    analyzer = HierarchicalClusterOxygenVisualization()
    analyzer.run_hierarchical_analysis()