#!/usr/bin/env python3
"""Quick analysis of real oxygen data with fast embedding methods."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_load_oxygen_data():
    """Quick load of oxygen data for one plate."""
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
    
    # Get plate with most data
    plate_query = """
    SELECT plate_id, COUNT(*) as count
    FROM supabase.public.processed_data
    WHERE is_excluded = false
    GROUP BY plate_id
    ORDER BY count DESC
    LIMIT 1
    """
    
    best_plate = conn.execute(plate_query).fetchone()
    plate_id = best_plate[0]
    count = best_plate[1]
    
    logger.info(f"📊 Selected plate {plate_id} with {count:,} measurements")
    
    # Load first 24 wells for quick analysis
    data_query = f"""
    SELECT 
        well_number,
        timestamp,
        median_o2
    FROM supabase.public.processed_data
    WHERE plate_id = '{plate_id}'
    AND well_number <= 24
    AND is_excluded = false
    ORDER BY well_number, timestamp
    """
    
    df = conn.execute(data_query).df()
    
    # Also get treatment info if available
    treatment_query = f"""
    SELECT DISTINCT
        well_number,
        plate_id
    FROM supabase.public.processed_data
    WHERE plate_id = '{plate_id}'
    AND well_number <= 24
    """
    
    treatments = conn.execute(treatment_query).df()
    
    conn.close()
    
    logger.info(f"✅ Loaded {len(df):,} oxygen measurements for {df['well_number'].nunique()} wells")
    
    return df, plate_id

def prepare_quick_matrix(df):
    """Quick preparation of time series matrix."""
    # Pivot to matrix format
    pivot_df = df.pivot_table(
        index='well_number',
        columns='timestamp',
        values='median_o2',
        aggfunc='mean'
    )
    
    # Fill missing values with forward fill then backward fill
    pivot_df = pivot_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    
    X = pivot_df.values
    wells = pivot_df.index.values
    
    logger.info(f"✅ Matrix shape: {X.shape} (wells × time points)")
    
    return X, wells

def visualize_oxygen_patterns(X, wells):
    """Visualize oxygen consumption patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sample time series
    ax = axes[0, 0]
    for i in range(min(5, len(X))):
        ax.plot(X[i], alpha=0.7, label=f'Well {wells[i]}')
    ax.set_title('Sample Oxygen Time Series')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Median O2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Heatmap of all wells
    ax = axes[0, 1]
    # Downsample for visualization
    step = max(1, X.shape[1] // 100)
    im = ax.imshow(X[:, ::step], aspect='auto', cmap='RdBu_r')
    ax.set_title('Oxygen Levels Heatmap')
    ax.set_xlabel('Time (downsampled)')
    ax.set_ylabel('Well')
    plt.colorbar(im, ax=ax, label='Median O2')
    
    # 3. Distribution of oxygen values
    ax = axes[1, 0]
    ax.hist(X.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Oxygen Values')
    ax.set_xlabel('Median O2')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # 4. Mean trajectory with std
    ax = axes[1, 1]
    mean_trajectory = X.mean(axis=0)
    std_trajectory = X.std(axis=0)
    time_points = np.arange(len(mean_trajectory))
    
    ax.plot(time_points[::step], mean_trajectory[::step], 'b-', label='Mean')
    ax.fill_between(time_points[::step], 
                    (mean_trajectory - std_trajectory)[::step],
                    (mean_trajectory + std_trajectory)[::step],
                    alpha=0.3, color='blue', label='±1 STD')
    ax.set_title('Average Oxygen Trajectory')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Median O2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_oxygen_data_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Saved visualization to: real_oxygen_data_analysis.png")
    plt.close()

def run_fast_embeddings(X):
    """Run only fast embedding methods."""
    from embeddings import FourierEmbedder, SAXEmbedder, CustomFeaturesEmbedder
    
    methods = {
        'fourier': FourierEmbedder(n_components=30),
        'sax': SAXEmbedder(n_segments=15, n_symbols=4),
        'custom': CustomFeaturesEmbedder()
    }
    
    results = {}
    
    for name, embedder in methods.items():
        logger.info(f"\n🚀 Running {name} embedder...")
        try:
            import time
            start = time.time()
            embeddings = embedder.fit_transform(X)
            duration = time.time() - start
            
            logger.info(f"✅ {name}: {embeddings.shape} in {duration:.2f}s")
            results[name] = embeddings
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
    
    return results

def visualize_embeddings(results):
    """Visualize embedding results."""
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (name, embeddings) in enumerate(results.items()):
        ax = axes[i]
        
        # Use PCA for 2D visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=range(len(embeddings)), cmap='viridis', alpha=0.7)
        ax.set_title(f'{name.capitalize()} Embeddings')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax, label='Well Index')
    
    plt.tight_layout()
    plt.savefig('oxygen_embeddings_visualization.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Saved embeddings visualization to: oxygen_embeddings_visualization.png")
    plt.close()

def analyze_embedding_quality(results, X):
    """Quick quality analysis of embeddings."""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    logger.info("\n📈 Embedding Quality Analysis:")
    
    for name, embeddings in results.items():
        # Try clustering with k=3 (e.g., control, low toxicity, high toxicity)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Silhouette score
        sil_score = silhouette_score(embeddings, labels)
        
        logger.info(f"\n{name.capitalize()}:")
        logger.info(f"  Embedding dimensions: {embeddings.shape[1]}")
        logger.info(f"  Silhouette score (k=3): {sil_score:.3f}")
        logger.info(f"  Cluster sizes: {np.bincount(labels)}")

if __name__ == "__main__":
    logger.info("🧬 Quick analysis of real oxygen data...")
    
    try:
        # Load real oxygen data
        df, plate_id = quick_load_oxygen_data()
        
        # Prepare matrix
        X, wells = prepare_quick_matrix(df)
        
        # Visualize patterns
        visualize_oxygen_patterns(X, wells)
        
        # Run fast embeddings
        results = run_fast_embeddings(X)
        
        # Visualize embeddings
        if results:
            visualize_embeddings(results)
            analyze_embedding_quality(results, X)
        
        logger.info("\n✅ SUCCESS! Analysis complete.")
        logger.info(f"📊 Analyzed {len(X)} organoid wells with {X.shape[1]} time points each")
        logger.info(f"📊 Generated {len(results)} different embeddings")
        logger.info("📊 Check the generated PNG files for visualizations")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise