#!/usr/bin/env python3
"""Quick test with synthetic data to validate implementation."""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_organoid_data(n_samples=100, n_timepoints=336, n_treatments=8):
    """
    Generate synthetic organoid time series data.
    
    Args:
        n_samples: Number of time series
        n_timepoints: Length of each time series (336 = 2 weeks hourly)
        n_treatments: Number of different treatments
    
    Returns:
        X: Time series data (n_samples, n_timepoints)
        y: Treatment labels
        metadata: Additional metadata
    """
    np.random.seed(42)
    
    # Time points (hours)
    time = np.arange(n_timepoints)
    
    X = np.zeros((n_samples, n_timepoints))
    y = np.zeros(n_samples, dtype=int)
    metadata = []
    
    for i in range(n_samples):
        # Assign treatment
        treatment = i % n_treatments
        y[i] = treatment
        
        # Base oxygen consumption pattern
        # Healthy baseline around 80% oxygen
        baseline = 80 + np.random.normal(0, 5)
        
        # Add circadian rhythm (24-hour cycle)
        circadian = 10 * np.sin(2 * np.pi * time / 24 + np.random.uniform(0, 2*np.pi))
        
        # Add weekly rhythm (slower metabolism)
        weekly = 5 * np.sin(2 * np.pi * time / (24 * 7) + np.random.uniform(0, 2*np.pi))
        
        # Treatment effects
        if treatment == 0:  # Control
            treatment_effect = 0
        elif treatment <= 3:  # Mild toxicity
            # Gradual decline starting after 2 days
            treatment_effect = -np.maximum(0, (time - 48) / 10) * np.random.uniform(0.5, 1.5)
        elif treatment <= 6:  # Moderate toxicity
            # Faster decline
            treatment_effect = -np.maximum(0, (time - 24) / 5) * np.random.uniform(1.0, 2.0)
        else:  # Severe toxicity
            # Rapid decline
            treatment_effect = -np.maximum(0, (time - 12) / 3) * np.random.uniform(2.0, 3.0)
        
        # Add noise
        noise = np.random.normal(0, 2, n_timepoints)
        
        # Combine all components
        ts = baseline + circadian + weekly + treatment_effect + noise
        
        # Ensure values stay in reasonable range (0-100% oxygen)
        ts = np.clip(ts, 5, 100)
        
        # Add some missing values (simulate measurement gaps)
        if np.random.random() < 0.3:  # 30% of series have gaps
            gap_start = np.random.randint(50, n_timepoints - 50)
            gap_length = np.random.randint(5, 20)
            ts[gap_start:gap_start + gap_length] = np.nan
        
        X[i] = ts
        
        metadata.append({
            'series_id': i,
            'treatment': treatment,
            'well': f'A{(i % 96) + 1:02d}',
            'plate': f'plate_{i // 96}',
            'replicate': i % 4
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    logger.info(f"Generated {n_samples} synthetic time series with {n_treatments} treatments")
    return X, y, metadata_df

def test_embedding_methods():
    """Test embedding methods with synthetic data."""
    from embeddings import (
        FourierEmbedder, SAXEmbedder, Catch22Embedder, CustomFeaturesEmbedder,
        AutoencoderEmbedder, TransformerEmbedder
    )
    
    # Generate synthetic data
    X, y, metadata = generate_synthetic_organoid_data(n_samples=50, n_timepoints=168)  # 1 week
    
    logger.info(f"Testing with data shape: {X.shape}")
    
    # Test embedding methods
    methods = {
        'fourier': FourierEmbedder(n_components=20),
        'sax': SAXEmbedder(n_segments=10, n_symbols=4),
        'catch22': Catch22Embedder(),
        'custom': CustomFeaturesEmbedder(),
        'autoencoder': AutoencoderEmbedder(embedding_dim=32, epochs=5, batch_size=16),
        'transformer': TransformerEmbedder(embedding_dim=32, epochs=5, batch_size=16)
    }
    
    results = {}
    
    for name, embedder in methods.items():
        logger.info(f"Testing {name} embedder...")
        try:
            start_time = time.time()
            embeddings = embedder.fit_transform(X, y)
            duration = time.time() - start_time
            
            logger.info(f"{name}: {embeddings.shape} in {duration:.2f}s")
            results[name] = {
                'embeddings': embeddings,
                'shape': embeddings.shape,
                'duration': duration,
                'config': embedder.get_config()
            }
            
        except Exception as e:
            logger.error(f"Failed {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results, X, y, metadata

if __name__ == "__main__":
    import time
    
    logger.info("Starting synthetic data test...")
    
    try:
        results, X, y, metadata = test_embedding_methods()
        
        logger.info("\n=== RESULTS SUMMARY ===")
        for method, result in results.items():
            if 'error' in result:
                logger.error(f"{method}: ERROR - {result['error']}")
            else:
                logger.info(f"{method}: {result['shape']} in {result['duration']:.2f}s")
        
        # Test evaluation metrics
        logger.info("\nTesting evaluation metrics...")
        from evaluation.metrics.embedding_metrics import EmbeddingEvaluator
        
        evaluator = EmbeddingEvaluator()
        
        for method, result in results.items():
            if 'embeddings' in result:
                logger.info(f"Evaluating {method}...")
                try:
                    metrics = evaluator.evaluate_embedding_quality(
                        result['embeddings'], 
                        original_data=X,
                        true_labels=y
                    )
                    
                    # Print key metrics
                    clustering = metrics.get('clustering', {})
                    if 'best_silhouette_score' in clustering:
                        logger.info(f"  Silhouette: {clustering['best_silhouette_score']:.3f}")
                    
                    neighborhood = metrics.get('neighborhood_preservation', {})
                    if 'mean_neighborhood_preservation' in neighborhood:
                        logger.info(f"  Neighborhood: {neighborhood['mean_neighborhood_preservation']:.3f}")
                        
                except Exception as e:
                    logger.error(f"  Evaluation failed: {e}")
        
        logger.info("\n✅ Synthetic test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise