"""Main experiment runner for organoid embedding experiments."""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.loaders.supabase_loader import SupabaseDataLoader
from embeddings import (
    BaseEmbedder,
    DTWEmbedder, FourierEmbedder, SAXEmbedder,
    TSFreshEmbedder, Catch22Embedder, CustomFeaturesEmbedder,
    AutoencoderEmbedder, TransformerEmbedder, TripletNetworkEmbedder
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner for embedding comparisons."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.start_time = time.time()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize data loader
        self.data_loader = SupabaseDataLoader()
        
        # Results storage
        self.results = {}
        self.embeddings = {}
        self.metadata = {}
        
    def setup_directories(self):
        """Create necessary directories for results."""
        self.results_dir = Path(self.config.experiment.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.experiment.name
        self.exp_dir = self.results_dir / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        (self.exp_dir / "embeddings").mkdir(exist_ok=True)
        (self.exp_dir / "models").mkdir(exist_ok=True)
        (self.exp_dir / "plots").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Experiment directory: {self.exp_dir}")
        
    def load_data(self) -> tuple:
        """Load and prepare experimental data."""
        logger.info("Loading experimental data...")
        
        data_config = self.config.data
        
        # Load time series data
        if data_config.get('use_full_experiment', True):
            df = self.data_loader.get_full_experiment_data(
                limit=data_config.get('limit', None),
                time_filter=data_config.get('time_filter', None)
            )
        else:
            df = self.data_loader.get_time_series_data(
                limit=data_config.get('limit', None),
                well_filter=data_config.get('well_filter', None),
                time_filter=data_config.get('time_filter', None)
            )
        
        # Prepare time series matrix
        X, metadata_df = self.data_loader.prepare_time_series_matrix(
            df,
            fill_method=data_config.get('fill_method', 'interpolate'),
            normalize=data_config.get('normalize', True),
            remove_outliers=data_config.get('remove_outliers', False)
        )
        
        logger.info(f"Loaded {X.shape[0]} time series of length {X.shape[1]}")
        
        # Store metadata
        self.metadata['data_shape'] = X.shape
        self.metadata['metadata_df'] = metadata_df
        
        # Extract labels if available
        y = None
        if 'target_column' in data_config and data_config.target_column in metadata_df.columns:
            y = metadata_df[data_config.target_column].values
            logger.info(f"Using target column: {data_config.target_column}")
        
        # Extract replicate information for triplet networks
        replicate_info = None
        if all(col in metadata_df.columns for col in ['drug_name', 'concentration', 'replicate']):
            replicate_info = metadata_df[['drug_name', 'concentration', 'replicate']].values
            logger.info("Replicate information available for triplet training")
        
        return X, y, replicate_info, metadata_df
    
    def create_embedder(self, method_name: str, method_config: DictConfig) -> BaseEmbedder:
        """Create embedder instance from configuration."""
        logger.info(f"Creating embedder: {method_name}")
        
        # Convert DictConfig to dict for **kwargs
        kwargs = OmegaConf.to_container(method_config, resolve=True) or {}
        
        # Remove the 'name' key if it exists
        kwargs.pop('name', None)
        
        embedder_map = {
            'dtw': DTWEmbedder,
            'fourier': FourierEmbedder,
            'sax': SAXEmbedder,
            'tsfresh': TSFreshEmbedder,
            'catch22': Catch22Embedder,
            'custom_features': CustomFeaturesEmbedder,
            'autoencoder': AutoencoderEmbedder,
            'transformer': TransformerEmbedder,
            'triplet_network': TripletNetworkEmbedder
        }
        
        if method_name not in embedder_map:
            raise ValueError(f"Unknown embedding method: {method_name}")
        
        embedder_class = embedder_map[method_name]
        return embedder_class(**kwargs)
    
    def run_embedding_method(self, method_name: str, method_config: DictConfig,
                           X: np.ndarray, y: Optional[np.ndarray] = None,
                           replicate_info: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run a single embedding method."""
        logger.info(f"Running embedding method: {method_name}")
        
        start_time = time.time()
        
        try:
            # Create embedder
            embedder = self.create_embedder(method_name, method_config)
            
            # Fit and transform
            if method_name == 'triplet_network' and replicate_info is not None:
                # Special handling for triplet networks
                embedder.fit(X, y)
                # Train with replicate info
                history = embedder.train_model(
                    X, y, 
                    replicate_info=replicate_info,
                    **method_config.get('training', {})
                )
                embeddings = embedder.transform(X)
            else:
                embeddings = embedder.fit_transform(X, y)
                history = getattr(embedder, 'training_history', None)
            
            # Calculate embedding time
            embedding_time = time.time() - start_time
            
            # Save embedder
            model_path = self.exp_dir / "models" / f"{method_name}_embedder.joblib"
            joblib.dump(embedder, model_path)
            
            # Save embeddings
            embedding_path = self.exp_dir / "embeddings" / f"{method_name}_embeddings.npy"
            np.save(embedding_path, embeddings)
            
            # Collect results
            result = {
                'embedder': embedder,
                'embeddings': embeddings,
                'embedding_time': embedding_time,
                'config': method_config,
                'training_history': history,
                'model_path': str(model_path),
                'embedding_path': str(embedding_path)
            }
            
            logger.info(f"Completed {method_name} in {embedding_time:.2f}s, "
                       f"embedding shape: {embeddings.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run {method_name}: {str(e)}")
            return {
                'error': str(e),
                'embedding_time': time.time() - start_time
            }
    
    def run_all_methods(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       replicate_info: Optional[np.ndarray] = None):
        """Run all configured embedding methods."""
        logger.info("Running all embedding methods...")
        
        for method_name, method_config in self.config.methods.items():
            if method_config.get('enabled', True):
                result = self.run_embedding_method(
                    method_name, method_config, X, y, replicate_info
                )
                self.results[method_name] = result
            else:
                logger.info(f"Skipping disabled method: {method_name}")
    
    def evaluate_embeddings(self):
        """Evaluate quality of embeddings (placeholder for now)."""
        logger.info("Evaluating embeddings...")
        
        # This would include metrics like:
        # - Silhouette score
        # - Clustering quality
        # - Visualization quality (t-SNE, UMAP)
        # - Downstream task performance
        
        # For now, just log basic statistics
        for method_name, result in self.results.items():
            if 'embeddings' in result:
                embeddings = result['embeddings']
                logger.info(f"{method_name}: shape={embeddings.shape}, "
                           f"mean={np.mean(embeddings):.4f}, "
                           f"std={np.std(embeddings):.4f}")
    
    def save_results(self):
        """Save experiment results and metadata."""
        logger.info("Saving experiment results...")
        
        # Save configuration
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(self.config, f)
        
        # Save metadata
        metadata_path = self.exp_dir / "metadata.joblib"
        joblib.dump(self.metadata, metadata_path)
        
        # Save summary results
        summary = {}
        for method_name, result in self.results.items():
            summary[method_name] = {
                'success': 'error' not in result,
                'embedding_time': result.get('embedding_time', 0),
                'embedding_shape': result['embeddings'].shape if 'embeddings' in result else None,
                'error': result.get('error', None)
            }
        
        summary_path = self.exp_dir / "summary.joblib"
        joblib.dump(summary, summary_path)
        
        # Save results (without large objects)
        results_light = {}
        for method_name, result in self.results.items():
            results_light[method_name] = {
                k: v for k, v in result.items() 
                if k not in ['embedder', 'embeddings']  # Exclude large objects
            }
        
        results_path = self.exp_dir / "results.joblib"
        joblib.dump(results_light, results_path)
        
        logger.info(f"Results saved to: {self.exp_dir}")
    
    def run(self):
        """Run the complete experiment."""
        logger.info(f"Starting experiment: {self.config.experiment.name}")
        
        try:
            # Load data
            X, y, replicate_info, metadata_df = self.load_data()
            
            # Run embedding methods
            self.run_all_methods(X, y, replicate_info)
            
            # Evaluate results
            self.evaluate_embeddings()
            
            # Save everything
            self.save_results()
            
            total_time = time.time() - self.start_time
            logger.info(f"Experiment completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for experiments."""
    
    # Print configuration
    logger.info("Experiment Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Run experiment
    runner = ExperimentRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()