"""TSFresh feature extraction embedding implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from ..base import FeatureEmbedder

logger = logging.getLogger(__name__)


class TSFreshEmbedder(FeatureEmbedder):
    """
    TSFresh (Time Series Feature extraction based on scalable hypothesis tests) embedding.
    
    Extracts a comprehensive set of time series features and optionally performs
    feature selection based on statistical significance.
    """
    
    def __init__(self, feature_set: str = 'comprehensive', 
                 feature_selection: bool = True,
                 custom_fc_parameters: Optional[Dict] = None,
                 n_jobs: int = 1, chunksize: Optional[int] = None,
                 disable_progressbar: bool = False, **kwargs):
        """
        Initialize TSFresh embedder.
        
        Args:
            feature_set: Feature set to use ('comprehensive', 'minimal', 'custom')
            feature_selection: Whether to perform feature selection
            custom_fc_parameters: Custom feature calculation parameters
            n_jobs: Number of parallel jobs
            chunksize: Chunk size for parallel processing
            disable_progressbar: Whether to disable progress bar
        """
        super().__init__(name="TSFresh", **kwargs)
        self.feature_set = feature_set
        self.feature_selection = feature_selection
        self.custom_fc_parameters = custom_fc_parameters
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.disable_progressbar = disable_progressbar
        
        # Set feature calculation parameters
        if custom_fc_parameters is not None:
            self.fc_parameters = custom_fc_parameters
        elif feature_set == 'comprehensive':
            self.fc_parameters = ComprehensiveFCParameters()
        elif feature_set == 'minimal':
            self.fc_parameters = MinimalFCParameters()
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        self.selected_features = None
        
    def _prepare_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """
        Convert time series array to TSFresh-compatible DataFrame.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            DataFrame in TSFresh format
        """
        data_list = []
        
        for sample_idx in range(len(X)):
            ts = X[sample_idx]
            
            # Handle NaN values
            valid_mask = ~np.isnan(ts)
            if np.sum(valid_mask) == 0:
                # All NaN - create minimal valid series
                data_list.append({
                    'id': sample_idx,
                    'time': 0,
                    'value': 0.0
                })
                continue
            
            # Get valid time points and values
            valid_times = np.where(valid_mask)[0]
            valid_values = ts[valid_mask]
            
            # Create DataFrame rows
            for time_idx, value in zip(valid_times, valid_values):
                data_list.append({
                    'id': sample_idx,
                    'time': time_idx,
                    'value': value
                })
        
        df = pd.DataFrame(data_list)
        return df
    
    def _prepare_target(self, y: Optional[np.ndarray], n_samples: int) -> Optional[pd.Series]:
        """
        Prepare target variable for feature selection.
        
        Args:
            y: Target labels
            n_samples: Number of samples
            
        Returns:
            Pandas Series with target values
        """
        if y is None:
            return None
        
        if len(y) != n_samples:
            raise ValueError(f"Target length ({len(y)}) doesn't match number of samples ({n_samples})")
        
        return pd.Series(y, name='target')
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract TSFresh features from time series data.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            Extracted features
        """
        logger.info(f"Extracting TSFresh features for {len(X)} time series")
        
        # Convert to TSFresh format
        df = self._prepare_dataframe(X)
        
        # Extract features
        features_df = extract_features(
            df,
            column_id='id',
            column_sort='time',
            column_value='value',
            default_fc_parameters=self.fc_parameters,
            n_jobs=self.n_jobs,
            chunksize=self.chunksize,
            disable_progressbar=self.disable_progressbar
        )
        
        # Handle missing values
        features_df = impute(features_df)
        
        # Apply feature selection if fitted
        if self.selected_features is not None:
            # Keep only selected features that exist in current extraction
            available_features = list(set(self.selected_features) & set(features_df.columns))
            if len(available_features) < len(self.selected_features):
                logger.warning(f"Only {len(available_features)}/{len(self.selected_features)} selected features available")
            features_df = features_df[available_features]
        
        # Ensure consistent ordering for samples
        features_df = features_df.sort_index()
        
        return features_df.values
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TSFreshEmbedder':
        """
        Fit TSFresh embedder with optional feature selection.
        
        Args:
            X: Training time series data
            y: Target labels for feature selection
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting TSFresh embedder on {len(X)} samples")
        
        # Extract features
        features = self.extract_features(X)
        features_df = pd.DataFrame(features)
        
        # Perform feature selection if requested and target is provided
        if self.feature_selection and y is not None:
            logger.info("Performing feature selection")
            
            target_series = self._prepare_target(y, len(X))
            
            try:
                selected_features_df = select_features(
                    features_df,
                    target_series,
                    n_jobs=self.n_jobs
                )
                
                self.selected_features = list(selected_features_df.columns)
                logger.info(f"Selected {len(self.selected_features)} features out of {features_df.shape[1]}")
                
                # Use selected features
                features = selected_features_df.values
                
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}. Using all features.")
                self.selected_features = None
        
        # Fit scaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        self.embedding_dim = features.shape[1]
        self.is_fitted = True
        
        logger.info(f"TSFresh embedder fitted with {self.embedding_dim} features")
        return self
    
    def get_feature_names(self) -> list:
        """Get names of extracted features."""
        if self.selected_features is not None:
            return self.selected_features.copy()
        
        # Generate feature names based on parameters
        # This is a simplified version - actual TSFresh feature names are more complex
        feature_names = []
        
        for function_name in self.fc_parameters.keys():
            if isinstance(self.fc_parameters[function_name], list):
                for param_dict in self.fc_parameters[function_name]:
                    if param_dict is None:
                        feature_names.append(function_name)
                    else:
                        param_str = '_'.join([f"{k}_{v}" for k, v in param_dict.items()])
                        feature_names.append(f"{function_name}__{param_str}")
            else:
                feature_names.append(function_name)
        
        return feature_names
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration parameters."""
        config = super().get_config()
        config.update({
            'feature_set': self.feature_set,
            'feature_selection': self.feature_selection,
            'n_selected_features': len(self.selected_features) if self.selected_features else None,
            'total_features': self.embedding_dim
        })
        return config