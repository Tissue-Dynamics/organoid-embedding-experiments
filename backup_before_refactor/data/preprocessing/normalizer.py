"""Time series normalization utilities."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class TimeSeriesNormalizer:
    """
    Normalization utilities for organoid time series data.
    
    Provides various normalization strategies including per-series,
    global, and control-based normalization approaches.
    """
    
    def __init__(self, method: str = 'zscore', 
                 global_normalization: bool = False,
                 handle_missing: str = 'skip'):
        """
        Initialize time series normalizer.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust', 'unit_norm')
            global_normalization: Whether to normalize globally across all series
            handle_missing: How to handle missing values ('skip', 'interpolate', 'mean')
        """
        self.method = method
        self.global_normalization = global_normalization
        self.handle_missing = handle_missing
        
        # Storage for fitted parameters
        self.fitted_params = {}
        self.is_fitted = False
        
    def _handle_missing_values(self, ts: np.ndarray) -> np.ndarray:
        """Handle missing values according to specified strategy."""
        if not np.any(np.isnan(ts)):
            return ts
        
        if self.handle_missing == 'skip':
            return ts  # Return as-is, normalization will handle NaNs
        
        elif self.handle_missing == 'interpolate':
            # Linear interpolation
            valid_mask = ~np.isnan(ts)
            if np.sum(valid_mask) < 2:
                return ts  # Can't interpolate with less than 2 points
            
            valid_indices = np.where(valid_mask)[0]
            interpolated = np.interp(
                np.arange(len(ts)), 
                valid_indices, 
                ts[valid_indices]
            )
            return interpolated
        
        elif self.handle_missing == 'mean':
            # Replace with mean of valid values
            mean_val = np.nanmean(ts)
            ts_filled = ts.copy()
            ts_filled[np.isnan(ts_filled)] = mean_val
            return ts_filled
        
        else:
            raise ValueError(f"Unknown missing value strategy: {self.handle_missing}")
    
    def _zscore_normalize(self, ts: np.ndarray, 
                         global_params: Optional[Dict] = None) -> np.ndarray:
        """Z-score normalization (zero mean, unit variance)."""
        ts = self._handle_missing_values(ts)
        
        if global_params:
            mean_val = global_params['mean']
            std_val = global_params['std']
        else:
            mean_val = np.nanmean(ts)
            std_val = np.nanstd(ts)
        
        if std_val == 0 or np.isnan(std_val):
            # Constant series or all NaN
            return np.zeros_like(ts)
        
        normalized = (ts - mean_val) / std_val
        return normalized
    
    def _minmax_normalize(self, ts: np.ndarray,
                         global_params: Optional[Dict] = None) -> np.ndarray:
        """Min-max normalization to [0, 1] range."""
        ts = self._handle_missing_values(ts)
        
        if global_params:
            min_val = global_params['min']
            max_val = global_params['max']
        else:
            min_val = np.nanmin(ts)
            max_val = np.nanmax(ts)
        
        if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
            # Constant series or all NaN
            return np.zeros_like(ts)
        
        normalized = (ts - min_val) / (max_val - min_val)
        return normalized
    
    def _robust_normalize(self, ts: np.ndarray,
                         global_params: Optional[Dict] = None) -> np.ndarray:
        """Robust normalization using median and IQR."""
        ts = self._handle_missing_values(ts)
        
        if global_params:
            median_val = global_params['median']
            iqr_val = global_params['iqr']
        else:
            median_val = np.nanmedian(ts)
            q75 = np.nanpercentile(ts, 75)
            q25 = np.nanpercentile(ts, 25)
            iqr_val = q75 - q25
        
        if iqr_val == 0 or np.isnan(iqr_val):
            # No variation or all NaN
            return np.zeros_like(ts)
        
        normalized = (ts - median_val) / iqr_val
        return normalized
    
    def _unit_norm_normalize(self, ts: np.ndarray) -> np.ndarray:
        """Unit norm normalization (L2 norm = 1)."""
        ts = self._handle_missing_values(ts)
        
        # Handle NaN values
        valid_mask = ~np.isnan(ts)
        if not np.any(valid_mask):
            return ts
        
        norm = np.linalg.norm(ts[valid_mask])
        if norm == 0:
            return ts
        
        normalized = ts.copy()
        normalized[valid_mask] = ts[valid_mask] / norm
        return normalized
    
    def _compute_global_params(self, data: List[np.ndarray]) -> Dict:
        """Compute global normalization parameters across all series."""
        all_values = []
        
        for ts in data:
            ts_clean = self._handle_missing_values(ts)
            valid_values = ts_clean[~np.isnan(ts_clean)]
            all_values.extend(valid_values)
        
        if not all_values:
            raise ValueError("No valid values found in data")
        
        all_values = np.array(all_values)
        
        params = {}
        
        if self.method == 'zscore':
            params['mean'] = np.mean(all_values)
            params['std'] = np.std(all_values)
        
        elif self.method == 'minmax':
            params['min'] = np.min(all_values)
            params['max'] = np.max(all_values)
        
        elif self.method == 'robust':
            params['median'] = np.median(all_values)
            q75 = np.percentile(all_values, 75)
            q25 = np.percentile(all_values, 25)
            params['iqr'] = q75 - q25
        
        return params
    
    def fit(self, data: List[np.ndarray]) -> 'TimeSeriesNormalizer':
        """
        Fit normalization parameters to data.
        
        Args:
            data: List of time series arrays
            
        Returns:
            Self for method chaining
        """
        if self.global_normalization:
            self.fitted_params = self._compute_global_params(data)
            logger.info(f"Fitted global {self.method} normalization: {self.fitted_params}")
        else:
            self.fitted_params = {}
            logger.info(f"Fitted per-series {self.method} normalization")
        
        self.is_fitted = True
        return self
    
    def transform(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply normalization to data.
        
        Args:
            data: List of time series arrays
            
        Returns:
            List of normalized time series
        """
        normalized_data = []
        
        for ts in data:
            global_params = self.fitted_params if self.global_normalization else None
            
            if self.method == 'zscore':
                normalized_ts = self._zscore_normalize(ts, global_params)
            elif self.method == 'minmax':
                normalized_ts = self._minmax_normalize(ts, global_params)
            elif self.method == 'robust':
                normalized_ts = self._robust_normalize(ts, global_params)
            elif self.method == 'unit_norm':
                normalized_ts = self._unit_norm_normalize(ts)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            normalized_data.append(normalized_ts)
        
        return normalized_data
    
    def fit_transform(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: List of normalized time series arrays
            
        Returns:
            List of denormalized time series
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
        
        if self.method == 'unit_norm':
            raise ValueError("Inverse transform not supported for unit norm")
        
        denormalized_data = []
        
        for ts in data:
            if self.global_normalization:
                params = self.fitted_params
            else:
                # For per-series normalization, we can't perfectly reverse
                # without storing individual parameters
                logger.warning("Inverse transform with per-series normalization may be inaccurate")
                params = None
            
            if self.method == 'zscore' and params:
                denormalized = ts * params['std'] + params['mean']
            elif self.method == 'minmax' and params:
                denormalized = ts * (params['max'] - params['min']) + params['min']
            elif self.method == 'robust' and params:
                denormalized = ts * params['iqr'] + params['median']
            else:
                # Fallback: return as-is
                denormalized = ts
            
            denormalized_data.append(denormalized)
        
        return denormalized_data


class ControlBasedNormalizer:
    """
    Control-based normalization for organoid experiments.
    
    Normalizes treatment samples relative to control conditions.
    """
    
    def __init__(self, normalization_method: str = 'relative_change',
                 baseline_method: str = 'mean'):
        """
        Initialize control-based normalizer.
        
        Args:
            normalization_method: 'relative_change', 'fold_change', 'z_score'
            baseline_method: How to compute control baseline ('mean', 'median')
        """
        self.normalization_method = normalization_method
        self.baseline_method = baseline_method
        
        self.control_baseline = None
        self.control_std = None
        
    def fit_controls(self, control_data: List[np.ndarray]):
        """
        Fit normalization using control data.
        
        Args:
            control_data: List of control time series
        """
        if not control_data:
            raise ValueError("No control data provided")
        
        # Compute control statistics at each time point
        max_length = max(len(ts) for ts in control_data)
        
        # Pad shorter series with NaN
        padded_controls = []
        for ts in control_data:
            padded = np.full(max_length, np.nan)
            padded[:len(ts)] = ts
            padded_controls.append(padded)
        
        control_matrix = np.array(padded_controls)
        
        if self.baseline_method == 'mean':
            self.control_baseline = np.nanmean(control_matrix, axis=0)
        elif self.baseline_method == 'median':
            self.control_baseline = np.nanmedian(control_matrix, axis=0)
        else:
            raise ValueError(f"Unknown baseline method: {self.baseline_method}")
        
        self.control_std = np.nanstd(control_matrix, axis=0)
        
        logger.info(f"Fitted control-based normalization with {len(control_data)} controls")
    
    def normalize_to_controls(self, ts: np.ndarray) -> np.ndarray:
        """
        Normalize a time series relative to controls.
        
        Args:
            ts: Time series to normalize
            
        Returns:
            Normalized time series
        """
        if self.control_baseline is None:
            raise ValueError("Must fit controls before normalization")
        
        # Ensure same length
        min_length = min(len(ts), len(self.control_baseline))
        ts_trimmed = ts[:min_length]
        baseline_trimmed = self.control_baseline[:min_length]
        std_trimmed = self.control_std[:min_length]
        
        if self.normalization_method == 'relative_change':
            # (ts - control) / control
            normalized = np.divide(
                ts_trimmed - baseline_trimmed,
                baseline_trimmed,
                out=np.zeros_like(ts_trimmed),
                where=(baseline_trimmed != 0)
            )
        
        elif self.normalization_method == 'fold_change':
            # ts / control
            normalized = np.divide(
                ts_trimmed,
                baseline_trimmed,
                out=np.ones_like(ts_trimmed),
                where=(baseline_trimmed != 0)
            )
        
        elif self.normalization_method == 'z_score':
            # (ts - control_mean) / control_std
            normalized = np.divide(
                ts_trimmed - baseline_trimmed,
                std_trimmed,
                out=np.zeros_like(ts_trimmed),
                where=(std_trimmed != 0)
            )
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        return normalized
    
    def batch_normalize(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize a batch of time series.
        
        Args:
            data: List of time series to normalize
            
        Returns:
            List of normalized time series
        """
        return [self.normalize_to_controls(ts) for ts in data]