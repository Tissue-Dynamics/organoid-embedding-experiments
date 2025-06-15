"""Outlier detection utilities for organoid time series."""

import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Outlier detection for organoid time series data.
    
    Provides multiple methods for detecting outliers including
    statistical methods and machine learning approaches.
    """
    
    def __init__(self, method: str = 'zscore', 
                 threshold: float = 3.0,
                 contamination: float = 0.1):
        """
        Initialize outlier detector.
        
        Args:
            method: Detection method ('zscore', 'iqr', 'isolation_forest', 'modified_zscore')
            threshold: Threshold for statistical methods
            contamination: Expected proportion of outliers for ML methods
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        
        valid_methods = ['zscore', 'iqr', 'isolation_forest', 'modified_zscore']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def detect_point_outliers(self, ts: np.ndarray) -> np.ndarray:
        """
        Detect outlier points within a single time series.
        
        Args:
            ts: Time series data
            
        Returns:
            Boolean array indicating outlier points
        """
        if self.method == 'zscore':
            return self._zscore_outliers(ts)
        elif self.method == 'modified_zscore':
            return self._modified_zscore_outliers(ts)
        elif self.method == 'iqr':
            return self._iqr_outliers(ts)
        elif self.method == 'isolation_forest':
            return self._isolation_forest_outliers(ts)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _zscore_outliers(self, ts: np.ndarray) -> np.ndarray:
        """Z-score based outlier detection."""
        # Handle NaN values
        valid_mask = ~np.isnan(ts)
        outliers = np.zeros_like(ts, dtype=bool)
        
        if np.sum(valid_mask) < 3:
            return outliers
        
        valid_values = ts[valid_mask]
        z_scores = np.abs(stats.zscore(valid_values))
        
        # Mark outliers in original array
        outlier_indices = np.where(valid_mask)[0][z_scores > self.threshold]
        outliers[outlier_indices] = True
        
        return outliers
    
    def _modified_zscore_outliers(self, ts: np.ndarray) -> np.ndarray:
        """Modified Z-score using median absolute deviation."""
        valid_mask = ~np.isnan(ts)
        outliers = np.zeros_like(ts, dtype=bool)
        
        if np.sum(valid_mask) < 3:
            return outliers
        
        valid_values = ts[valid_mask]
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))
        
        if mad == 0:
            return outliers
        
        # Modified z-score
        modified_z_scores = 0.6745 * (valid_values - median) / mad
        
        # Mark outliers
        outlier_indices = np.where(valid_mask)[0][np.abs(modified_z_scores) > self.threshold]
        outliers[outlier_indices] = True
        
        return outliers
    
    def _iqr_outliers(self, ts: np.ndarray) -> np.ndarray:
        """Interquartile range based outlier detection."""
        valid_mask = ~np.isnan(ts)
        outliers = np.zeros_like(ts, dtype=bool)
        
        if np.sum(valid_mask) < 4:
            return outliers
        
        valid_values = ts[valid_mask]
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return outliers
        
        # IQR bounds
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr
        
        # Mark outliers
        outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
        outlier_indices = np.where(valid_mask)[0][outlier_mask]
        outliers[outlier_indices] = True
        
        return outliers
    
    def _isolation_forest_outliers(self, ts: np.ndarray) -> np.ndarray:
        """Isolation Forest based outlier detection."""
        valid_mask = ~np.isnan(ts)
        outliers = np.zeros_like(ts, dtype=bool)
        
        if np.sum(valid_mask) < 10:  # Need minimum samples
            return outliers
        
        valid_values = ts[valid_mask].reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(valid_values)
        
        # Mark outliers (-1 indicates outlier)
        outlier_indices = np.where(valid_mask)[0][predictions == -1]
        outliers[outlier_indices] = True
        
        return outliers
    
    def detect_series_outliers(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Detect outlier time series from a collection.
        
        Args:
            data: List of time series arrays
            
        Returns:
            Boolean array indicating outlier series
        """
        # Extract features from each series
        features = []
        
        for ts in data:
            valid_values = ts[~np.isnan(ts)]
            
            if len(valid_values) == 0:
                # All NaN series
                features.append([0, 0, 0, 0, 0])
                continue
            
            # Basic statistical features
            ts_features = [
                np.mean(valid_values),
                np.std(valid_values),
                np.median(valid_values),
                np.min(valid_values),
                np.max(valid_values)
            ]
            
            features.append(ts_features)
        
        features = np.array(features)
        
        # Detect outliers based on features
        if self.method == 'isolation_forest':
            # Use Isolation Forest on features
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            predictions = iso_forest.fit_predict(features)
            return predictions == -1
        
        else:
            # Use multivariate statistical methods
            outliers = np.zeros(len(data), dtype=bool)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Compute Mahalanobis distance
            try:
                cov_inv = np.linalg.pinv(np.cov(features_scaled.T))
                mean_features = np.mean(features_scaled, axis=0)
                
                for i, feature_vec in enumerate(features_scaled):
                    diff = feature_vec - mean_features
                    mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
                    
                    # Use chi-square distribution critical value
                    critical_value = stats.chi2.ppf(1 - 0.05, features_scaled.shape[1])
                    outliers[i] = mahal_dist > critical_value
                    
            except np.linalg.LinAlgError:
                # Fallback to simple thresholding
                for feature_idx in range(features_scaled.shape[1]):
                    feature_outliers = self._zscore_outliers(features_scaled[:, feature_idx])
                    outliers |= feature_outliers
            
            return outliers
    
    def remove_outliers(self, ts: np.ndarray, replace_method: str = 'interpolate') -> np.ndarray:
        """
        Remove or replace outlier points in time series.
        
        Args:
            ts: Time series data
            replace_method: How to handle outliers ('remove', 'interpolate', 'median')
            
        Returns:
            Time series with outliers handled
        """
        outliers = self.detect_point_outliers(ts)
        
        if not np.any(outliers):
            return ts.copy()
        
        ts_clean = ts.copy()
        
        if replace_method == 'remove':
            # Replace with NaN
            ts_clean[outliers] = np.nan
            
        elif replace_method == 'interpolate':
            # Replace with NaN then interpolate
            ts_clean[outliers] = np.nan
            
            # Simple linear interpolation
            valid_mask = ~np.isnan(ts_clean)
            if np.sum(valid_mask) >= 2:
                valid_indices = np.where(valid_mask)[0]
                ts_clean = np.interp(
                    np.arange(len(ts_clean)),
                    valid_indices,
                    ts_clean[valid_indices]
                )
            
        elif replace_method == 'median':
            # Replace with local median
            window_size = 5
            
            for outlier_idx in np.where(outliers)[0]:
                # Get local window
                start_idx = max(0, outlier_idx - window_size // 2)
                end_idx = min(len(ts), outlier_idx + window_size // 2 + 1)
                
                window = ts[start_idx:end_idx]
                window_clean = window[~outliers[start_idx:end_idx]]
                
                if len(window_clean) > 0:
                    ts_clean[outlier_idx] = np.median(window_clean)
                else:
                    ts_clean[outlier_idx] = np.nan
        
        else:
            raise ValueError(f"Unknown replace method: {replace_method}")
        
        return ts_clean
    
    def batch_remove_outliers(self, data: List[np.ndarray], 
                            **kwargs) -> List[np.ndarray]:
        """
        Remove outliers from a batch of time series.
        
        Args:
            data: List of time series arrays
            **kwargs: Arguments for remove_outliers
            
        Returns:
            List of cleaned time series
        """
        cleaned_data = []
        total_outliers = 0
        
        for ts in data:
            ts_clean = self.remove_outliers(ts, **kwargs)
            cleaned_data.append(ts_clean)
            
            # Count outliers
            outliers = self.detect_point_outliers(ts)
            total_outliers += np.sum(outliers)
        
        logger.info(f"Removed {total_outliers} outlier points from {len(data)} series")
        
        return cleaned_data