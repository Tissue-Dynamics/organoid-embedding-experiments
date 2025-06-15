"""Catch22 feature extraction embedding implementation."""

import numpy as np
from typing import Optional, List
import logging
import pycatch22 as catch22

from ..base import FeatureEmbedder

logger = logging.getLogger(__name__)


class Catch22Embedder(FeatureEmbedder):
    """
    Catch22 (Canonical Time-series CHaracteristics) embedding.
    
    Extracts the canonical set of 22 time series features that are
    highly comparative and interpretable.
    """
    
    def __init__(self, catch24: bool = False, **kwargs):
        """
        Initialize Catch22 embedder.
        
        Args:
            catch24: Whether to include the 2 additional features (catch24)
        """
        super().__init__(name="Catch22", **kwargs)
        self.catch24 = catch24
        
        # Get feature names - pycatch22 uses different attribute names
        try:
            if catch24:
                self.feature_names = getattr(catch22, 'catch24_names', [f'catch24_{i}' for i in range(24)])
            else:
                self.feature_names = getattr(catch22, 'catch22_names', [f'catch22_{i}' for i in range(22)])
        except AttributeError:
            # Fallback to generic names
            n_features = 24 if catch24 else 22
            self.feature_names = [f'catch22_{i}' for i in range(n_features)]
    
    def _extract_catch22_features(self, ts: np.ndarray) -> np.ndarray:
        """
        Extract catch22 features from a single time series.
        
        Args:
            ts: Single time series
            
        Returns:
            Feature vector
        """
        # Handle NaN values
        if np.all(np.isnan(ts)):
            # All NaN - return zero features
            n_features = 24 if self.catch24 else 22
            return np.zeros(n_features)
        
        # Remove NaN values
        clean_ts = ts[~np.isnan(ts)]
        
        if len(clean_ts) < 3:
            # Too few valid points
            n_features = 24 if self.catch24 else 22
            return np.zeros(n_features)
        
        try:
            # pycatch22.catch22_all returns a dict with 'names' and 'values' keys
            result = catch22.catch22_all(clean_ts)
            
            if isinstance(result, dict) and 'values' in result:
                values = result['values']
            else:
                values = result
            
            if self.catch24:
                # Use all available features up to 24
                features = values[:24] if len(values) >= 24 else list(values) + [0.0] * (24 - len(values))
            else:
                # Use first 22 features
                features = values[:22] if len(values) >= 22 else list(values) + [0.0] * (22 - len(values))
            
            # Handle any remaining NaN or inf values
            features = np.array(features, dtype=np.float64)
            features[~np.isfinite(features)] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract catch22 features: {e}")
            n_features = 24 if self.catch24 else 22
            return np.zeros(n_features)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract catch22 features from time series data.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            Catch22 features of shape (n_samples, 22) or (n_samples, 24)
        """
        n_samples = len(X)
        n_features = 24 if self.catch24 else 22
        features = np.zeros((n_samples, n_features))
        
        logger.info(f"Extracting catch22 features for {n_samples} time series")
        
        for i in range(n_samples):
            features[i] = self._extract_catch22_features(X[i])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return self.feature_names.copy()
    
    def get_feature_descriptions(self) -> List[str]:
        """Get descriptions of extracted features."""
        # Feature descriptions based on catch22 documentation
        descriptions = [
            "DN_HistogramMode_5: Mode of z-scored distribution (5-bin histogram)",
            "DN_HistogramMode_10: Mode of z-scored distribution (10-bin histogram)",
            "CO_f1ecac: First 1/e crossing of autocorrelation function",
            "CO_FirstMin_ac: First minimum of autocorrelation function",
            "CO_HistogramAMI_even_2_5: Automutual information (histogram method)",
            "CO_trev_1_num: Time-reversal asymmetry statistic",
            "MD_hrv_classic_pnn40: pNN40 heart rate variability measure",
            "SB_BinaryStats_mean_longstretch1: Longest stretch of consecutive values above mean",
            "SB_TransitionMatrix_3ac_sumdiagcov: Transition matrix (3-state symbolic)",
            "PD_PeriodicityWang_th0_01: Periodicity measure of Wang et al.",
            "CO_Embed2_Dist_tau_d_expfit_meandiff: Exponential fit to inter-point distances",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi: Automutual information statistic",
            "FC_LocalSimple_mean1_tauresrat: Ratios of successive tau values",
            "DN_OutlierInclude_p_001_mdrmd: Median absolute deviation outlier analysis",
            "DN_OutlierInclude_n_001_mdrmd: Median absolute deviation outlier analysis",
            "SP_Summaries_welch_rect_area_5_1: Power spectral density area",
            "SB_BinaryStats_diff_longstretch0: Stretch of consecutive decreasing values",
            "SB_MotifThree_quantile_hh: 3-element motif analysis",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1: Fluctuation analysis",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1: Detrended fluctuation analysis",
            "SP_Summaries_welch_rect_centroid: Power spectral density centroid",
            "FC_LocalSimple_mean3_stderr: Simple local time-series forecasting"
        ]
        
        if self.catch24:
            descriptions.extend([
                "CO_trev_1_num: Time-reversal asymmetry statistic (alternative)",
                "CO_HistogramAMI_even_2_5: Automutual information (alternative)"
            ])
        
        return descriptions
    
    def get_config(self) -> dict:
        """Return configuration parameters."""
        config = super().get_config()
        config.update({
            'catch24': self.catch24,
            'n_features': 24 if self.catch24 else 22
        })
        return config