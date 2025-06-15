"""Custom domain-specific feature extraction for organoid time series."""

import numpy as np
from typing import List, Optional, Tuple
import logging
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid as trapz
from sklearn.linear_model import LinearRegression

from ..base import FeatureEmbedder

logger = logging.getLogger(__name__)


class CustomFeaturesEmbedder(FeatureEmbedder):
    """
    Custom feature extraction for organoid oxygen time series data.
    
    Extracts domain-specific features relevant to organoid behavior,
    drug responses, and oxygen consumption patterns.
    """
    
    def __init__(self, smoothing_window: int = 5, 
                 response_window_hours: float = 24.0,
                 sampling_rate_hours: float = 1.0,
                 include_baseline_features: bool = True,
                 include_response_features: bool = True,
                 include_stability_features: bool = True,
                 **kwargs):
        """
        Initialize custom features embedder.
        
        Args:
            smoothing_window: Window size for smoothing (in time points)
            response_window_hours: Window for measuring acute response
            sampling_rate_hours: Sampling rate in hours per time point
            include_baseline_features: Whether to include baseline characteristics
            include_response_features: Whether to include drug response features
            include_stability_features: Whether to include stability measures
        """
        super().__init__(name="CustomFeatures", **kwargs)
        self.smoothing_window = smoothing_window
        self.response_window_hours = response_window_hours
        self.sampling_rate_hours = sampling_rate_hours
        self.include_baseline_features = include_baseline_features
        self.include_response_features = include_response_features
        self.include_stability_features = include_stability_features
        
        # Convert response window to time points
        self.response_window_points = int(response_window_hours / sampling_rate_hours)
        
    def _smooth_series(self, ts: np.ndarray) -> np.ndarray:
        """Apply smoothing to time series."""
        if len(ts) < self.smoothing_window:
            return ts
        
        try:
            # Use Savitzky-Golay filter for smoothing
            window_length = min(self.smoothing_window, len(ts))
            if window_length % 2 == 0:
                window_length -= 1  # Must be odd
            if window_length < 3:
                return ts
            
            return savgol_filter(ts, window_length, polyorder=1)
        except:
            # Fallback to simple moving average
            return np.convolve(ts, np.ones(self.smoothing_window)/self.smoothing_window, mode='same')
    
    def _extract_baseline_features(self, ts: np.ndarray) -> List[float]:
        """Extract baseline oxygen consumption characteristics."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(ts),           # Mean oxygen level
            np.std(ts),            # Variability
            np.median(ts),         # Median level
            stats.iqr(ts),         # Interquartile range
            np.min(ts),            # Minimum oxygen
            np.max(ts),            # Maximum oxygen
            np.ptp(ts)             # Peak-to-peak range
        ])
        
        # Distributional features
        try:
            skewness = stats.skew(ts)
            kurtosis = stats.kurtosis(ts)
        except:
            skewness = 0.0
            kurtosis = 0.0
        
        features.extend([skewness, kurtosis])
        
        # Percentiles
        percentiles = np.percentile(ts, [5, 25, 75, 95])
        features.extend(percentiles)
        
        return features
    
    def _extract_trend_features(self, ts: np.ndarray) -> List[float]:
        """Extract trend and slope characteristics."""
        features = []
        
        if len(ts) < 3:
            return [0.0] * 8
        
        # Overall linear trend
        x = np.arange(len(ts))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)
            features.extend([slope, intercept, r_value, p_value])
        except:
            features.extend([0.0, np.mean(ts), 0.0, 1.0])
        
        # Smoothed trend
        ts_smooth = self._smooth_series(ts)
        try:
            slope_smooth, _, r_smooth, _ = stats.linregress(x, ts_smooth)[:4]
            features.extend([slope_smooth, r_smooth])
        except:
            features.extend([0.0, 0.0])
        
        # Trend changes (second derivative)
        if len(ts) > 2:
            diff1 = np.diff(ts)
            diff2 = np.diff(diff1)
            features.extend([
                np.mean(np.abs(diff2)),  # Mean curvature
                np.std(diff2)            # Curvature variability
            ])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_response_features(self, ts: np.ndarray) -> List[float]:
        """Extract drug response characteristics."""
        features = []
        
        if len(ts) < self.response_window_points:
            # Time series too short for response analysis
            return [0.0] * 12
        
        # Define baseline and response periods
        baseline_period = ts[:self.response_window_points]
        response_period = ts[self.response_window_points:2*self.response_window_points]
        
        if len(response_period) == 0:
            response_period = ts[self.response_window_points:]
        
        # Baseline statistics
        baseline_mean = np.mean(baseline_period)
        baseline_std = np.std(baseline_period)
        
        # Response statistics
        response_mean = np.mean(response_period)
        response_std = np.std(response_period)
        
        # Response magnitude
        absolute_change = response_mean - baseline_mean
        relative_change = absolute_change / baseline_mean if baseline_mean != 0 else 0.0
        normalized_change = absolute_change / baseline_std if baseline_std != 0 else 0.0
        
        features.extend([
            absolute_change,
            relative_change,
            normalized_change,
            response_mean,
            response_std
        ])
        
        # Time to response (simplified)
        response_threshold = baseline_mean + 2 * baseline_std  # 2 sigma threshold
        time_to_response = len(ts)  # Default if no response found
        
        for i in range(self.response_window_points, len(ts)):
            if abs(ts[i] - baseline_mean) > 2 * baseline_std:
                time_to_response = i - self.response_window_points
                break
        
        features.append(time_to_response * self.sampling_rate_hours)  # Convert to hours
        
        # Recovery characteristics (if time series is long enough)
        if len(ts) > 3 * self.response_window_points:
            recovery_period = ts[2*self.response_window_points:3*self.response_window_points]
            recovery_mean = np.mean(recovery_period)
            
            # Recovery magnitude
            recovery_change = recovery_mean - response_mean
            recovery_to_baseline = abs(recovery_mean - baseline_mean) / abs(response_mean - baseline_mean) if response_mean != baseline_mean else 0.0
            
            features.extend([
                recovery_change,
                recovery_mean,
                recovery_to_baseline
            ])
        else:
            features.extend([0.0, response_mean, 0.0])
        
        # Peak response characteristics
        if len(response_period) > 0:
            peak_response = np.max(np.abs(response_period - baseline_mean))
            features.extend([peak_response])
        else:
            features.extend([0.0])
        
        # Area under the curve (response)
        if len(response_period) > 1:
            response_auc = trapz(np.abs(response_period - baseline_mean))
            features.extend([response_auc])
        else:
            features.extend([0.0])
        
        return features
    
    def _extract_oscillation_features(self, ts: np.ndarray) -> List[float]:
        """Extract oscillation and periodicity features."""
        features = []
        
        if len(ts) < 10:
            return [0.0] * 6
        
        # Smooth the series for peak detection
        ts_smooth = self._smooth_series(ts)
        
        # Find peaks and troughs
        try:
            peaks, peak_properties = find_peaks(ts_smooth, prominence=np.std(ts_smooth)*0.5)
            troughs, trough_properties = find_peaks(-ts_smooth, prominence=np.std(ts_smooth)*0.5)
            
            # Peak characteristics
            n_peaks = len(peaks)
            n_troughs = len(troughs)
            
            # Average peak/trough spacing (period estimation)
            if len(peaks) > 1:
                peak_spacing = np.mean(np.diff(peaks)) * self.sampling_rate_hours
            else:
                peak_spacing = 0.0
            
            if len(troughs) > 1:
                trough_spacing = np.mean(np.diff(troughs)) * self.sampling_rate_hours
            else:
                trough_spacing = 0.0
            
            # Amplitude characteristics
            if len(peaks) > 0:
                peak_amplitudes = ts_smooth[peaks]
                mean_peak_amplitude = np.mean(peak_amplitudes)
                std_peak_amplitude = np.std(peak_amplitudes) if len(peak_amplitudes) > 1 else 0.0
            else:
                mean_peak_amplitude = 0.0
                std_peak_amplitude = 0.0
            
            features.extend([
                n_peaks, n_troughs, peak_spacing, trough_spacing,
                mean_peak_amplitude, std_peak_amplitude
            ])
            
        except:
            features.extend([0.0] * 6)
        
        return features
    
    def _extract_stability_features(self, ts: np.ndarray) -> List[float]:
        """Extract stability and noise characteristics."""
        features = []
        
        if len(ts) < 3:
            return [0.0] * 8
        
        # Coefficient of variation
        cv = np.std(ts) / np.mean(ts) if np.mean(ts) != 0 else 0.0
        features.append(cv)
        
        # Successive differences
        diff1 = np.diff(ts)
        mean_abs_diff = np.mean(np.abs(diff1))
        std_diff = np.std(diff1)
        
        features.extend([mean_abs_diff, std_diff])
        
        # Signal-to-noise ratio estimate
        ts_smooth = self._smooth_series(ts)
        noise = ts - ts_smooth
        signal_power = np.var(ts_smooth)
        noise_power = np.var(noise)
        snr = signal_power / noise_power if noise_power > 0 else float('inf')
        snr = min(snr, 1000)  # Cap extremely high SNR values
        
        features.append(snr)
        
        # Autocorrelation at lag 1
        if len(ts) > 1:
            try:
                autocorr_1 = np.corrcoef(ts[:-1], ts[1:])[0, 1]
                if np.isnan(autocorr_1):
                    autocorr_1 = 0.0
            except:
                autocorr_1 = 0.0
        else:
            autocorr_1 = 0.0
        
        features.append(autocorr_1)
        
        # Number of zero crossings (relative to mean)
        mean_val = np.mean(ts)
        zero_crossings = np.sum(np.diff(np.signbit(ts - mean_val)))
        features.append(zero_crossings)
        
        # Longest run above/below mean
        above_mean = ts > mean_val
        runs = []
        current_run = 1
        
        for i in range(1, len(above_mean)):
            if above_mean[i] == above_mean[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        max_run_length = max(runs) if runs else 0
        features.append(max_run_length)
        
        # Hurst exponent approximation (simplified)
        try:
            if len(ts) > 10:
                # Simple rescaled range approximation
                n = len(ts)
                mean_ts = np.mean(ts)
                cum_devs = np.cumsum(ts - mean_ts)
                R = np.max(cum_devs) - np.min(cum_devs)
                S = np.std(ts)
                if S > 0:
                    hurst = np.log(R/S) / np.log(n)
                else:
                    hurst = 0.5
            else:
                hurst = 0.5
        except:
            hurst = 0.5
        
        features.append(hurst)
        
        return features
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract custom organoid features from time series data.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            Custom features
        """
        n_samples = len(X)
        logger.info(f"Extracting custom organoid features for {n_samples} time series")
        
        all_features = []
        
        for i in range(n_samples):
            ts = X[i]
            
            # Handle NaN values
            if np.all(np.isnan(ts)):
                # All NaN - return zero features
                sample_features = [0.0] * self._get_total_features()
            else:
                # Remove leading/trailing NaNs and interpolate internal NaNs
                valid_mask = ~np.isnan(ts)
                if np.any(valid_mask):
                    first_valid = np.argmax(valid_mask)
                    last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
                    ts_clean = ts[first_valid:last_valid+1]
                    
                    # Simple interpolation for internal NaNs
                    if np.any(np.isnan(ts_clean)):
                        from scipy.interpolate import interp1d
                        x = np.arange(len(ts_clean))
                        valid = ~np.isnan(ts_clean)
                        if np.sum(valid) > 1:
                            f = interp1d(x[valid], ts_clean[valid], kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                            ts_clean = f(x)
                        else:
                            ts_clean = np.full_like(ts_clean, np.nanmean(ts_clean))
                else:
                    ts_clean = np.zeros(1)
                
                sample_features = []
                
                # Extract different feature groups
                if self.include_baseline_features:
                    sample_features.extend(self._extract_baseline_features(ts_clean))
                    sample_features.extend(self._extract_trend_features(ts_clean))
                    sample_features.extend(self._extract_oscillation_features(ts_clean))
                
                if self.include_response_features:
                    sample_features.extend(self._extract_response_features(ts_clean))
                
                if self.include_stability_features:
                    sample_features.extend(self._extract_stability_features(ts_clean))
            
            # Handle any NaN features
            sample_features = [f if np.isfinite(f) else 0.0 for f in sample_features]
            all_features.append(sample_features)
        
        return np.array(all_features)
    
    def _get_total_features(self) -> int:
        """Calculate total number of features."""
        total = 0
        
        if self.include_baseline_features:
            total += 13  # baseline features
            total += 8   # trend features
            total += 6   # oscillation features
        
        if self.include_response_features:
            total += 12  # response features
        
        if self.include_stability_features:
            total += 8   # stability features
        
        return total
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        names = []
        
        if self.include_baseline_features:
            # Baseline features
            names.extend([
                'mean', 'std', 'median', 'iqr', 'min', 'max', 'range',
                'skewness', 'kurtosis', 'p5', 'p25', 'p75', 'p95'
            ])
            
            # Trend features
            names.extend([
                'slope', 'intercept', 'r_value', 'p_value',
                'slope_smooth', 'r_smooth', 'mean_curvature', 'curvature_std'
            ])
            
            # Oscillation features
            names.extend([
                'n_peaks', 'n_troughs', 'peak_spacing_hours', 'trough_spacing_hours',
                'mean_peak_amplitude', 'std_peak_amplitude'
            ])
        
        if self.include_response_features:
            names.extend([
                'absolute_change', 'relative_change', 'normalized_change',
                'response_mean', 'response_std', 'time_to_response_hours',
                'recovery_change', 'recovery_mean', 'recovery_to_baseline',
                'peak_response', 'response_auc'
            ])
        
        if self.include_stability_features:
            names.extend([
                'cv', 'mean_abs_diff', 'std_diff', 'snr', 'autocorr_1',
                'zero_crossings', 'max_run_length', 'hurst_exponent'
            ])
        
        return names