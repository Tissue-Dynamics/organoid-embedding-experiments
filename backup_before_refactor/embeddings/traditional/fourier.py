"""Fourier Transform embedding implementation."""

import numpy as np
from typing import Optional, Tuple
import logging
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

from ..base import FeatureEmbedder

logger = logging.getLogger(__name__)


class FourierEmbedder(FeatureEmbedder):
    """
    Fourier Transform embedding for time series data.
    
    Extracts frequency domain features including magnitude spectrum,
    phase spectrum, and derived frequency characteristics.
    """
    
    def __init__(self, n_components: Optional[int] = None, 
                 include_phase: bool = False, include_derived: bool = True,
                 freq_bands: Optional[list] = None, **kwargs):
        """
        Initialize Fourier embedder.
        
        Args:
            n_components: Number of frequency components to keep (None = all)
            include_phase: Whether to include phase information
            include_derived: Whether to include derived frequency features
            freq_bands: Frequency bands for band-specific features
        """
        super().__init__(name="Fourier", **kwargs)
        self.n_components = n_components
        self.include_phase = include_phase
        self.include_derived = include_derived
        self.freq_bands = freq_bands or [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)]
        self.sampling_rate = kwargs.get('sampling_rate', 1.0)  # Hz
        
    def _compute_frequency_features(self, ts: np.ndarray) -> np.ndarray:
        """
        Compute frequency domain features for a single time series.
        
        Args:
            ts: Time series data
            
        Returns:
            Frequency features
        """
        # Remove NaN values and interpolate if necessary
        if np.any(np.isnan(ts)):
            valid_idx = ~np.isnan(ts)
            if np.sum(valid_idx) < 3:  # Too few valid points
                return np.full(self._get_feature_dim(len(ts)), np.nan)
            
            # Simple linear interpolation
            x_valid = np.where(valid_idx)[0]
            y_valid = ts[valid_idx]
            ts = np.interp(np.arange(len(ts)), x_valid, y_valid)
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(ts))
        ts_windowed = ts * window
        
        # Compute FFT
        fft_values = fft(ts_windowed)
        freqs = fftfreq(len(ts), 1.0 / self.sampling_rate)
        
        # Take only positive frequencies (symmetric spectrum)
        n_freq = len(freqs) // 2
        fft_values = fft_values[:n_freq]
        freqs = freqs[:n_freq]
        
        # Magnitude and phase
        magnitude = np.abs(fft_values)
        phase = np.angle(fft_values) if self.include_phase else None
        
        features = []
        
        # Magnitude spectrum features
        if self.n_components is not None:
            magnitude = magnitude[:min(self.n_components, len(magnitude))]
        features.extend(magnitude)
        
        # Phase spectrum features
        if self.include_phase:
            if phase is not None:
                if self.n_components is not None:
                    phase = phase[:min(self.n_components, len(phase))]
                features.extend(phase)
        
        # Derived frequency features
        if self.include_derived:
            # Power spectral density
            psd = magnitude ** 2
            total_power = np.sum(psd)
            
            if total_power > 0:
                # Use truncated arrays for spectral statistics
                freqs_truncated = freqs[:len(magnitude)]
                psd_truncated = psd[:len(magnitude)]
                total_power_truncated = np.sum(psd_truncated)
                
                if total_power_truncated > 0:
                    # Spectral centroid (center of mass of spectrum)
                    spectral_centroid = np.sum(freqs_truncated * psd_truncated) / total_power_truncated
                    
                    # Spectral spread (standard deviation of spectrum)
                    spectral_spread = np.sqrt(np.sum(((freqs_truncated - spectral_centroid) ** 2) * psd_truncated) / total_power_truncated)
                    
                    if spectral_spread > 0:
                        # Spectral skewness
                        spectral_skewness = np.sum(((freqs_truncated - spectral_centroid) ** 3) * psd_truncated) / (total_power_truncated * spectral_spread ** 3)
                        
                        # Spectral kurtosis
                        spectral_kurtosis = np.sum(((freqs_truncated - spectral_centroid) ** 4) * psd_truncated) / (total_power_truncated * spectral_spread ** 4)
                    else:
                        spectral_skewness = 0.0
                        spectral_kurtosis = 0.0
                    
                    # Peak frequency
                    peak_freq = freqs_truncated[np.argmax(psd_truncated)]
                else:
                    spectral_centroid = spectral_spread = spectral_skewness = spectral_kurtosis = peak_freq = 0.0
                
                features.extend([spectral_centroid, spectral_spread, spectral_skewness, 
                               spectral_kurtosis, peak_freq])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Band-specific power - ensure arrays are same length
            freqs_for_bands = freqs[:len(magnitude)]  # Match magnitude length
            psd_for_bands = psd[:len(magnitude)]      # Match magnitude length
            
            for low_freq, high_freq in self.freq_bands:
                band_mask = (freqs_for_bands >= low_freq) & (freqs_for_bands <= high_freq)
                band_power = np.sum(psd_for_bands[band_mask])
                relative_band_power = band_power / total_power if total_power > 0 else 0.0
                features.extend([band_power, relative_band_power])
        
        return np.array(features)
    
    def _get_feature_dim(self, ts_length: int) -> int:
        """Calculate the number of features that will be extracted."""
        n_freq = ts_length // 2
        
        # Magnitude features
        n_magnitude = min(self.n_components, n_freq) if self.n_components else n_freq
        n_features = n_magnitude
        
        # Phase features
        if self.include_phase:
            n_features += n_magnitude
        
        # Derived features
        if self.include_derived:
            n_features += 5  # 5 spectral moments/characteristics
            n_features += len(self.freq_bands) * 2  # Power and relative power per band
        
        return n_features
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract Fourier features from time series data.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            Fourier features of shape (n_samples, n_features)
        """
        n_samples = len(X)
        feature_dim = self._get_feature_dim(X.shape[1])
        features = np.zeros((n_samples, feature_dim))
        
        logger.info(f"Extracting Fourier features for {n_samples} time series")
        
        for i in range(n_samples):
            features[i] = self._compute_frequency_features(X[i])
        
        # Handle any NaN features
        nan_mask = np.isnan(features)
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} NaN features, replacing with column means")
            col_means = np.nanmean(features, axis=0)
            for j in range(features.shape[1]):
                nan_rows = nan_mask[:, j]
                features[nan_rows, j] = col_means[j]
        
        return features
    
    def get_feature_names(self) -> list:
        """Get names of extracted features."""
        names = []
        
        # Magnitude features
        n_magnitude = self.n_components if self.n_components else "all"
        for i in range(self.n_components if self.n_components else 100):  # Placeholder
            names.append(f"magnitude_{i}")
        
        # Phase features
        if self.include_phase:
            for i in range(self.n_components if self.n_components else 100):
                names.append(f"phase_{i}")
        
        # Derived features
        if self.include_derived:
            names.extend(['spectral_centroid', 'spectral_spread', 'spectral_skewness',
                         'spectral_kurtosis', 'peak_frequency'])
            
            for low_freq, high_freq in self.freq_bands:
                names.extend([f"band_power_{low_freq}_{high_freq}", 
                             f"relative_band_power_{low_freq}_{high_freq}"])
        
        return names