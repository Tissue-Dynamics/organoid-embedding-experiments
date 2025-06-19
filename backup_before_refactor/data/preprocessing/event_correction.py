"""Event correction utilities for organoid experiments."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class EventCorrector:
    """
    Event correction for organoid time series data.
    
    Handles experimental artifacts like media changes that cause
    global shifts in oxygen levels across all wells.
    """
    
    def __init__(self, detection_method: str = 'global_shift',
                 correction_method: str = 'baseline_adjustment',
                 min_shift_magnitude: float = 5.0,
                 shift_window_hours: float = 2.0):
        """
        Initialize event corrector.
        
        Args:
            detection_method: Method for detecting events ('global_shift', 'known_events')
            correction_method: How to correct events ('baseline_adjustment', 'detrend', 'normalize')
            min_shift_magnitude: Minimum magnitude to consider as event
            shift_window_hours: Time window to detect shifts
        """
        self.detection_method = detection_method
        self.correction_method = correction_method
        self.min_shift_magnitude = min_shift_magnitude
        self.shift_window_hours = shift_window_hours
        
    def detect_global_shifts(self, data: List[np.ndarray],
                           timestamps: Optional[List[np.ndarray]] = None,
                           sampling_rate_hours: float = 1.0) -> List[int]:
        """
        Detect global shift events affecting multiple time series.
        
        Args:
            data: List of time series arrays
            timestamps: Optional timestamp arrays
            sampling_rate_hours: Sampling rate in hours
            
        Returns:
            List of time indices where shifts occur
        """
        if len(data) == 0:
            return []
        
        # Find common time length
        min_length = min(len(ts) for ts in data)
        window_size = int(self.shift_window_hours / sampling_rate_hours)
        
        # Compute median signal across all series at each time point
        median_signal = np.zeros(min_length)
        
        for t in range(min_length):
            values_at_t = []
            for ts in data:
                if t < len(ts) and not np.isnan(ts[t]):
                    values_at_t.append(ts[t])
            
            if values_at_t:
                median_signal[t] = np.median(values_at_t)
            else:
                median_signal[t] = np.nan
        
        # Detect shifts in median signal
        shift_indices = []
        
        if window_size < 2:
            window_size = 2
        
        for t in range(window_size, min_length - window_size):
            # Compare before and after windows
            before_window = median_signal[t-window_size:t]
            after_window = median_signal[t:t+window_size]
            
            # Remove NaN values
            before_valid = before_window[~np.isnan(before_window)]
            after_valid = after_window[~np.isnan(after_window)]
            
            if len(before_valid) == 0 or len(after_valid) == 0:
                continue
            
            # Calculate shift magnitude
            before_mean = np.mean(before_valid)
            after_mean = np.mean(after_valid)
            shift_magnitude = abs(after_mean - before_mean)
            
            # Check if shift is significant
            if shift_magnitude >= self.min_shift_magnitude:
                # Additional statistical test
                try:
                    _, p_value = stats.ttest_ind(before_valid, after_valid)
                    if p_value < 0.05:  # Significant difference
                        shift_indices.append(t)
                except:
                    # Fallback to magnitude only
                    shift_indices.append(t)
        
        # Remove consecutive detections (keep only first)
        filtered_shifts = []
        for shift_idx in shift_indices:
            if not filtered_shifts or shift_idx - filtered_shifts[-1] > window_size:
                filtered_shifts.append(shift_idx)
        
        logger.info(f"Detected {len(filtered_shifts)} global shift events")
        return filtered_shifts
    
    def detect_known_events(self, event_times: List[float],
                          timestamps: Optional[List[np.ndarray]] = None,
                          sampling_rate_hours: float = 1.0) -> List[int]:
        """
        Convert known event times to time indices.
        
        Args:
            event_times: List of event times (in hours or timestamp units)
            timestamps: Optional timestamp arrays
            sampling_rate_hours: Sampling rate in hours
            
        Returns:
            List of time indices corresponding to events
        """
        if not event_times:
            return []
        
        event_indices = []
        
        if timestamps and len(timestamps) > 0:
            # Use actual timestamps
            reference_timestamps = timestamps[0]  # Use first series as reference
            
            for event_time in event_times:
                # Find closest timestamp
                time_diffs = np.abs(reference_timestamps - event_time)
                closest_idx = np.argmin(time_diffs)
                event_indices.append(closest_idx)
        
        else:
            # Convert time to indices
            for event_time in event_times:
                event_idx = int(event_time / sampling_rate_hours)
                event_indices.append(event_idx)
        
        logger.info(f"Converted {len(event_times)} known events to indices")
        return event_indices
    
    def correct_baseline_shift(self, ts: np.ndarray, 
                             shift_indices: List[int]) -> np.ndarray:
        """
        Correct baseline shifts by adjusting segments.
        
        Args:
            ts: Time series data
            shift_indices: Indices where shifts occur
            
        Returns:
            Corrected time series
        """
        if not shift_indices:
            return ts.copy()
        
        corrected_ts = ts.copy()
        
        # Define segments between shifts
        segment_starts = [0] + shift_indices
        segment_ends = shift_indices + [len(ts)]
        
        # Reference: use first segment baseline
        reference_segment = corrected_ts[segment_starts[0]:segment_ends[0]]
        reference_baseline = np.nanmedian(reference_segment)
        
        # Adjust each subsequent segment
        for i in range(1, len(segment_starts)):
            segment_start = segment_starts[i]
            segment_end = segment_ends[i]
            
            segment = corrected_ts[segment_start:segment_end]
            segment_baseline = np.nanmedian(segment)
            
            if not np.isnan(segment_baseline) and not np.isnan(reference_baseline):
                # Adjust segment to match reference baseline
                adjustment = reference_baseline - segment_baseline
                corrected_ts[segment_start:segment_end] += adjustment
        
        return corrected_ts
    
    def correct_linear_detrend(self, ts: np.ndarray,
                             shift_indices: List[int]) -> np.ndarray:
        """
        Correct by detrending each segment separately.
        
        Args:
            ts: Time series data
            shift_indices: Indices where shifts occur
            
        Returns:
            Detrended time series
        """
        if not shift_indices:
            return self._detrend_segment(ts, 0, len(ts))
        
        corrected_ts = ts.copy()
        
        # Define segments
        segment_starts = [0] + shift_indices
        segment_ends = shift_indices + [len(ts)]
        
        # Detrend each segment
        for i in range(len(segment_starts)):
            segment_start = segment_starts[i]
            segment_end = segment_ends[i]
            
            segment_corrected = self._detrend_segment(
                corrected_ts, segment_start, segment_end
            )
            corrected_ts[segment_start:segment_end] = segment_corrected
        
        return corrected_ts
    
    def _detrend_segment(self, ts: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """Detrend a segment of time series."""
        segment = ts[start_idx:end_idx]
        
        # Handle NaN values
        valid_mask = ~np.isnan(segment)
        if np.sum(valid_mask) < 2:
            return segment
        
        # Fit linear trend
        x = np.arange(len(segment))
        valid_x = x[valid_mask]
        valid_y = segment[valid_mask]
        
        try:
            slope, intercept = np.polyfit(valid_x, valid_y, 1)
            trend = slope * x + intercept
            detrended = segment - trend + np.nanmean(segment)
            return detrended
        except:
            return segment
    
    def correct_segment_normalization(self, ts: np.ndarray,
                                    shift_indices: List[int]) -> np.ndarray:
        """
        Normalize each segment separately.
        
        Args:
            ts: Time series data
            shift_indices: Indices where shifts occur
            
        Returns:
            Normalized time series
        """
        if not shift_indices:
            return self._normalize_segment(ts, 0, len(ts))
        
        corrected_ts = ts.copy()
        
        # Define segments
        segment_starts = [0] + shift_indices
        segment_ends = shift_indices + [len(ts)]
        
        # Normalize each segment
        for i in range(len(segment_starts)):
            segment_start = segment_starts[i]
            segment_end = segment_ends[i]
            
            segment_normalized = self._normalize_segment(
                corrected_ts, segment_start, segment_end
            )
            corrected_ts[segment_start:segment_end] = segment_normalized
        
        return corrected_ts
    
    def _normalize_segment(self, ts: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """Normalize a segment to zero mean, unit variance."""
        segment = ts[start_idx:end_idx]
        
        valid_values = segment[~np.isnan(segment)]
        if len(valid_values) == 0:
            return segment
        
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        if std_val == 0:
            return segment - mean_val
        
        normalized = (segment - mean_val) / std_val
        return normalized
    
    def correct_events(self, data: List[np.ndarray],
                      event_indices: Optional[List[int]] = None,
                      timestamps: Optional[List[np.ndarray]] = None,
                      **kwargs) -> List[np.ndarray]:
        """
        Apply event correction to a batch of time series.
        
        Args:
            data: List of time series arrays
            event_indices: Known event indices (if None, will detect)
            timestamps: Optional timestamp arrays
            **kwargs: Additional parameters
            
        Returns:
            List of corrected time series
        """
        if event_indices is None:
            # Detect events automatically
            if self.detection_method == 'global_shift':
                event_indices = self.detect_global_shifts(data, timestamps, **kwargs)
            else:
                event_indices = []
        
        if not event_indices:
            logger.info("No events detected, returning original data")
            return [ts.copy() for ts in data]
        
        # Apply correction to each series
        corrected_data = []
        
        for ts in data:
            if self.correction_method == 'baseline_adjustment':
                corrected_ts = self.correct_baseline_shift(ts, event_indices)
            elif self.correction_method == 'detrend':
                corrected_ts = self.correct_linear_detrend(ts, event_indices)
            elif self.correction_method == 'normalize':
                corrected_ts = self.correct_segment_normalization(ts, event_indices)
            else:
                raise ValueError(f"Unknown correction method: {self.correction_method}")
            
            corrected_data.append(corrected_ts)
        
        logger.info(f"Applied {self.correction_method} correction for {len(event_indices)} events")
        return corrected_data
    
    def analyze_correction_quality(self, original_data: List[np.ndarray],
                                 corrected_data: List[np.ndarray]) -> Dict:
        """
        Analyze the quality of event correction.
        
        Args:
            original_data: Original time series data
            corrected_data: Corrected time series data
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Calculate variance reduction
        original_vars = [np.nanvar(ts) for ts in original_data]
        corrected_vars = [np.nanvar(ts) for ts in corrected_data]
        
        metrics['mean_variance_reduction'] = (
            np.mean(original_vars) - np.mean(corrected_vars)
        ) / np.mean(original_vars)
        
        # Calculate correlation improvement (between series)
        if len(original_data) > 1:
            # Compute pairwise correlations
            original_corrs = []
            corrected_corrs = []
            
            for i in range(len(original_data)):
                for j in range(i + 1, len(original_data)):
                    ts1_orig = original_data[i]
                    ts2_orig = original_data[j]
                    ts1_corr = corrected_data[i]
                    ts2_corr = corrected_data[j]
                    
                    # Find common valid points
                    min_len = min(len(ts1_orig), len(ts2_orig))
                    ts1_orig = ts1_orig[:min_len]
                    ts2_orig = ts2_orig[:min_len]
                    ts1_corr = ts1_corr[:min_len]
                    ts2_corr = ts2_corr[:min_len]
                    
                    valid_mask = ~(np.isnan(ts1_orig) | np.isnan(ts2_orig) | 
                                  np.isnan(ts1_corr) | np.isnan(ts2_corr))
                    
                    if np.sum(valid_mask) > 10:
                        orig_corr = np.corrcoef(ts1_orig[valid_mask], ts2_orig[valid_mask])[0, 1]
                        corr_corr = np.corrcoef(ts1_corr[valid_mask], ts2_corr[valid_mask])[0, 1]
                        
                        if not np.isnan(orig_corr) and not np.isnan(corr_corr):
                            original_corrs.append(abs(orig_corr))
                            corrected_corrs.append(abs(corr_corr))
            
            if original_corrs:
                metrics['mean_correlation_change'] = np.mean(corrected_corrs) - np.mean(original_corrs)
        
        return metrics