"""Time series interpolation utilities for missing data."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
import logging
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class TimeSeriesInterpolator:
    """
    Interpolation utilities for handling missing data in organoid time series.
    
    Provides various interpolation methods suitable for biological time series
    with irregular sampling and missing values.
    """
    
    def __init__(self, method: str = 'linear', 
                 max_gap_size: Optional[int] = None,
                 preserve_trends: bool = True):
        """
        Initialize time series interpolator.
        
        Args:
            method: Interpolation method ('linear', 'cubic', 'spline', 'polynomial')
            max_gap_size: Maximum gap size to interpolate (None = no limit)
            preserve_trends: Whether to preserve local trends during interpolation
        """
        self.method = method
        self.max_gap_size = max_gap_size
        self.preserve_trends = preserve_trends
        
        # Validate method
        valid_methods = ['linear', 'cubic', 'spline', 'polynomial', 'forward_fill', 'backward_fill']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def _find_gaps(self, ts: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find consecutive gaps (NaN regions) in time series.
        
        Args:
            ts: Time series with potential NaN values
            
        Returns:
            List of (start_idx, end_idx) tuples for each gap
        """
        is_nan = np.isnan(ts)
        gaps = []
        
        in_gap = False
        gap_start = 0
        
        for i, is_missing in enumerate(is_nan):
            if is_missing and not in_gap:
                # Start of gap
                in_gap = True
                gap_start = i
            elif not is_missing and in_gap:
                # End of gap
                in_gap = False
                gaps.append((gap_start, i - 1))
        
        # Handle case where series ends with gap
        if in_gap:
            gaps.append((gap_start, len(ts) - 1))
        
        return gaps
    
    def _interpolate_gap(self, ts: np.ndarray, gap_start: int, gap_end: int,
                        timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolate a single gap in the time series.
        
        Args:
            ts: Time series data
            gap_start: Start index of gap
            gap_end: End index of gap
            timestamps: Optional timestamp array
            
        Returns:
            Time series with gap filled
        """
        ts_filled = ts.copy()
        gap_size = gap_end - gap_start + 1
        
        # Check if gap is too large
        if self.max_gap_size and gap_size > self.max_gap_size:
            logger.debug(f"Gap size {gap_size} exceeds maximum {self.max_gap_size}, skipping")
            return ts_filled
        
        # Find surrounding valid values
        valid_before = None
        valid_after = None
        
        # Look backwards for valid value
        for i in range(gap_start - 1, -1, -1):
            if not np.isnan(ts[i]):
                valid_before = i
                break
        
        # Look forwards for valid value
        for i in range(gap_end + 1, len(ts)):
            if not np.isnan(ts[i]):
                valid_after = i
                break
        
        # Determine interpolation strategy based on available data
        if valid_before is None and valid_after is None:
            # No valid data available
            return ts_filled
        
        elif valid_before is None:
            # Only forward data - backward fill
            ts_filled[gap_start:gap_end + 1] = ts[valid_after]
            return ts_filled
        
        elif valid_after is None:
            # Only backward data - forward fill
            ts_filled[gap_start:gap_end + 1] = ts[valid_before]
            return ts_filled
        
        # Both sides available - interpolate
        if self.method == 'linear':
            ts_filled[gap_start:gap_end + 1] = self._linear_interpolate(
                ts, gap_start, gap_end, valid_before, valid_after, timestamps
            )
        
        elif self.method == 'cubic':
            ts_filled[gap_start:gap_end + 1] = self._cubic_interpolate(
                ts, gap_start, gap_end, valid_before, valid_after, timestamps
            )
        
        elif self.method == 'spline':
            ts_filled[gap_start:gap_end + 1] = self._spline_interpolate(
                ts, gap_start, gap_end, valid_before, valid_after, timestamps
            )
        
        elif self.method == 'polynomial':
            ts_filled[gap_start:gap_end + 1] = self._polynomial_interpolate(
                ts, gap_start, gap_end, valid_before, valid_after, timestamps
            )
        
        elif self.method == 'forward_fill':
            ts_filled[gap_start:gap_end + 1] = ts[valid_before]
        
        elif self.method == 'backward_fill':
            ts_filled[gap_start:gap_end + 1] = ts[valid_after]
        
        return ts_filled
    
    def _linear_interpolate(self, ts: np.ndarray, gap_start: int, gap_end: int,
                           valid_before: int, valid_after: int,
                           timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear interpolation between two points."""
        if timestamps is not None:
            # Use actual timestamps for interpolation
            x_points = [timestamps[valid_before], timestamps[valid_after]]
            y_points = [ts[valid_before], ts[valid_after]]
            x_interp = timestamps[gap_start:gap_end + 1]
        else:
            # Use indices
            x_points = [valid_before, valid_after]
            y_points = [ts[valid_before], ts[valid_after]]
            x_interp = np.arange(gap_start, gap_end + 1)
        
        return np.interp(x_interp, x_points, y_points)
    
    def _cubic_interpolate(self, ts: np.ndarray, gap_start: int, gap_end: int,
                          valid_before: int, valid_after: int,
                          timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Cubic interpolation using surrounding points."""
        # Get more surrounding points for cubic interpolation
        context_size = 5
        start_idx = max(0, valid_before - context_size)
        end_idx = min(len(ts), valid_after + context_size + 1)
        
        # Extract valid points in the context
        context_indices = []
        context_values = []
        
        for i in range(start_idx, end_idx):
            if not np.isnan(ts[i]):
                context_indices.append(i)
                context_values.append(ts[i])
        
        if len(context_values) < 2:
            # Fall back to linear
            return self._linear_interpolate(ts, gap_start, gap_end, valid_before, valid_after, timestamps)
        
        if timestamps is not None:
            x_known = [timestamps[i] for i in context_indices]
            x_interp = timestamps[gap_start:gap_end + 1]
        else:
            x_known = context_indices
            x_interp = np.arange(gap_start, gap_end + 1)
        
        try:
            f = interp1d(x_known, context_values, kind='cubic', 
                        bounds_error=False, fill_value='extrapolate')
            return f(x_interp)
        except:
            # Fall back to linear if cubic fails
            return self._linear_interpolate(ts, gap_start, gap_end, valid_before, valid_after, timestamps)
    
    def _spline_interpolate(self, ts: np.ndarray, gap_start: int, gap_end: int,
                           valid_before: int, valid_after: int,
                           timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Spline interpolation using smoothing splines."""
        # Get surrounding context
        context_size = 10
        start_idx = max(0, valid_before - context_size)
        end_idx = min(len(ts), valid_after + context_size + 1)
        
        # Extract valid points
        context_indices = []
        context_values = []
        
        for i in range(start_idx, end_idx):
            if not np.isnan(ts[i]):
                context_indices.append(i)
                context_values.append(ts[i])
        
        if len(context_values) < 4:
            # Fall back to cubic
            return self._cubic_interpolate(ts, gap_start, gap_end, valid_before, valid_after, timestamps)
        
        if timestamps is not None:
            x_known = [timestamps[i] for i in context_indices]
            x_interp = timestamps[gap_start:gap_end + 1]
        else:
            x_known = context_indices
            x_interp = np.arange(gap_start, gap_end + 1)
        
        try:
            # Use smoothing parameter based on gap size
            smoothing = min(len(context_values) * 0.1, 1.0)
            spline = UnivariateSpline(x_known, context_values, s=smoothing)
            return spline(x_interp)
        except:
            # Fall back to cubic
            return self._cubic_interpolate(ts, gap_start, gap_end, valid_before, valid_after, timestamps)
    
    def _polynomial_interpolate(self, ts: np.ndarray, gap_start: int, gap_end: int,
                               valid_before: int, valid_after: int,
                               timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Polynomial interpolation with trend preservation."""
        # Use local polynomial fitting
        context_size = 8
        start_idx = max(0, valid_before - context_size)
        end_idx = min(len(ts), valid_after + context_size + 1)
        
        # Extract valid points
        context_indices = []
        context_values = []
        
        for i in range(start_idx, end_idx):
            if not np.isnan(ts[i]):
                context_indices.append(i)
                context_values.append(ts[i])
        
        if len(context_values) < 3:
            return self._linear_interpolate(ts, gap_start, gap_end, valid_before, valid_after, timestamps)
        
        # Fit polynomial (degree based on available points)
        degree = min(3, len(context_values) - 1)
        
        if timestamps is not None:
            x_known = np.array([timestamps[i] for i in context_indices])
            x_interp = timestamps[gap_start:gap_end + 1]
        else:
            x_known = np.array(context_indices)
            x_interp = np.arange(gap_start, gap_end + 1)
        
        try:
            coeffs = np.polyfit(x_known, context_values, degree)
            return np.polyval(coeffs, x_interp)
        except:
            return self._linear_interpolate(ts, gap_start, gap_end, valid_before, valid_after, timestamps)
    
    def interpolate(self, ts: np.ndarray, 
                   timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolate missing values in a time series.
        
        Args:
            ts: Time series with potential missing values
            timestamps: Optional timestamp array
            
        Returns:
            Time series with interpolated values
        """
        if not np.any(np.isnan(ts)):
            return ts.copy()
        
        # Find all gaps
        gaps = self._find_gaps(ts)
        
        if not gaps:
            return ts.copy()
        
        # Interpolate each gap
        ts_interpolated = ts.copy()
        
        for gap_start, gap_end in gaps:
            ts_interpolated = self._interpolate_gap(
                ts_interpolated, gap_start, gap_end, timestamps
            )
        
        return ts_interpolated
    
    def batch_interpolate(self, data: List[np.ndarray],
                         timestamps: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Interpolate missing values in a batch of time series.
        
        Args:
            data: List of time series arrays
            timestamps: Optional list of timestamp arrays
            
        Returns:
            List of interpolated time series
        """
        interpolated_data = []
        
        for i, ts in enumerate(data):
            ts_timestamps = timestamps[i] if timestamps else None
            interpolated_ts = self.interpolate(ts, ts_timestamps)
            interpolated_data.append(interpolated_ts)
        
        logger.info(f"Interpolated {len(data)} time series using {self.method}")
        
        return interpolated_data
    
    def interpolate_to_regular_grid(self, ts: np.ndarray, timestamps: np.ndarray,
                                   target_timestamps: np.ndarray) -> np.ndarray:
        """
        Interpolate time series to a regular timestamp grid.
        
        Args:
            ts: Time series values
            timestamps: Original timestamps
            target_timestamps: Target timestamp grid
            
        Returns:
            Time series interpolated to target grid
        """
        # Remove NaN values
        valid_mask = ~np.isnan(ts)
        if not np.any(valid_mask):
            return np.full_like(target_timestamps, np.nan)
        
        valid_times = timestamps[valid_mask]
        valid_values = ts[valid_mask]
        
        if len(valid_values) == 1:
            # Single point - constant interpolation
            return np.full_like(target_timestamps, valid_values[0])
        
        # Choose interpolation method
        if self.method in ['linear', 'forward_fill', 'backward_fill']:
            kind = 'linear'
            fill_value = 'extrapolate'
        elif self.method in ['cubic', 'spline']:
            kind = 'cubic'
            fill_value = 'extrapolate'
        else:
            kind = 'linear'
            fill_value = 'extrapolate'
        
        try:
            f = interp1d(valid_times, valid_values, kind=kind,
                        bounds_error=False, fill_value=fill_value)
            interpolated = f(target_timestamps)
            
            # Handle extrapolation limits
            if fill_value != 'extrapolate':
                # Set out-of-bounds values to NaN
                mask = (target_timestamps < valid_times[0]) | (target_timestamps > valid_times[-1])
                interpolated[mask] = np.nan
            
            return interpolated
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, using linear fallback")
            return np.interp(target_timestamps, valid_times, valid_values)