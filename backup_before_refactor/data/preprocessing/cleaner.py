"""Time series data cleaning utilities."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class TimeSeriesCleaner:
    """
    Comprehensive time series cleaning for organoid data.
    
    Handles common issues in organoid oxygen time series including:
    - Missing values
    - Duplicate timestamps
    - Irregular sampling
    - Data quality assessment
    """
    
    def __init__(self, min_valid_ratio: float = 0.5, 
                 max_gap_hours: float = 6.0,
                 sampling_rate_hours: float = 1.0):
        """
        Initialize time series cleaner.
        
        Args:
            min_valid_ratio: Minimum ratio of valid (non-NaN) values required
            max_gap_hours: Maximum acceptable gap in hours
            sampling_rate_hours: Expected sampling rate in hours
        """
        self.min_valid_ratio = min_valid_ratio
        self.max_gap_hours = max_gap_hours
        self.sampling_rate_hours = sampling_rate_hours
        
    def assess_quality(self, ts: np.ndarray, timestamps: Optional[np.ndarray] = None) -> dict:
        """
        Assess time series data quality.
        
        Args:
            ts: Time series values
            timestamps: Timestamps (optional)
            
        Returns:
            Dictionary with quality metrics
        """
        quality = {}
        
        # Basic statistics
        total_points = len(ts)
        valid_points = np.sum(~np.isnan(ts))
        missing_ratio = 1 - (valid_points / total_points)
        
        quality.update({
            'total_points': total_points,
            'valid_points': valid_points,
            'missing_ratio': missing_ratio,
            'has_sufficient_data': missing_ratio <= (1 - self.min_valid_ratio)
        })
        
        if valid_points == 0:
            quality['is_usable'] = False
            return quality
        
        # Value statistics
        valid_values = ts[~np.isnan(ts)]
        quality.update({
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values),
            'range': np.ptp(valid_values)
        })
        
        # Timestamp analysis
        if timestamps is not None:
            quality.update(self._assess_temporal_quality(timestamps))
        
        # Missing value patterns
        quality.update(self._assess_missing_patterns(ts))
        
        # Overall usability
        quality['is_usable'] = (
            quality['has_sufficient_data'] and
            quality['std'] > 0 and  # Not constant
            quality.get('max_gap_hours', 0) <= self.max_gap_hours
        )
        
        return quality
    
    def _assess_temporal_quality(self, timestamps: np.ndarray) -> dict:
        """Assess temporal quality of timestamps."""
        quality = {}
        
        # Convert to hours if needed
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            
            # Assume timestamps are in seconds if very large numbers
            if np.median(time_diffs) > 7200:  # More than 2 hours in seconds
                time_diffs = time_diffs / 3600  # Convert to hours
            
            quality.update({
                'median_interval_hours': np.median(time_diffs),
                'mean_interval_hours': np.mean(time_diffs),
                'std_interval_hours': np.std(time_diffs),
                'min_interval_hours': np.min(time_diffs),
                'max_interval_hours': np.max(time_diffs),
                'max_gap_hours': np.max(time_diffs),
                'regular_sampling': np.std(time_diffs) < 0.1  # Less than 6 minutes variation
            })
        
        # Check for duplicates
        unique_timestamps = len(np.unique(timestamps))
        quality.update({
            'duplicate_timestamps': len(timestamps) - unique_timestamps,
            'has_duplicates': unique_timestamps < len(timestamps)
        })
        
        return quality
    
    def _assess_missing_patterns(self, ts: np.ndarray) -> dict:
        """Assess missing value patterns."""
        is_missing = np.isnan(ts)
        quality = {}
        
        if not np.any(is_missing):
            quality.update({
                'missing_runs': [],
                'max_missing_run': 0,
                'num_missing_runs': 0,
                'missing_at_start': False,
                'missing_at_end': False
            })
            return quality
        
        # Find consecutive missing runs
        missing_runs = []
        in_run = False
        run_start = 0
        
        for i, missing in enumerate(is_missing):
            if missing and not in_run:
                # Start of missing run
                in_run = True
                run_start = i
            elif not missing and in_run:
                # End of missing run
                in_run = False
                missing_runs.append((run_start, i - 1))
        
        # Handle case where series ends with missing values
        if in_run:
            missing_runs.append((run_start, len(ts) - 1))
        
        # Calculate run lengths
        run_lengths = [end - start + 1 for start, end in missing_runs]
        
        quality.update({
            'missing_runs': missing_runs,
            'max_missing_run': max(run_lengths) if run_lengths else 0,
            'num_missing_runs': len(missing_runs),
            'missing_at_start': is_missing[0],
            'missing_at_end': is_missing[-1]
        })
        
        return quality
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         time_col: str = 'timestamp',
                         value_col: str = 'value',
                         method: str = 'mean') -> pd.DataFrame:
        """
        Remove duplicate timestamps by aggregating values.
        
        Args:
            df: DataFrame with time series data
            time_col: Name of timestamp column
            value_col: Name of value column
            method: Aggregation method ('mean', 'median', 'first', 'last')
            
        Returns:
            DataFrame with duplicates removed
        """
        if method == 'mean':
            agg_func = 'mean'
        elif method == 'median':
            agg_func = 'median'
        elif method == 'first':
            agg_func = 'first'
        elif method == 'last':
            agg_func = 'last'
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Group by timestamp and aggregate
        grouped = df.groupby(time_col)[value_col].agg(agg_func).reset_index()
        
        logger.info(f"Removed {len(df) - len(grouped)} duplicate timestamps using {method}")
        
        return grouped
    
    def trim_series(self, ts: np.ndarray, 
                   remove_leading_nan: bool = True,
                   remove_trailing_nan: bool = True) -> Tuple[np.ndarray, slice]:
        """
        Trim leading/trailing NaN values from time series.
        
        Args:
            ts: Time series values
            remove_leading_nan: Whether to remove leading NaNs
            remove_trailing_nan: Whether to remove trailing NaNs
            
        Returns:
            Trimmed time series and slice object indicating the trim
        """
        valid_mask = ~np.isnan(ts)
        
        if not np.any(valid_mask):
            # All NaN
            return ts, slice(0, 0)
        
        # Find first and last valid indices
        valid_indices = np.where(valid_mask)[0]
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        
        # Determine trim boundaries
        start_idx = first_valid if remove_leading_nan else 0
        end_idx = last_valid + 1 if remove_trailing_nan else len(ts)
        
        trim_slice = slice(start_idx, end_idx)
        trimmed_ts = ts[trim_slice]
        
        logger.debug(f"Trimmed series from {len(ts)} to {len(trimmed_ts)} points")
        
        return trimmed_ts, trim_slice
    
    def filter_by_quality(self, data: List[np.ndarray],
                         timestamps: Optional[List[np.ndarray]] = None,
                         return_indices: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
        """
        Filter time series based on quality criteria.
        
        Args:
            data: List of time series arrays
            timestamps: Optional list of timestamp arrays
            return_indices: Whether to return indices of kept series
            
        Returns:
            Filtered data (and optionally indices)
        """
        kept_data = []
        kept_indices = []
        
        for i, ts in enumerate(data):
            ts_timestamps = timestamps[i] if timestamps else None
            quality = self.assess_quality(ts, ts_timestamps)
            
            if quality['is_usable']:
                kept_data.append(ts)
                kept_indices.append(i)
            else:
                logger.debug(f"Filtered out series {i}: {quality}")
        
        logger.info(f"Kept {len(kept_data)}/{len(data)} time series after quality filtering")
        
        if return_indices:
            return kept_data, kept_indices
        else:
            return kept_data
    
    def clean_time_series(self, ts: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         trim_nan: bool = True,
                         min_length: int = 10) -> Tuple[np.ndarray, bool]:
        """
        Apply full cleaning pipeline to a single time series.
        
        Args:
            ts: Time series values
            timestamps: Optional timestamps
            trim_nan: Whether to trim leading/trailing NaNs
            min_length: Minimum required length after cleaning
            
        Returns:
            Cleaned time series and success flag
        """
        original_length = len(ts)
        
        # Trim NaN values
        if trim_nan:
            ts, _ = self.trim_series(ts)
        
        # Check minimum length
        if len(ts) < min_length:
            logger.debug(f"Series too short after trimming: {len(ts)} < {min_length}")
            return ts, False
        
        # Assess quality
        quality = self.assess_quality(ts, timestamps)
        success = quality['is_usable']
        
        if success:
            logger.debug(f"Successfully cleaned series: {original_length} -> {len(ts)} points")
        else:
            logger.debug(f"Series failed quality check: {quality}")
        
        return ts, success
    
    def batch_clean(self, data: List[np.ndarray],
                   timestamps: Optional[List[np.ndarray]] = None,
                   **kwargs) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Apply cleaning pipeline to a batch of time series.
        
        Args:
            data: List of time series arrays
            timestamps: Optional list of timestamp arrays
            **kwargs: Additional arguments for clean_time_series
            
        Returns:
            List of cleaned time series and success flags
        """
        cleaned_data = []
        success_flags = []
        
        for i, ts in enumerate(data):
            ts_timestamps = timestamps[i] if timestamps else None
            cleaned_ts, success = self.clean_time_series(ts, ts_timestamps, **kwargs)
            
            cleaned_data.append(cleaned_ts)
            success_flags.append(success)
        
        n_success = sum(success_flags)
        logger.info(f"Cleaned {n_success}/{len(data)} time series successfully")
        
        return cleaned_data, success_flags