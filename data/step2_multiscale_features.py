#!/usr/bin/env python3
"""
Step 2: Multi-Timescale Feature Extraction
Rolling window catch22 and SAX features at 24h, 48h, 96h resolutions.
"""

import os
import sys
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()


class MultiScaleFeatureExtractor:
    """Extract features at multiple temporal resolutions with rolling windows."""
    
    def __init__(self, window_sizes=[24, 48, 96], overlap_ratio=0.5):
        """
        Initialize feature extractor.
        
        Args:
            window_sizes: List of window sizes in hours [24, 48, 96]
            overlap_ratio: Overlap between consecutive windows (0.5 = 50% overlap)
        """
        self.window_sizes = window_sizes
        self.overlap_ratio = overlap_ratio
        self.step_sizes = {ws: int(ws * (1 - overlap_ratio)) for ws in window_sizes}
        
        print(f"Initialized MultiScaleFeatureExtractor:")
        print(f"  Window sizes: {window_sizes} hours")
        print(f"  Overlap ratio: {overlap_ratio} ({overlap_ratio*100}%)")
        print(f"  Step sizes: {self.step_sizes} hours")
    
    def create_rolling_windows(self, time_series_data, window_size_hours):
        """
        Create rolling windows from time series data.
        
        Args:
            time_series_data: DataFrame with columns ['elapsed_hours', 'o2_percent']
            window_size_hours: Window size in hours
            
        Returns:
            List of window dictionaries with metadata
        """
        if len(time_series_data) == 0:
            return []
        
        # Sort by time
        data = time_series_data.sort_values('elapsed_hours').copy()
        step_size = self.step_sizes[window_size_hours]
        
        # Calculate window start times
        max_time = data['elapsed_hours'].max()
        window_starts = np.arange(0, max_time - window_size_hours + 1, step_size)
        
        windows = []
        for i, start_time in enumerate(window_starts):
            end_time = start_time + window_size_hours
            
            # Extract data in window
            window_mask = (data['elapsed_hours'] >= start_time) & (data['elapsed_hours'] < end_time)
            window_data = data[window_mask].copy()
            
            if len(window_data) < 5:  # Minimum points for meaningful features
                continue
            
            window_info = {
                'window_number': i,
                'start_time': start_time,
                'end_time': end_time,
                'window_size': window_size_hours,
                'n_points': len(window_data),
                'o2_values': window_data['o2_percent'].values,
                'time_values': window_data['elapsed_hours'].values,
                'duration_actual': window_data['elapsed_hours'].max() - window_data['elapsed_hours'].min()
            }
            
            windows.append(window_info)
        
        return windows
    
    def extract_basic_features(self, window_info):
        """Extract basic statistical features from a window."""
        o2_values = window_info['o2_values']
        
        if len(o2_values) < 2:
            return {}
        
        features = {
            'mean': np.mean(o2_values),
            'std': np.std(o2_values),
            'min': np.min(o2_values),
            'max': np.max(o2_values),
            'median': np.median(o2_values),
            'q25': np.percentile(o2_values, 25),
            'q75': np.percentile(o2_values, 75),
            'range': np.max(o2_values) - np.min(o2_values),
            'cv': np.std(o2_values) / np.mean(o2_values) if np.mean(o2_values) != 0 else np.inf,
            'skewness': self._safe_skewness(o2_values),
            'kurtosis': self._safe_kurtosis(o2_values),
            'slope': self._calculate_trend_slope(window_info['time_values'], o2_values),
            'n_points': len(o2_values),
            'duration_actual': window_info['duration_actual']
        }
        
        return features
    
    def _safe_skewness(self, values):
        """Calculate skewness with error handling."""
        try:
            from scipy.stats import skew
            return skew(values)
        except:
            return 0.0
    
    def _safe_kurtosis(self, values):
        """Calculate kurtosis with error handling."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(values)
        except:
            return 0.0
    
    def _calculate_trend_slope(self, time_values, o2_values):
        """Calculate linear trend slope."""
        if len(time_values) < 2:
            return 0.0
        
        try:
            # Simple linear regression slope
            time_centered = time_values - np.mean(time_values)
            o2_centered = o2_values - np.mean(o2_values)
            slope = np.sum(time_centered * o2_centered) / np.sum(time_centered ** 2)
            return slope
        except:
            return 0.0
    
    def extract_catch22_features(self, window_info):
        """Extract catch22 features from a window."""
        o2_values = window_info['o2_values']
        
        if len(o2_values) < 10:  # catch22 needs sufficient data points
            return {'catch22_error': 'insufficient_data', 'catch22_n_points': len(o2_values)}
        
        try:
            import pycatch22
            
            # Convert to numpy array and ensure float64 dtype
            o2_array = np.array(o2_values, dtype=np.float64)
            
            # Extract all catch22 features
            features_dict = pycatch22.catch22_all(o2_array)
            
            # Create feature dictionary
            catch22_features = {}
            
            if isinstance(features_dict, dict) and 'names' in features_dict and 'values' in features_dict:
                feature_names = features_dict['names']
                feature_values = features_dict['values']
                
                for name, value in zip(feature_names, feature_values):
                    # Handle NaN/inf values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    catch22_features[f"catch22_{name}"] = float(value)
            else:
                # Fallback: use basic features
                basic_features = self.extract_basic_features(window_info)
                catch22_features = {f"catch22_{k}": v for k, v in basic_features.items()}
                catch22_features['catch22_fallback'] = True
            
            catch22_features['catch22_success'] = True
            return catch22_features
            
        except Exception as e:
            print(f"    Warning: catch22 failed ({str(e)}), using basic features")
            # Fallback to basic features
            basic_features = self.extract_basic_features(window_info)
            fallback_features = {f"catch22_{k}": v for k, v in basic_features.items()}
            fallback_features['catch22_error'] = str(e)
            fallback_features['catch22_success'] = False
            return fallback_features
    
    def extract_sax_features(self, window_info, hierarchical=True):
        """Extract hierarchical SAX features from a window."""
        o2_values = window_info['o2_values']
        
        if len(o2_values) < 8:  # Minimum for meaningful SAX
            return {'sax_error': 'insufficient_data'}
        
        all_sax_features = {}
        
        # Hierarchical SAX configurations
        if hierarchical:
            sax_configs = [
                {'n_symbols': 4, 'alphabet_size': 3, 'level': 'coarse'},
                {'n_symbols': 8, 'alphabet_size': 4, 'level': 'medium'},
                {'n_symbols': 16, 'alphabet_size': 6, 'level': 'fine'}
            ]
        else:
            sax_configs = [{'n_symbols': 8, 'alphabet_size': 4, 'level': 'default'}]
        
        for config in sax_configs:
            try:
                sax_features = self._extract_single_sax(o2_values, **config)
                
                # Add level prefix to feature names
                level_prefix = f"sax_{config['level']}_"
                for key, value in sax_features.items():
                    all_sax_features[level_prefix + key] = value
                    
            except Exception as e:
                all_sax_features[f"sax_{config['level']}_error"] = str(e)
        
        return all_sax_features
    
    def _extract_single_sax(self, o2_values, n_symbols, alphabet_size, level):
        """Extract SAX features for a single configuration."""
        # Normalize values
        o2_norm = (o2_values - np.mean(o2_values)) / (np.std(o2_values) + 1e-8)
        
        # Reduce to n_symbols via averaging (PAA - Piecewise Aggregate Approximation)
        if len(o2_norm) < n_symbols:
            # If fewer points than symbols, use simple downsampling
            segments = o2_norm
        else:
            segment_size = len(o2_norm) / n_symbols
            segments = []
            
            for i in range(n_symbols):
                start_idx = int(i * segment_size)
                end_idx = int((i + 1) * segment_size) if i < n_symbols - 1 else len(o2_norm)
                if end_idx > start_idx:
                    segment_mean = np.mean(o2_norm[start_idx:end_idx])
                    segments.append(segment_mean)
        
        if len(segments) == 0:
            return {'error': 'no_segments'}
        
        # Convert to symbols using Gaussian breakpoints
        breakpoints = self._get_breakpoints(alphabet_size)
        symbols = []
        for segment in segments:
            symbol = sum(segment > bp for bp in breakpoints)
            symbols.append(symbol)
        
        # Create SAX features
        sax_features = {
            'string': ''.join(chr(ord('a') + s) for s in symbols),
            'entropy': self._calculate_entropy(symbols),
            'n_unique': len(set(symbols)),
            'transitions': sum(1 for i in range(1, len(symbols)) if symbols[i] != symbols[i-1]),
            'longest_run': self._longest_run(symbols),
            'complexity': self._lempel_ziv_complexity(symbols),
            'n_symbols': n_symbols,
            'alphabet_size': alphabet_size
        }
        
        # Add pattern-specific features
        pattern_features = self._extract_pattern_features(symbols, alphabet_size)
        sax_features.update(pattern_features)
        
        return sax_features
    
    def _get_breakpoints(self, alphabet_size):
        """Get Gaussian breakpoints for SAX discretization."""
        if alphabet_size == 3:
            return [-0.43, 0.43]
        elif alphabet_size == 4:
            return [-0.67, 0, 0.67]
        elif alphabet_size == 5:
            return [-0.84, -0.25, 0.25, 0.84]
        elif alphabet_size == 6:
            return [-1.07, -0.43, 0, 0.43, 1.07]
        else:
            # General case: equi-probable breakpoints
            from scipy.stats import norm
            return [norm.ppf(i / alphabet_size) for i in range(1, alphabet_size)]
    
    def _longest_run(self, symbols):
        """Calculate longest run of identical symbols."""
        if len(symbols) == 0:
            return 0
        
        max_run = 1
        current_run = 1
        
        for i in range(1, len(symbols)):
            if symbols[i] == symbols[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run
    
    def _lempel_ziv_complexity(self, symbols):
        """Calculate Lempel-Ziv complexity of symbol sequence."""
        if len(symbols) <= 1:
            return len(symbols)
        
        # Simple LZ complexity estimation
        sequence = ''.join(map(str, symbols))
        complexity = 0
        i = 0
        
        while i < len(sequence):
            substring = sequence[i]
            j = i + 1
            
            while j <= len(sequence) and substring in sequence[:i]:
                if j < len(sequence):
                    substring += sequence[j]
                j += 1
            
            complexity += 1
            i = j - 1 if substring in sequence[:i] else j
        
        return complexity
    
    def _extract_pattern_features(self, symbols, alphabet_size):
        """Extract pattern-based features from symbol sequence."""
        if len(symbols) == 0:
            return {}
        
        features = {}
        
        # Symbol frequency features
        from collections import Counter
        symbol_counts = Counter(symbols)
        
        features['dominant_symbol'] = symbol_counts.most_common(1)[0][0]
        features['dominant_freq'] = symbol_counts.most_common(1)[0][1] / len(symbols)
        features['uniformity'] = len(symbol_counts) / alphabet_size
        
        # Trend features
        trend_symbols = []
        for i in range(1, len(symbols)):
            if symbols[i] > symbols[i-1]:
                trend_symbols.append(1)  # increasing
            elif symbols[i] < symbols[i-1]:
                trend_symbols.append(-1)  # decreasing
            else:
                trend_symbols.append(0)  # stable
        
        if trend_symbols:
            features['trend_increasing'] = sum(1 for t in trend_symbols if t == 1) / len(trend_symbols)
            features['trend_decreasing'] = sum(1 for t in trend_symbols if t == -1) / len(trend_symbols)
            features['trend_stable'] = sum(1 for t in trend_symbols if t == 0) / len(trend_symbols)
        else:
            features['trend_increasing'] = 0
            features['trend_decreasing'] = 0
            features['trend_stable'] = 1
        
        return features
    
    def _calculate_entropy(self, symbols):
        """Calculate entropy of symbol sequence."""
        try:
            from collections import Counter
            counts = Counter(symbols)
            total = len(symbols)
            entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
            return entropy
        except:
            return 0.0
    
    def process_well(self, well_data, well_id, baseline_data=None):
        """
        Process a single well to extract multi-scale features.
        
        Args:
            well_data: DataFrame with time series for one well
            well_id: Identifier for the well
            baseline_data: Optional baseline period data
            
        Returns:
            List of feature dictionaries
        """
        print(f"  Processing well {well_id} ({len(well_data)} points)")
        
        all_features = []
        
        # Process each window size
        for window_size in self.window_sizes:
            windows = self.create_rolling_windows(well_data, window_size)
            
            print(f"    {window_size}h windows: {len(windows)} created")
            
            for window_info in windows:
                # Extract different feature types
                basic_features = self.extract_basic_features(window_info)
                catch22_features = self.extract_catch22_features(window_info)
                sax_features = self.extract_sax_features(window_info)
                
                # Combine all features
                feature_record = {
                    'well_id': well_id,
                    'window_size': window_size,
                    'window_number': window_info['window_number'],
                    'start_time': window_info['start_time'],
                    'end_time': window_info['end_time'],
                    'n_points': window_info['n_points'],
                    'duration_actual': window_info['duration_actual'],
                    **basic_features,
                    **catch22_features,
                    **sax_features
                }
                
                all_features.append(feature_record)
        
        return all_features


def load_step1_data():
    """Load the Step 1 quality assessment results."""
    data_path = project_root / "results" / "data" / "step1_quality_assessment_all_plates.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Step 1 data not found: {data_path}")
    
    step1_data = pd.read_parquet(data_path)
    print(f"Loaded Step 1 data: {len(step1_data)} wells from {step1_data['plate_id'].nunique()} plates")
    
    return step1_data


def load_time_series_data(plate_id, well_id):
    """Load time series data for a specific well using DuckDB."""
    # Setup DuckDB connection
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # Extract plate_id and well_number from well_id
    well_number = int(well_id.split('_')[-1])
    
    query = f"""
    SELECT 
        timestamp,
        median_o2 as o2_percent
    FROM postgres_scan('{postgres_string}', 'public', 'processed_data')
    WHERE plate_id::text = '{plate_id}' 
    AND well_number = {well_number}
    AND is_excluded = false
    ORDER BY timestamp
    """
    
    data = conn.execute(query).fetchdf()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate elapsed hours
    min_timestamp = data['timestamp'].min()
    data['elapsed_hours'] = (data['timestamp'] - min_timestamp).dt.total_seconds() / 3600
    
    conn.close()
    return data


def main():
    """Main Step 2 pipeline for multi-scale feature extraction."""
    print("=== Step 2: Multi-Timescale Feature Extraction ===\n")
    
    # Load Step 1 results
    step1_data = load_step1_data()
    
    # Initialize feature extractor
    extractor = MultiScaleFeatureExtractor(
        window_sizes=[24, 48, 96],
        overlap_ratio=0.5
    )
    
    # Process a sample of wells first (for testing)
    sample_wells = step1_data.head(10)  # Start with 10 wells
    
    print(f"\nProcessing {len(sample_wells)} sample wells...")
    
    all_features = []
    
    for idx, well_row in sample_wells.iterrows():
        try:
            # Load time series data
            well_data = load_time_series_data(well_row['plate_id'], well_row['well_id'])
            
            if len(well_data) < 10:  # Skip wells with too little data
                print(f"  Skipping {well_row['well_id']} - insufficient data ({len(well_data)} points)")
                continue
            
            # Extract features
            well_features = extractor.process_well(
                well_data, 
                well_row['well_id'],
                baseline_data=None  # TODO: Add baseline data if needed
            )
            
            all_features.extend(well_features)
            
        except Exception as e:
            print(f"  Error processing {well_row['well_id']}: {e}")
            continue
    
    if len(all_features) == 0:
        print("❌ No features extracted")
        return
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"\n=== Feature Extraction Results ===")
    print(f"Total feature records: {len(features_df)}")
    print(f"Wells processed: {features_df['well_id'].nunique()}")
    print(f"Window sizes: {sorted(features_df['window_size'].unique())}")
    print(f"Feature columns: {len([c for c in features_df.columns if c not in ['well_id', 'window_size', 'window_number', 'start_time', 'end_time']])}")
    
    # Save results
    output_dir = project_root / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "step2_multiscale_features_sample.parquet"
    features_df.to_parquet(output_path, index=False)
    
    csv_path = output_dir / "step2_multiscale_features_sample.csv"
    features_df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Step 2 Sample Complete!")
    print(f"📊 Features saved to: {output_path}")
    print(f"📄 CSV saved to: {csv_path}")
    print(f"🔍 Next: Install catch22 and implement full feature extraction")
    
    return features_df


if __name__ == "__main__":
    main()