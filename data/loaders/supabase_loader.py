"""
Supabase data loader for oxygen time series data
"""
import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class SupabaseDataLoader:
    """Load oxygen time series data from Supabase"""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client"""
        load_dotenv()
        
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided")
            
        self.client: Client = create_client(self.url, self.key)
        logger.info("Connected to Supabase")
    
    def get_plates(self, limit: int = 100) -> pd.DataFrame:
        """Get list of available plates"""
        response = self.client.table('plate_table').select('*').limit(limit).execute()
        return pd.DataFrame(response.data)
    
    def get_well_map(self, plate_id: str) -> pd.DataFrame:
        """Get well mapping data for a plate"""
        response = self.client.table('well_map_data')\
            .select('*')\
            .eq('plate_id', plate_id)\
            .eq('is_excluded', False)\
            .execute()
        return pd.DataFrame(response.data)
    
    def get_time_series_data(self, 
                           plate_id: str, 
                           well_numbers: Optional[List[int]] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get oxygen time series data for specified wells
        
        Args:
            plate_id: Plate identifier
            well_numbers: List of well numbers to fetch (None for all)
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with columns: timestamp, well_number, median_o2
        """
        query = self.client.table('processed_data')\
            .select('timestamp,well_number,median_o2,is_excluded')\
            .eq('plate_id', plate_id)\
            .eq('is_excluded', False)
        
        if well_numbers:
            query = query.in_('well_number', well_numbers)
            
        if start_time:
            query = query.gte('timestamp', start_time.isoformat())
            
        if end_time:
            query = query.lte('timestamp', end_time.isoformat())
        
        # Fetch in batches due to size
        all_data = []
        limit = 10000
        offset = 0
        
        while True:
            response = query.limit(limit).offset(offset).execute()
            data = response.data
            
            if not data:
                break
                
            all_data.extend(data)
            offset += limit
            
            if len(data) < limit:
                break
                
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['well_number', 'timestamp'])
        
        logger.info(f"Loaded {len(df)} data points for plate {plate_id}")
        return df
    
    def get_events(self, plate_id: str) -> pd.DataFrame:
        """Get experimental events (e.g., media changes)"""
        response = self.client.table('event_table')\
            .select('*')\
            .eq('plate_id', plate_id)\
            .eq('is_excluded', False)\
            .execute()
        
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['occurred_at'] = pd.to_datetime(df['occurred_at'])
        return df
    
    def get_drugs_info(self, drug_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get drug information"""
        query = self.client.table('drugs').select('*')
        
        if drug_names:
            query = query.in_('drug', drug_names)
            
        response = query.execute()
        return pd.DataFrame(response.data)
    
    def get_full_experiment_data(self, plate_id: str = None, limit: Optional[int] = None, time_filter: Optional[str] = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Get all data for an experiment
        
        Returns:
            Dictionary with keys: 'time_series', 'well_map', 'events', 'plate_info'
        """
        # Get plate info
        plate_info = self.client.table('plate_table')\
            .select('*')\
            .eq('id', plate_id)\
            .single()\
            .execute()
        
        # Get well mapping
        well_map = self.get_well_map(plate_id)
        
        # Get time series data
        time_series = self.get_time_series_data(plate_id)
        
        # Apply limit if specified
        if limit is not None and len(time_series) > limit:
            time_series = time_series.head(limit)
        
        # Get events
        events = self.get_events(plate_id)
        
        # Get drug info for drugs in this plate
        unique_drugs = well_map['drug'].unique()
        drugs = self.get_drugs_info(list(unique_drugs))
        
        return {
            'plate_info': pd.DataFrame([plate_info.data]),
            'well_map': well_map,
            'time_series': time_series,
            'events': events,
            'drugs': drugs
        }
    
    def prepare_time_series_matrix(self, 
                                 data: pd.DataFrame,
                                 well_map: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convert time series data to matrix format
        
        Args:
            data: Time series DataFrame
            well_map: Well mapping DataFrame
            
        Returns:
            matrix: (n_wells, n_timepoints) array
            metadata: DataFrame with well metadata
        """
        # Pivot to wide format
        pivot = data.pivot(index='timestamp', columns='well_number', values='median_o2')
        
        # Ensure all wells from well_map are included
        all_wells = sorted(well_map['well_number'].unique())
        missing_wells = set(all_wells) - set(pivot.columns)
        for well in missing_wells:
            pivot[well] = np.nan
            
        # Sort columns
        pivot = pivot[sorted(pivot.columns)]
        
        # Convert to numpy array (wells x time)
        matrix = pivot.values.T
        
        # Create metadata DataFrame
        metadata = well_map.set_index('well_number').loc[sorted(pivot.columns)]
        metadata = metadata.reset_index()
        
        return matrix, metadata
    
    def get_control_wells(self, well_map: pd.DataFrame) -> List[int]:
        """Identify control wells (no drug or vehicle control)"""
        controls = well_map[
            (well_map['drug'].str.lower().isin(['control', 'dmso', 'vehicle', 'none'])) |
            (well_map['concentration'] == 0)
        ]
        return controls['well_number'].tolist()


def load_experiment_batch(loader: SupabaseDataLoader, 
                         plate_ids: List[str],
                         max_workers: int = 4) -> Dict[str, Dict]:
    """Load multiple experiments in parallel"""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    results = {}
    lock = threading.Lock()
    
    def load_plate(plate_id):
        try:
            data = loader.get_full_experiment_data(plate_id)
            with lock:
                results[plate_id] = data
            logger.info(f"Loaded plate {plate_id}")
        except Exception as e:
            logger.error(f"Error loading plate {plate_id}: {e}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(load_plate, plate_ids)
    
    return results
