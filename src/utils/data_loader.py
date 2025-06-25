"""
Data Loading Utilities

PURPOSE:
    Provides reusable functions for loading oxygen consumption data,
    media change events, and drug metadata from the database.
    Used by analysis scripts to avoid code duplication.

CLASSES:
    DataLoader: Main class for database connections and data loading
    
METHODS:
    load_oxygen_data(): Load raw oxygen consumption time series
    load_media_events(): Load media change event data
    load_drug_metadata(): Load drug information including DILI scores
    load_well_metadata(): Load well-to-drug mapping
"""

import pandas as pd
import numpy as np
import duckdb
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse


class DataLoader:
    """Handles database connections and data loading operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize data loader with database connection.
        
        Args:
            database_url: PostgreSQL connection string. If None, uses DATABASE_URL env var.
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not provided or set in environment")
        
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection via DuckDB."""
        self.conn = duckdb.connect(':memory:')
        self.conn.execute(f"ATTACH '{self.database_url}' AS db (TYPE postgres)")
    
    def load_oxygen_data(self, plate_ids: Optional[list] = None, 
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load oxygen consumption time series data.
        
        Args:
            plate_ids: List of plate IDs to load. If None, loads all.
            limit: Limit number of plates (for testing)
            
        Returns:
            DataFrame with columns: plate_id, well_id, drug, concentration,
                                  elapsed_hours, o2, timestamp
        """
        plate_filter = ""
        if plate_ids:
            plate_list = "', '".join(plate_ids)
            plate_filter = f"WHERE p.plate_id IN ('{plate_list}')"
        elif limit:
            plate_filter = f"""WHERE p.plate_id IN (
                SELECT DISTINCT plate_id FROM db.public.processed_data 
                LIMIT {limit}
            )"""
        
        query = f"""
        WITH well_map AS (
            SELECT DISTINCT
                plate_id,
                well_number,
                drug,
                concentration
            FROM db.public.well_map_data
        )
        SELECT 
            p.plate_id,
            p.plate_id || '_' || p.well_number as well_id,
            p.well_number,
            COALESCE(w.drug, 'Unknown') as drug,
            COALESCE(w.concentration, 0) as concentration,
            DATE_PART('epoch', p.timestamp - MIN(p.timestamp) OVER (PARTITION BY p.plate_id)) / 3600.0 as elapsed_hours,
            p.median_o2 as o2,
            p.timestamp
        FROM db.public.processed_data p
        LEFT JOIN well_map w ON p.plate_id = w.plate_id AND p.well_number = w.well_number
        WHERE (p.is_excluded = false OR p.is_excluded IS NULL)
        {' AND ' + plate_filter.replace('WHERE ', '') if plate_filter else ''}
        ORDER BY p.plate_id, p.well_number, p.timestamp
        """
        
        return self.conn.execute(query).df()
    
    def load_media_events(self) -> pd.DataFrame:
        """
        Load media change events from database.
        
        Returns:
            DataFrame with columns: plate_id, well_number, event_time, event_type
        """
        query = """
        SELECT 
            plate_id,
            well_number,
            event_time,
            event_type
        FROM db.public.event_table
        WHERE event_type LIKE '%media%'
        ORDER BY plate_id, well_number, event_time
        """
        
        return self.conn.execute(query).df()
    
    def load_drug_metadata(self) -> pd.DataFrame:
        """
        Load drug metadata including DILI scores.
        
        Returns:
            DataFrame with drug information and risk scores
        """
        query = """
        SELECT 
            drug,
            dili_risk_score,
            drug_class,
            mechanism_of_action
        FROM db.public.drugs
        WHERE dili_risk_score IS NOT NULL
        """
        
        try:
            return self.conn.execute(query).df()
        except:
            # Fallback if some columns don't exist
            query = """
            SELECT 
                drug,
                dili_risk_score
            FROM db.public.drugs
            WHERE dili_risk_score IS NOT NULL
            """
            return self.conn.execute(query).df()
    
    def load_well_metadata(self, plate_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Load well-to-drug mapping.
        
        Args:
            plate_ids: List of plate IDs to load. If None, loads all.
            
        Returns:
            DataFrame with well metadata
        """
        plate_filter = ""
        if plate_ids:
            plate_list = "', '".join(plate_ids)
            plate_filter = f"WHERE plate_id IN ('{plate_list}')"
        
        query = f"""
        SELECT DISTINCT
            plate_id,
            well_number,
            drug,
            concentration,
            plate_id || '_' || well_number as well_id
        FROM db.public.well_map_data
        {plate_filter}
        ORDER BY plate_id, well_number
        """
        
        return self.conn.execute(query).df()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()