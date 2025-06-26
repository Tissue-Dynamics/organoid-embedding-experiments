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
from typing import Optional, Dict, Tuple, List
from urllib.parse import urlparse


class DataLoader:
    """Handles database connections and data loading operations."""
    
    conn: Optional[duckdb.DuckDBPyConnection]
    
    def __init__(self, database_url: Optional[str] = None, use_local: Optional[bool] = None):
        """
        Initialize data loader with database connection.
        
        Args:
            database_url: PostgreSQL connection string. If None, uses DATABASE_URL env var.
            use_local: If True, use local DuckDB file. If False, use remote PostgreSQL.
                      If None (default), use local if available, otherwise remote.
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.conn = None
        self.is_local = False
        
        # Determine which database to use
        local_db_path = Path("data/database/organoid_data.duckdb")
        local_sample_path = Path("data/database/organoid_data_sample.duckdb")
        
        if use_local is True:
            # Explicitly use local
            if local_db_path.exists():
                self._connect_local(local_db_path)
            elif local_sample_path.exists():
                print("Note: Using sample database. Full database not found.")
                self._connect_local(local_sample_path)
            else:
                raise ValueError("Local database file not found. Run export_to_duckdb.py first.")
        elif use_local is False:
            # Explicitly use remote
            if not self.database_url:
                raise ValueError("DATABASE_URL not provided or set in environment")
            self._connect_remote()
        else:
            # Auto-detect (default behavior)
            if local_db_path.exists():
                print("Using local database file.")
                self._connect_local(local_db_path)
            elif local_sample_path.exists():
                print("Using local sample database file.")
                self._connect_local(local_sample_path)
            elif self.database_url:
                print("Using remote PostgreSQL database.")
                self._connect_remote()
            else:
                raise ValueError("No database available. Set DATABASE_URL or create local database.")
    
    def _connect_local(self, db_path: Path):
        """Connect to local DuckDB file."""
        self.conn = duckdb.connect(str(db_path))
        self.is_local = True
        assert self.conn is not None
    
    def _connect_remote(self):
        """Establish database connection via DuckDB to PostgreSQL."""
        self.conn = duckdb.connect(':memory:')
        assert self.conn is not None
        self.conn.execute(f"ATTACH '{self.database_url}' AS db (TYPE postgres)")
        self.is_local = False
    
    def _execute_and_convert(self, query: str) -> pd.DataFrame:
        """Execute query and convert UUIDs to strings for compatibility."""
        assert self.conn is not None, "Database connection not established"
        
        # Adjust query for local vs remote database
        if self.is_local:
            # Remove db.public. prefix for local database
            query = query.replace("db.public.", "")
        
        df = self.conn.execute(query).df()
        
        # Convert UUID columns to strings
        for col in df.columns:
            if df[col].dtype == 'object' and len(df) > 0:
                # Check if first non-null value is UUID
                first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if first_val and hasattr(first_val, 'hex'):  # UUID type
                    df[col] = df[col].astype(str)
        
        return df
    
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
        
        return self._execute_and_convert(query)
    
    def load_media_events(self) -> pd.DataFrame:
        """
        Load media change events from database.
        
        Returns:
            DataFrame with columns: plate_id, occurred_at, title, description
        """
        query = """
        SELECT 
            plate_id,
            occurred_at as event_time,
            title,
            description
        FROM db.public.event_table
        WHERE title = 'Medium Change'
           AND is_excluded = false
        ORDER BY plate_id, occurred_at
        """
        
        return self._execute_and_convert(query)
    
    def load_all_events(self) -> pd.DataFrame:
        """
        Load all events from database.
        
        Returns:
            DataFrame with event data
        """
        query = """
        SELECT 
            id,
            plate_id,
            occurred_at,
            title,
            description,
            is_excluded
        FROM db.public.event_table
        WHERE is_excluded = false
        ORDER BY plate_id, occurred_at
        """
        
        return self._execute_and_convert(query)
    
    def load_drug_metadata(self) -> pd.DataFrame:
        """
        Load drug metadata including DILI scores.
        
        Returns:
            DataFrame with drug information and risk scores
        """
        query = """
        SELECT 
            drug,
            severity as dili_risk_score,
            binary_dili,
            likelihood,
            dili,
            dili_risk_category,
            COALESCE(atc, '') as drug_class,
            COALESCE(metabolism_cyp_enzymes, '') as mechanism_of_action
        FROM db.public.drugs
        WHERE drug IS NOT NULL
        """
        
        return self._execute_and_convert(query)
    
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
        
        return self._execute_and_convert(query)
    
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
    
    def load_processed_data_summary(self) -> pd.DataFrame:
        """
        Load summary statistics for processed_data by plate.
        
        Returns:
            DataFrame with plate-level statistics
        """
        query = """
        SELECT 
            plate_id,
            COUNT(*) as record_count,
            COUNT(DISTINCT well_number) as unique_wells,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            ROUND(EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600.0, 2) as duration_hours,
            COUNT(DISTINCT cycle_num) as total_cycles,
            MIN(median_o2) as min_o2,
            MAX(median_o2) as max_o2,
            AVG(median_o2) as avg_o2,
            SUM(CASE WHEN is_excluded = true THEN 1 ELSE 0 END) as excluded_records,
            ROUND(100.0 * SUM(CASE WHEN is_excluded = true THEN 1 ELSE 0 END) / COUNT(*), 2) as exclusion_rate
        FROM db.public.processed_data
        GROUP BY plate_id
        ORDER BY start_time
        """
        
        return self._execute_and_convert(query)
    
    def load_excluded_data(self, plate_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load excluded data records with reasons.
        
        Args:
            plate_ids: Optional list of plate IDs to filter
            
        Returns:
            DataFrame with excluded records
        """
        plate_filter = ""
        if plate_ids:
            plate_list = "', '".join(plate_ids)
            plate_filter = f"AND plate_id IN ('{plate_list}')"
        
        query = f"""
        SELECT 
            plate_id,
            well_number,
            timestamp,
            median_o2,
            exclusion_reason,
            excluded_by,
            excluded_at
        FROM db.public.processed_data
        WHERE is_excluded = true
        {plate_filter}
        ORDER BY plate_id, well_number, timestamp
        """
        
        return self._execute_and_convert(query)
    
    def load_cycle_statistics(self, plate_id: str) -> pd.DataFrame:
        """
        Load cycle-level statistics for a specific plate.
        
        Args:
            plate_id: Plate ID to analyze
            
        Returns:
            DataFrame with cycle statistics
        """
        query = f"""
        SELECT 
            cycle_num,
            MIN(cycle_time_stamp) as cycle_start,
            COUNT(DISTINCT well_number) as wells_measured,
            COUNT(*) as measurements,
            MIN(median_o2) as min_o2,
            MAX(median_o2) as max_o2,
            AVG(median_o2) as avg_o2,
            STDDEV(median_o2) as std_o2,
            SUM(CASE WHEN is_excluded = true THEN 1 ELSE 0 END) as excluded_count
        FROM db.public.processed_data
        WHERE plate_id = '{plate_id}'
        GROUP BY cycle_num
        ORDER BY cycle_num
        """
        
        return self._execute_and_convert(query)
    
    def validate_processed_data(self) -> Dict[str, any]:
        """
        Validate processed_data quality and completeness.
        
        Returns:
            Dictionary with validation results
        """
        query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT plate_id) as unique_plates,
            COUNT(DISTINCT plate_id || '_' || well_number) as unique_wells,
            MIN(timestamp) as earliest_timestamp,
            MAX(timestamp) as latest_timestamp,
            MIN(median_o2) as min_o2,
            MAX(median_o2) as max_o2,
            COUNT(DISTINCT cycle_num) as unique_cycles,
            SUM(CASE WHEN is_excluded = true THEN 1 ELSE 0 END) as excluded_count,
            COUNT(DISTINCT exclusion_reason) as unique_exclusion_reasons,
            COUNT(CASE WHEN median_o2 < -50 THEN 1 END) as extreme_low_o2,
            COUNT(CASE WHEN median_o2 > 150 THEN 1 END) as extreme_high_o2,
            COUNT(CASE WHEN median_o2 IS NULL THEN 1 END) as null_o2_count
        FROM db.public.processed_data
        """
        
        assert self.conn is not None, "Database connection not established"
        result = self.conn.execute(query).df()
        
        return result.iloc[0].to_dict()
    
    def export_all_data(self, output_dir: str = "data/extracted", formats: List[str] = ["parquet"]) -> Dict[str, str]:
        """
        Export all data to local files for offline analysis.
        
        Args:
            output_dir: Directory to save exported files
            formats: List of formats to export ("parquet", "csv", "json")
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        from pathlib import Path
        import time
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        print(f"Exporting all data to {output_path}")
        print("=" * 50)
        
        # Define datasets to export
        datasets = [
            ("drugs", self.load_drug_metadata),
            ("events", self.load_all_events),
            ("well_metadata", self.load_well_metadata),
            ("processed_summary", self.load_processed_data_summary),
            ("event_summary", self.load_event_summary),
        ]
        
        # Add sample oxygen data (limited due to size)
        try:
            datasets.append(("oxygen_sample", lambda: self.load_oxygen_data(limit=2)))
        except:
            print("Warning: Could not export oxygen sample data")
        
        # Add media change intervals
        try:
            datasets.append(("media_intervals", lambda: self.analyze_event_intervals('Medium Change')))
        except:
            print("Warning: Could not export media change intervals")
        
        for dataset_name, loader_func in datasets:
            try:
                print(f"Exporting {dataset_name}...")
                start_time = time.time()
                
                df = loader_func()
                
                for fmt in formats:
                    if fmt == "parquet":
                        try:
                            import pyarrow
                            file_path = output_path / f"{dataset_name}.parquet"
                            df.to_parquet(file_path, index=False)
                            exported_files[f"{dataset_name}.parquet"] = str(file_path)
                        except ImportError:
                            print(f"  Warning: pyarrow not available, skipping parquet for {dataset_name}")
                    
                    elif fmt == "csv":
                        file_path = output_path / f"{dataset_name}.csv"
                        df.to_csv(file_path, index=False)
                        exported_files[f"{dataset_name}.csv"] = str(file_path)
                    
                    elif fmt == "json":
                        file_path = output_path / f"{dataset_name}.json"
                        df.to_json(file_path, orient="records", date_format="iso")
                        exported_files[f"{dataset_name}.json"] = str(file_path)
                
                elapsed = time.time() - start_time
                print(f"  ✓ {dataset_name}: {len(df)} rows in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  ✗ {dataset_name}: Error - {e}")
        
        # Export validation summaries
        try:
            validation_data = {
                "processed_validation": self.validate_processed_data(),
                "event_validation": self.validate_event_data()
            }
            
            for name, data in validation_data.items():
                df = pd.DataFrame([data])
                for fmt in formats:
                    if fmt == "parquet":
                        try:
                            file_path = output_path / f"{name}.parquet"
                            df.to_parquet(file_path, index=False)
                            exported_files[f"{name}.parquet"] = str(file_path)
                        except ImportError:
                            pass
                    elif fmt == "csv":
                        file_path = output_path / f"{name}.csv"
                        df.to_csv(file_path, index=False)
                        exported_files[f"{name}.csv"] = str(file_path)
            
            print(f"  ✓ Validation summaries exported")
            
        except Exception as e:
            print(f"  ✗ Validation summaries: Error - {e}")
        
        print(f"\nExported {len(exported_files)} files to {output_path}")
        
        return exported_files
    
    def load_event_summary(self) -> pd.DataFrame:
        """
        Load summary of all event types.
        
        Returns:
            DataFrame with event type statistics
        """
        query = """
        SELECT 
            title as event_type,
            COUNT(*) as count,
            COUNT(DISTINCT plate_id) as plates_affected,
            MIN(occurred_at) as first_occurrence,
            MAX(occurred_at) as last_occurrence,
            COUNT(DISTINCT uploaded_by) as unique_uploaders,
            SUM(CASE WHEN is_excluded = true THEN 1 ELSE 0 END) as excluded_count
        FROM db.public.event_table
        GROUP BY title
        ORDER BY count DESC
        """
        
        return self._execute_and_convert(query)
    
    def load_plate_events(self, plate_id: str) -> pd.DataFrame:
        """
        Load all events for a specific plate.
        
        Args:
            plate_id: Plate ID to get events for
            
        Returns:
            DataFrame with plate events in chronological order
        """
        query = f"""
        SELECT 
            id,
            plate_id,
            occurred_at,
            title as event_type,
            description,
            uploaded_by,
            created_at,
            is_excluded
        FROM db.public.event_table
        WHERE plate_id = '{plate_id}'
        ORDER BY occurred_at
        """
        
        return self._execute_and_convert(query)
    
    def load_events_by_type(self, event_types: List[str]) -> pd.DataFrame:
        """
        Load events of specific types.
        
        Args:
            event_types: List of event types to load
            
        Returns:
            DataFrame with filtered events
        """
        type_list = "', '".join(event_types)
        query = f"""
        SELECT 
            id,
            plate_id,
            occurred_at,
            title as event_type,
            description,
            uploaded_by,
            is_excluded
        FROM db.public.event_table
        WHERE title IN ('{type_list}')
          AND is_excluded = false
        ORDER BY occurred_at
        """
        
        return self._execute_and_convert(query)
    
    def load_event_timeline(self, plate_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load event timeline with experimental context.
        
        Args:
            plate_ids: Optional list of plate IDs to filter
            
        Returns:
            DataFrame with events and experimental phases
        """
        plate_filter = ""
        if plate_ids:
            plate_list = "', '".join(plate_ids)
            plate_filter = f"WHERE e.plate_id IN ('{plate_list}')"
        
        query = f"""
        WITH plate_experiments AS (
            SELECT 
                plate_id,
                MIN(timestamp) as experiment_start,
                MAX(timestamp) as experiment_end
            FROM db.public.processed_data
            GROUP BY plate_id
        )
        SELECT 
            e.id,
            e.plate_id,
            e.occurred_at,
            e.title as event_type,
            e.description,
            p.experiment_start,
            p.experiment_end,
            EXTRACT(EPOCH FROM (e.occurred_at - p.experiment_start)) / 3600.0 as hours_since_start,
            CASE 
                WHEN e.occurred_at < p.experiment_start THEN 'pre-experiment'
                WHEN e.occurred_at > p.experiment_end THEN 'post-experiment'
                ELSE 'during-experiment'
            END as event_phase
        FROM db.public.event_table e
        LEFT JOIN plate_experiments p ON e.plate_id = p.plate_id
        WHERE e.is_excluded = false
        {' AND ' + plate_filter.replace('WHERE ', '') if plate_filter else ''}
        ORDER BY e.plate_id, e.occurred_at
        """
        
        return self._execute_and_convert(query)
    
    def validate_event_data(self) -> Dict[str, any]:
        """
        Validate event data quality and completeness.
        
        Returns:
            Dictionary with validation results
        """
        query = """
        WITH event_stats AS (
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT plate_id) as unique_plates,
                COUNT(DISTINCT title) as unique_event_types,
                COUNT(DISTINCT uploaded_by) as unique_uploaders,
                MIN(occurred_at) as earliest_event,
                MAX(occurred_at) as latest_event,
                SUM(CASE WHEN is_excluded = true THEN 1 ELSE 0 END) as excluded_events,
                SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END) as events_without_description
            FROM db.public.event_table
        ),
        critical_events AS (
            SELECT 
                SUM(CASE WHEN title = 'Drugs Start' THEN 1 ELSE 0 END) as drug_start_events,
                SUM(CASE WHEN title = 'Medium Change' THEN 1 ELSE 0 END) as media_change_events,
                SUM(CASE WHEN title = 'Experiment End' THEN 1 ELSE 0 END) as experiment_end_events,
                SUM(CASE WHEN title = 'Data Exclusion' THEN 1 ELSE 0 END) as exclusion_events
            FROM db.public.event_table
            WHERE is_excluded = false
        )
        SELECT * FROM event_stats, critical_events
        """
        
        assert self.conn is not None, "Database connection not established"
        result = self.conn.execute(query).df()
        
        return result.iloc[0].to_dict()
    
    def analyze_event_intervals(self, event_type: str = 'Medium Change') -> pd.DataFrame:
        """
        Analyze time intervals between specific events.
        
        Args:
            event_type: Type of event to analyze intervals for
            
        Returns:
            DataFrame with interval statistics by plate
        """
        query = f"""
        WITH event_intervals AS (
            SELECT 
                plate_id,
                occurred_at,
                LAG(occurred_at) OVER (PARTITION BY plate_id ORDER BY occurred_at) as prev_event,
                EXTRACT(EPOCH FROM (occurred_at - LAG(occurred_at) OVER (PARTITION BY plate_id ORDER BY occurred_at))) / 3600.0 as hours_between
            FROM db.public.event_table
            WHERE title = '{event_type}'
              AND is_excluded = false
        )
        SELECT 
            plate_id,
            COUNT(*) as event_count,
            AVG(hours_between) as avg_interval_hours,
            MIN(hours_between) as min_interval_hours,
            MAX(hours_between) as max_interval_hours,
            STDDEV(hours_between) as std_interval_hours
        FROM event_intervals
        WHERE hours_between IS NOT NULL
        GROUP BY plate_id
        HAVING COUNT(*) > 1
        ORDER BY plate_id
        """
        
        return self._execute_and_convert(query)