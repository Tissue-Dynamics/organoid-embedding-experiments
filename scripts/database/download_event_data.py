#!/usr/bin/env python3
"""
Download event data from Supabase and save locally for event-aware feature engineering.
This script connects to the database and downloads the complete event_table for local analysis.
"""

import os
import sys
import pandas as pd
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()


def download_event_data():
    """Download complete event data from Supabase and save locally."""
    print("Downloading event data from Supabase...")
    
    # Connect to database
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    # Parse connection
    parsed = urlparse(database_url)
    postgres_string = f"host={parsed.hostname} port={parsed.port} dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} sslmode=require"
    
    # Query to get all event data (convert UUIDs to strings)
    events_query = f"""
    SELECT 
        id::text as id,
        plate_id::text as plate_id,
        created_at,
        occurred_at,
        uploaded_by::text as uploaded_by,
        title,
        description,
        is_excluded,
        exclusion_reason
    FROM postgres_scan_pushdown('{postgres_string}', 'public', 'event_table')
    ORDER BY plate_id, occurred_at
    """
    
    try:
        events_df = conn.execute(events_query).fetchdf()
        print(f"  Downloaded {len(events_df):,} events")
        
        # Show event type distribution
        event_counts = events_df['title'].value_counts()
        print("\n  Event type distribution:")
        for event_type, count in event_counts.head(10).items():
            print(f"    {event_type}: {count:,}")
        
        # Convert timestamps to datetime
        events_df['created_at'] = pd.to_datetime(events_df['created_at'])
        events_df['occurred_at'] = pd.to_datetime(events_df['occurred_at'])
        
        # Create output directory
        output_dir = project_root / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet for efficient loading
        output_path = output_dir / "event_data.parquet"
        events_df.to_parquet(output_path, index=False)
        print(f"  Saved event data to: {output_path}")
        
        # Also save a CSV for human inspection
        csv_path = output_dir / "event_data.csv"
        events_df.to_csv(csv_path, index=False)
        print(f"  Saved event data (CSV) to: {csv_path}")
        
        return events_df, output_path
        
    except Exception as e:
        print(f"  Error downloading event data: {e}")
        return None, None


def analyze_critical_events(events_df):
    """Analyze the most important events for feature engineering."""
    print("\n=== Critical Event Analysis ===")
    
    # Focus on key events for feature engineering
    critical_events = ['Medium Change', 'Drugs Start', 'Drugs Added', 'Data Exclusion']
    
    for event_type in critical_events:
        event_subset = events_df[events_df['title'] == event_type]
        if len(event_subset) > 0:
            print(f"\n{event_type}:")
            print(f"  Count: {len(event_subset):,}")
            print(f"  Plates affected: {event_subset['plate_id'].nunique():,}")
            
            # Time distribution
            if len(event_subset) > 1:
                time_diff_hours = (event_subset['occurred_at'].max() - event_subset['occurred_at'].min()).total_seconds() / 3600
                print(f"  Time span: {time_diff_hours:.1f} hours")
            
            # Sample descriptions
            descriptions = event_subset['description'].dropna().unique()
            if len(descriptions) > 0:
                print(f"  Sample descriptions: {list(descriptions[:3])}")
    
    # Check for dosing events
    print("\n=== Dosing Event Analysis ===")
    dosing_events = events_df[events_df['title'].isin(['Drugs Start', 'Drugs Added'])]
    print(f"Total dosing events: {len(dosing_events):,}")
    print(f"Plates with dosing events: {dosing_events['plate_id'].nunique():,}")
    
    # Check plates per dosing event
    dosing_by_plate = dosing_events.groupby('plate_id').size()
    print(f"Dosing events per plate: {dosing_by_plate.mean():.1f} ± {dosing_by_plate.std():.1f}")
    
    # Check for media change patterns
    print("\n=== Media Change Analysis ===")
    media_events = events_df[events_df['title'] == 'Medium Change']
    print(f"Total media change events: {len(media_events):,}")
    print(f"Plates with media changes: {media_events['plate_id'].nunique():,}")
    
    # Media changes per plate
    media_by_plate = media_events.groupby('plate_id').size()
    print(f"Media changes per plate: {media_by_plate.mean():.1f} ± {media_by_plate.std():.1f}")
    print(f"Range: {media_by_plate.min()}-{media_by_plate.max()} media changes per plate")


def create_event_timeline_analysis(events_df, output_dir):
    """Create analysis of event timing for feature engineering."""
    print("\n=== Event Timeline Analysis ===")
    
    # Focus on plates with both dosing and media change events
    dosing_plates = set(events_df[events_df['title'].isin(['Drugs Start', 'Drugs Added'])]['plate_id'])
    media_plates = set(events_df[events_df['title'] == 'Medium Change']['plate_id'])
    complete_plates = dosing_plates.intersection(media_plates)
    
    print(f"Plates with dosing events: {len(dosing_plates)}")
    print(f"Plates with media change events: {len(media_plates)}")
    print(f"Plates with both dosing and media changes: {len(complete_plates)}")
    
    # Analyze timing for complete plates
    timeline_analysis = []
    
    for plate_id in list(complete_plates)[:10]:  # Sample 10 plates
        plate_events = events_df[events_df['plate_id'] == plate_id].sort_values('occurred_at')
        
        # Find dosing start
        dosing_events = plate_events[plate_events['title'].isin(['Drugs Start', 'Drugs Added'])]
        media_events = plate_events[plate_events['title'] == 'Medium Change']
        
        if len(dosing_events) > 0 and len(media_events) > 0:
            dose_start = dosing_events['occurred_at'].min()
            
            # Calculate time of each media change relative to dosing
            for _, media_event in media_events.iterrows():
                hours_since_dosing = (media_event['occurred_at'] - dose_start).total_seconds() / 3600
                
                timeline_analysis.append({
                    'plate_id': plate_id,
                    'event_type': 'Medium Change',
                    'hours_since_dosing': hours_since_dosing,
                    'occurred_at': media_event['occurred_at']
                })
    
    if timeline_analysis:
        timeline_df = pd.DataFrame(timeline_analysis)
        
        print(f"\nMedia change timing analysis ({len(timeline_df)} events):")
        print(f"  Mean time after dosing: {timeline_df['hours_since_dosing'].mean():.1f} ± {timeline_df['hours_since_dosing'].std():.1f} hours")
        print(f"  Range: {timeline_df['hours_since_dosing'].min():.1f} to {timeline_df['hours_since_dosing'].max():.1f} hours")
        
        # Save timeline analysis
        timeline_path = output_dir / "event_timeline_analysis.csv"
        timeline_df.to_csv(timeline_path, index=False)
        print(f"  Saved timeline analysis to: {timeline_path}")
        
        return timeline_df
    
    return None


def main():
    """Main pipeline for downloading and analyzing event data."""
    print("=== Event Data Download and Analysis ===\n")
    
    # Create output directory
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download event data
    events_df, events_path = download_event_data()
    
    if events_df is not None:
        # Analyze critical events
        analyze_critical_events(events_df)
        
        # Create timeline analysis
        timeline_df = create_event_timeline_analysis(events_df, output_dir)
        
        print(f"\n✅ Event data download complete!")
        print(f"📊 Events saved to: {events_path}")
        print(f"🔍 Key findings:")
        print(f"  - {len(events_df):,} total events across {events_df['plate_id'].nunique()} plates")
        print(f"  - {len(events_df[events_df['title'] == 'Medium Change']):,} media change events")
        print(f"  - {len(events_df[events_df['title'].isin(['Drugs Start', 'Drugs Added'])]):,} dosing events")
        print(f"  - Event types: {', '.join(events_df['title'].value_counts().head(5).index.tolist())}")
        
        return events_df
    else:
        print("❌ Failed to download event data")
        return None


if __name__ == "__main__":
    main()