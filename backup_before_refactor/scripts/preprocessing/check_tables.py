#!/usr/bin/env python3
"""Check available tables in database"""

import duckdb
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

print("Available tables:")
# Try different approaches
try:
    tables = conn.execute("SELECT * FROM postgres.information_schema.tables").fetchdf()
    print(f"Found {len(tables)} tables")
    print(tables[['table_schema', 'table_name']].head(20))
except Exception as e:
    print(f"Error with information_schema: {e}")

print("\nTrying direct access to known tables:")
known_tables = ['processed_data', 'well_map_data', 'drugs', 'media_change_events', 'oxygen_time_series_v2']
for table in known_tables:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM postgres.{table}").fetchone()[0]
        print(f"  ✓ {table}: {count:,} rows")
    except Exception as e:
        print(f"  ✗ {table}: {str(e).split('!')[0]}")

conn.close()