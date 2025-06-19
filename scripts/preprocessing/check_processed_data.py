#!/usr/bin/env python3
"""Check processed_data table structure"""

import duckdb
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres")

conn = duckdb.connect()
conn.execute(f"ATTACH '{DATABASE_URL}' AS postgres (TYPE postgres)")

print("Checking processed_data table structure:")
cols = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'processed_data'").fetchall()
print("\nColumns in processed_data:")
for col in cols:
    print(f"  - {col[0]}")

# Also check a sample
print("\nSample data:")
sample = conn.execute("SELECT * FROM postgres.processed_data LIMIT 5").fetchdf()
print(sample.columns.tolist())
print(sample)

conn.close()