#!/usr/bin/env python3
"""
Help connect to the actual sonic analysis project
This script will guide you through connecting to the correct Supabase instance
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

def main():
    print("=" * 80)
    print("SONIC ANALYSIS PROJECT CONNECTION HELPER")
    print("=" * 80)
    
    load_dotenv()
    
    print("\n1. CURRENT ENVIRONMENT VARIABLES:")
    print("=" * 50)
    
    # Check all relevant environment variables
    env_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'SONIC_SUPABASE_URL',
        'SONIC_SUPABASE_KEY',
        'DATABASE_URL',
        'DB_PASSWORD'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive info
            if 'key' in var.lower() or 'password' in var.lower():
                masked = value[:10] + "..." + value[-5:] if len(value) > 15 else "***"
                print(f"  {var}: {masked}")
            else:
                print(f"  {var}: {value}")
        else:
            print(f"  {var}: (not set)")
    
    print("\n2. CURRENT SUPABASE CONNECTION:")
    print("=" * 50)
    
    current_url = os.getenv('SUPABASE_URL')
    current_key = os.getenv('SUPABASE_KEY')
    
    try:
        client = create_client(current_url, current_key)
        print(f"✅ Connected to: {current_url}")
        
        # Try to get some basic info about this database
        try:
            result = client.table('drugs').select('*').limit(1).execute()
            print(f"✅ Can access drugs table: {len(result.data)} sample rows")
        except Exception as e:
            print(f"❌ Cannot access drugs table: {e}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    print("\n3. INSTRUCTIONS FOR SONIC ANALYSIS PROJECT:")
    print("=" * 50)
    
    print("""
To connect to the actual sonic analysis project with the 'processes dat table':

OPTION 1: If it's a different Supabase project
1. You need the Supabase URL and API key for the sonic analysis project
2. Add these to your .env file:
   SONIC_SUPABASE_URL=your-sonic-project-url
   SONIC_SUPABASE_KEY=your-sonic-project-key

OPTION 2: If it's the same project but different table names
1. The data might be in tables with different names
2. Let's search for tables that might contain the organoid data

OPTION 3: If data needs to be imported
1. You might need to import the processes dat table data first
2. We can help create import scripts once we know the data source
    """)
    
    print("\n4. SEARCHING FOR ORGANOID DATA IN CURRENT PROJECT:")
    print("=" * 50)
    
    # Search for any tables that might contain data
    possible_organoid_tables = [
        'processes_dat',
        'processes_dat_table', 
        'sonic_data',
        'organoid_processes',
        'raw_data',
        'sensor_data',
        'metabolic_data',
        'experiment_data'
    ]
    
    if current_url and current_key:
        client = create_client(current_url, current_key)
        
        for table_name in possible_organoid_tables:
            try:
                result = client.table(table_name).select('*').limit(1).execute()
                print(f"✅ Found table '{table_name}' with data!")
                
                if result.data:
                    df = pd.DataFrame(result.data)
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Sample data available")
                else:
                    print(f"   Table exists but is empty")
                    
            except Exception as e:
                if "does not exist" not in str(e):
                    print(f"❌ Error accessing '{table_name}': {str(e)[:50]}...")
    
    print("\n5. NEXT STEPS:")
    print("=" * 50)
    
    print("""
Please provide:

1. The correct Supabase URL for the sonic analysis project
2. The correct API key for the sonic analysis project  
3. OR confirm if the data is in the current project but with different table names
4. OR let me know if you need help importing the processes dat table data

Once you provide the correct connection details, I can:
- Analyze the real organoid data structure
- Adapt the embedding pipeline for your specific data
- Run experiments on your actual experimental results
    """)
    
    print("\n6. MANUAL CONNECTION TEST:")
    print("=" * 50)
    
    print("""
To manually test the sonic analysis project connection:

1. Update your .env file with the correct credentials:
   SONIC_SUPABASE_URL=your-sonic-url-here
   SONIC_SUPABASE_KEY=your-sonic-key-here

2. Run this script again to verify the connection

3. Or provide the credentials directly when prompted
    """)
    
    # Prompt for manual input if desired
    try:
        response = input("\nWould you like to try different credentials now? (y/n): ").lower().strip()
        
        if response == 'y':
            sonic_url = input("Enter Sonic Analysis Supabase URL: ").strip()
            sonic_key = input("Enter Sonic Analysis API Key: ").strip()
            
            if sonic_url and sonic_key:
                print(f"\nTesting connection to: {sonic_url}")
                
                try:
                    sonic_client = create_client(sonic_url, sonic_key)
                    
                    # Try accessing the processes dat table
                    table_variations = [
                        'processes_dat_table',
                        'processes dat table',
                        'processes_data',
                        'dat_table'
                    ]
                    
                    for table in table_variations:
                        try:
                            result = sonic_client.table(table).select('*').limit(1).execute()
                            print(f"✅ SUCCESS! Found '{table}' in sonic project!")
                            
                            if result.data:
                                df = pd.DataFrame(result.data)
                                print(f"   Columns: {list(df.columns)}")
                                print(f"   This appears to be your organoid data!")
                                
                                # Save credentials to .env
                                with open('.env', 'a') as f:
                                    f.write(f"\n# Sonic Analysis Project\n")
                                    f.write(f"SONIC_SUPABASE_URL={sonic_url}\n")
                                    f.write(f"SONIC_SUPABASE_KEY={sonic_key}\n")
                                print("✅ Credentials saved to .env file")
                                return
                                
                        except Exception as e:
                            print(f"❌ Table '{table}' not found: {str(e)[:50]}...")
                    
                    print("❌ Could not find processes dat table in the provided project")
                    
                except Exception as e:
                    print(f"❌ Connection failed: {e}")
            
    except KeyboardInterrupt:
        print("\n\nConnection test cancelled.")
    except:
        pass

if __name__ == "__main__":
    main()