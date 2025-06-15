#!/usr/bin/env python3
"""Check database schema and find actual table names."""

import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_schema():
    """Check actual database schema."""
    from data.loaders.supabase_loader import SupabaseDataLoader
    
    loader = SupabaseDataLoader()
    logger.info("✅ Connected to Supabase")
    
    # Try to find any tables by testing common table names more broadly
    test_names = [
        # Basic variations
        'test', 'public', 'users', 'auth',
        # Oxygen/organoid related
        'oxygen', 'o2', 'organoid', 'liver', 'hepatocyte',
        # Time series related  
        'ts', 'series', 'readings', 'values',
        # Experiment related
        'experiment', 'exp', 'samples', 'wells',
        # Generic data tables
        'table1', 'main', 'primary',
        # Common Supabase defaults
        'profiles', 'todos', 'notes'
    ]
    
    found_tables = []
    
    for name in test_names:
        try:
            result = loader.client.table(name).select('*').limit(1).execute()
            if hasattr(result, 'data'):
                logger.info(f"✅ Found table: {name}")
                found_tables.append(name)
                
                if result.data:
                    sample = result.data[0]
                    logger.info(f"   Sample columns: {list(sample.keys())}")
                else:
                    logger.info(f"   Table {name} is empty")
                    
        except Exception as e:
            # Most will fail, that's expected
            pass
    
    if not found_tables:
        logger.warning("🚫 No tables found!")
        logger.info("This could mean:")
        logger.info("1. The database is completely empty")
        logger.info("2. The API key doesn't have read permissions")
        logger.info("3. Tables are in a different schema")
        logger.info("4. RLS (Row Level Security) is blocking access")
        
        # Test basic connection by trying to create a simple table
        logger.info("\n🧪 Testing write permissions...")
        try:
            # Try to insert into a test table (this will fail if table doesn't exist)
            test_result = loader.client.table('test_connection').insert({'test': 'value'}).execute()
            logger.info("✅ Write test succeeded - can create data")
        except Exception as e:
            logger.info(f"❌ Write test failed: {e}")
            
    else:
        logger.info(f"\n📋 Found {len(found_tables)} accessible tables:")
        for table in found_tables:
            logger.info(f"  - {table}")
            
    return found_tables

def check_rls_and_permissions():
    """Check if Row Level Security might be blocking access."""
    logger.info("\n🔒 Checking permissions and RLS...")
    
    from data.loaders.supabase_loader import SupabaseDataLoader
    loader = SupabaseDataLoader()
    
    # Test if we can access system information
    try:
        # This should work if we have basic read access
        result = loader.client.table('plate_table').select('*').limit(0).execute()
        logger.info("✅ Can access plate_table structure (even if empty)")
        
        # Check if we can see the table structure by trying to get count
        count_result = loader.client.table('plate_table').select('*', count='exact').execute()
        logger.info(f"✅ plate_table row count: {count_result.count}")
        
    except Exception as e:
        logger.error(f"❌ Cannot access plate_table: {e}")

if __name__ == "__main__":
    logger.info("🔍 Checking database schema and permissions...")
    
    try:
        found_tables = check_database_schema()
        check_rls_and_permissions()
        
        if not found_tables:
            logger.info("\n💡 SUGGESTIONS:")
            logger.info("1. Check if data has been uploaded to the database")
            logger.info("2. Verify API key has correct permissions") 
            logger.info("3. Check if Row Level Security (RLS) is enabled")
            logger.info("4. Ensure you're connecting to the right project")
            logger.info("5. Try uploading some test data first")
            
        else:
            logger.info("\n✅ Database connection works!")
            logger.info("We found some tables, but not the expected organoid data tables.")
            logger.info("You may need to:")
            logger.info("1. Upload your organoid data")
            logger.info("2. Create the expected table structure")
            logger.info("3. Update the data loader to match your table names")
            
    except Exception as e:
        logger.error(f"Schema check failed: {e}")
        raise