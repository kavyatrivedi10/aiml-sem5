"""
Database setup script for PostgreSQL with pgvector extension.
Run this script once to initialize the database and create necessary tables.
"""
import os
from dotenv import find_dotenv, load_dotenv
from PineconeLocal.utils.postgres_utils import get_db_connection, create_table_if_not_exists

# Load environment variables
root_env_path = find_dotenv()
load_dotenv(root_env_path)

def main():
    """Setup database tables and extensions"""
    print("Setting up PostgreSQL database...")
    try:
        conn = get_db_connection()
        print("✓ Connected to PostgreSQL")
        
        create_table_if_not_exists(conn)
        print("✓ Tables and indexes created successfully")
        
        # Verify table exists
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'products'
        """)
        table_exists = cur.fetchone()[0] > 0
        
        if table_exists:
            print("✓ Products table verified")
        else:
            print("✗ Products table not found")
        
        cur.close()
        conn.close()
        print("\nDatabase setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        print("\nPlease ensure:")
        print("1. PostgreSQL is installed and running")
        print("2. pgvector extension is installed")
        print("3. Database credentials in .env file are correct")
        raise

if __name__ == "__main__":
    main()

