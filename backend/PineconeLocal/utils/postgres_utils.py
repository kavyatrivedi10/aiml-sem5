import os
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import sql
import json
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
import torch
import time
from dotenv import find_dotenv, load_dotenv

# Load environment variables from the root .env file
root_env_path = find_dotenv()
load_dotenv(root_env_path)

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "fashion_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        raise

def create_table_if_not_exists(conn):
    """Create the products table with pgvector extension if it doesn't exist"""
    try:
        cur = conn.cursor()
        
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table with vector columns
        create_table_query = """
        CREATE TABLE IF NOT EXISTS products (
            id VARCHAR(255) PRIMARY KEY,
            dense_vector vector(1024),
            sparse_vector JSONB,
            product_display_name VARCHAR(500),
            brand_name VARCHAR(255),
            master_category VARCHAR(100),
            sub_category VARCHAR(100),
            article_type VARCHAR(100),
            gender VARCHAR(50),
            color VARCHAR(100),
            season VARCHAR(50),
            occasion VARCHAR(100),
            is_jewellery BOOLEAN,
            style_image TEXT,
            pattern VARCHAR(100),
            sleeve_styling VARCHAR(100),
            sleeve_length VARCHAR(100),
            fabric VARCHAR(100),
            neck VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        
        # Create indexes for better query performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dense_vector ON products USING ivfflat (dense_vector vector_cosine_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sparse_vector ON products USING gin (sparse_vector);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_master_category ON products (master_category);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sub_category ON products (sub_category);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_article_type ON products (article_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_color ON products (color);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_brand_name ON products (brand_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gender ON products (gender);")
        
        conn.commit()
        cur.close()
        print("Table created/verified successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error creating table: {e}")
        raise

def setup_postgres():
    """Setup PostgreSQL connection and return connection, model, and bm25"""
    start_time = time.time()
    
    try:
        print('Connecting to PostgreSQL...')
        conn = get_db_connection()
        print('Connection established.')
        
        print('Creating table if not exists...')
        create_table_if_not_exists(conn)
        print('Table setup completed.')
        
        print('Getting CLIP and BM25 model...')
        model, bm25 = get_clip_and_bm25_model()
        print('Models obtained:')
        print('---- Model:', model)
        print('---- BM25:', bm25)
        
        print('Setup completed.')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
        return conn, model, bm25
    
    except Exception as e:
        print(f"Error setting up PostgreSQL: {e}")
        raise

def get_clip_and_bm25_model():
    """Get CLIP and BM25 models"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
        bm25 = BM25Encoder()
        return model, bm25
    except Exception as e:
        print(f"Error getting CLIP and BM25 model: {e}")
        raise

