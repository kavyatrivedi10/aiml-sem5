from PineconeLocal.utils.postgres_utils import setup_postgres, get_db_connection
import pickle
from PineconeLocal.utils.filters import build_hard_filters
import os
from PineconeLocal.utils.user_bio_data.userBio import current_user_bio_data
import random
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

def hybrid_scale(dense, sparse, alpha: float):
    """Scale dense and sparse vectors for hybrid search"""
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # For PostgreSQL, we'll handle scaling in the query
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def get_bio_data(hard_filters):
    """Add user bio data to filters"""
    hard_filters['gender'] = {"$in": ["unisex", current_user_bio_data.gender]}
    print(hard_filters)
    return hard_filters

def build_sql_where_clause(hard_filters):
    """Convert Pinecone-style filters to SQL WHERE clause"""
    if not hard_filters:
        return "", []
    
    conditions = []
    params = []
    
    # List of valid column names to prevent SQL injection
    valid_columns = ['occasion', 'article_type', 'color', 'brand_name', 'gender', 
                     'is_jewellery', 'master_category', 'product_display_name', 
                     'season', 'style_image', 'sub_category']
    
    for key, value in hard_filters.items():
        # Validate column name to prevent SQL injection
        if key not in valid_columns:
            continue
            
        if isinstance(value, dict):
            # Handle operators like $eq, $in
            if "$eq" in value:
                conditions.append(f'"{key}" = %s')
                params.append(value["$eq"])
            elif "$in" in value:
                placeholders = ",".join(["%s"] * len(value["$in"]))
                conditions.append(f'"{key}" IN ({placeholders})')
                params.extend(value["$in"])
        else:
            conditions.append(f'"{key}" = %s')
            params.append(value)
    
    where_clause = " AND ".join(conditions) if conditions else ""
    return where_clause, params

def calculate_sparse_score(sparse_vector, stored_sparse):
    """Calculate BM25 score from sparse vectors"""
    if not stored_sparse or not sparse_vector:
        return 0.0
    
    score = 0.0
    query_indices = set(sparse_vector.get('indices', []))
    query_values = {idx: val for idx, val in zip(sparse_vector.get('indices', []), sparse_vector.get('values', []))}
    
    stored_indices = stored_sparse.get('indices', [])
    stored_values = stored_sparse.get('values', [])
    stored_dict = {idx: val for idx, val in zip(stored_indices, stored_values)}
    
    # Calculate dot product of matching indices
    for idx in query_indices:
        if idx in stored_dict:
            score += query_values.get(idx, 0) * stored_dict[idx]
    
    return score

def perform_query(conn, bm25, model, query, hard_filters, top_k=5, alpha=0.05):
    """Perform hybrid search query on PostgreSQL"""
    hard_filters = get_bio_data(hard_filters=hard_filters)
    
    # Generate embeddings
    sparse = bm25.encode_queries(query)
    dense = model.encode(query).tolist()
    
    # Scale vectors
    hdense, hsparse = hybrid_scale(dense, sparse, alpha)
    
    # Build WHERE clause
    where_clause, where_params = build_sql_where_clause(hard_filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    
    # Convert dense vector to PostgreSQL vector format
    dense_vector_str = "[" + ",".join(map(str, hdense)) + "]"
    
    # Query with hybrid search
    # We'll use cosine similarity for dense vectors and calculate sparse score separately
    query_sql = f"""
        SELECT 
            id,
            dense_vector,
            sparse_vector,
            product_display_name,
            brand_name,
            master_category,
            sub_category,
            article_type,
            gender,
            color,
            season,
            occasion,
            is_jewellery,
            style_image,
            (1 - (dense_vector <=> %s::vector)) * %s as dense_score
        FROM products
        {where_sql}
        ORDER BY dense_score DESC
        LIMIT %s
    """
    
    params = [dense_vector_str, alpha]
    if where_params:
        params.extend(where_params)
    params.append(top_k * 10)  # Get more results to filter by sparse score
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query_sql, params)
        results = cur.fetchall()
        cur.close()
        
        # Calculate hybrid scores
        scored_results = []
        for row in results:
            stored_sparse = row['sparse_vector']
            sparse_score = calculate_sparse_score(hsparse, stored_sparse)
            hybrid_score = row['dense_score'] + (sparse_score * (1 - alpha))
            
            scored_results.append({
                'id': row['id'],
                'score': hybrid_score,
                'metadata': {
                    'id': row['id'],
                    'product_display_name': row['product_display_name'],
                    'brand_name': row['brand_name'],
                    'master_category': row['master_category'],
                    'sub_category': row['sub_category'],
                    'article_type': row['article_type'],
                    'gender': row['gender'],
                    'color': row['color'],
                    'season': row['season'],
                    'occasion': row['occasion'],
                    'is_jewellery': row['is_jewellery'],
                    'style_image': row['style_image']
                }
            })
        
        # Sort by hybrid score and return top_k
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return {'matches': scored_results[:top_k]}
    
    except Exception as e:
        print(f"Error performing query: {e}")
        raise

def query_postgres(query, conn, model, bm25, hard_filters):
    """Query PostgreSQL database"""
    top_k = 2
    result = perform_query(conn, bm25, model, query, hard_filters=hard_filters, top_k=top_k)
    
    print("Result of PostgreSQL query for query:", query, "\n\n")
    print(result["matches"])
    selected_item = {}
    if len(result["matches"]) > 0:
        selected_item = result["matches"][0]["metadata"]
    
    if selected_item == {}:
        filter_keys_to_keep = ['master_category', 'sub_category', 'gender']
        modified_hard_filters = {k: v for k, v in hard_filters.items() if k in filter_keys_to_keep}
        not_modified_hard_filters = {k: v for k, v in hard_filters.items() if k not in filter_keys_to_keep}
        
        # Extract and flatten the values from nested dictionaries
        filter_values = []
        for value in not_modified_hard_filters.values():
            if isinstance(value, dict):
                filter_values.extend(value.values())
            elif isinstance(value, list):
                filter_values.extend(value)
            else:
                filter_values.append(value)
        
        # Convert all values to strings
        filter_values_str = " ".join(map(str, filter_values))
        
        if filter_values_str:
            updated_query = f"{query} {filter_values_str}"
        else:
            updated_query = query
        
        print("modified query:", updated_query)
        print("modified hard filters:", modified_hard_filters)
        print("Result of updated PostgreSQL query for query:", updated_query, "\n\n")
        result = perform_query(conn, bm25, model, updated_query, hard_filters=modified_hard_filters, top_k=top_k)
        
        print("Result of modified query with removed filter keys:", updated_query, "\n\n")
        print(result["matches"])
        
        if len(result["matches"]) > 0:
            selected_item = random.choice(result["matches"])["metadata"]
    
    return selected_item

# Global variables for connection and models (initialized lazily)
_conn = None
_model = None
_bm25 = None

def get_connection_and_models():
    """Get or initialize connection and models"""
    global _conn, _model, _bm25
    try:
        # Check if connection is closed or invalid
        if _conn is None or _conn.closed:
            _conn, _model, _bm25 = setup_postgres()
        else:
            # Test if connection is still alive
            _conn.cursor().execute("SELECT 1")
            _conn.rollback()
    except Exception:
        # Connection is dead, reconnect
        try:
            if _conn:
                _conn.close()
        except:
            pass
        _conn, _model, _bm25 = setup_postgres()
    
    return _conn, _model, _bm25

def run_postgres_query(query, hard_filters):
    """Run PostgreSQL query with cached models"""
    conn, model, bm25 = get_connection_and_models()
    
    bm25_fname = os.path.join(os.path.dirname(__file__), 'bm25.pkl')
    
    # Load the fitted bm25 model if file exists
    if os.path.exists(bm25_fname):
        with open(bm25_fname, 'rb') as f:
            bm25 = pickle.load(f)
    
    return query_postgres(query, conn, model, bm25, hard_filters)

def main():
    query = "Peter England baby blue jeans"
    hard_filters = build_hard_filters(color="blue", brand_name="peter_england")
    run_postgres_query(query, hard_filters)

if __name__ == "__main__":
    main()

