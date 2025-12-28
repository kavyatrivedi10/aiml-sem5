import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from PineconeLocal.utils.postgres_utils import setup_postgres, get_db_connection
import pickle
import chardet
import time
import os
import json
import psycopg2
from psycopg2.extras import execute_values
import concurrent.futures

def get_images(data):
    images = []
    for index, row in data.iterrows():
        try:
            # Download image from the 'link' column
            image_url = row['style_image']
            response = requests.get(image_url)

            if response.status_code == 200:
                # Convert the downloaded content into a PIL image
                img = Image.open(BytesIO(response.content))
                images.append(img)
            else:
                print(f"Failed to download image for index {index}")
        except Exception as e:
            print(f"Error encountered while processing image at index {index}. Error: {e}")
    return images

def parallel_upsert(conn, upsert_data):
    """Insert data into PostgreSQL in parallel"""
    try:
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO products (
            id, dense_vector, sparse_vector, product_display_name, brand_name,
            master_category, sub_category, article_type, gender, color,
            season, occasion, is_jewellery, style_image, pattern,
            sleeve_styling, sleeve_length, fabric, neck
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            dense_vector = EXCLUDED.dense_vector,
            sparse_vector = EXCLUDED.sparse_vector,
            product_display_name = EXCLUDED.product_display_name,
            brand_name = EXCLUDED.brand_name,
            master_category = EXCLUDED.master_category,
            sub_category = EXCLUDED.sub_category,
            article_type = EXCLUDED.article_type,
            gender = EXCLUDED.gender,
            color = EXCLUDED.color,
            season = EXCLUDED.season,
            occasion = EXCLUDED.occasion,
            is_jewellery = EXCLUDED.is_jewellery,
            style_image = EXCLUDED.style_image,
            pattern = EXCLUDED.pattern,
            sleeve_styling = EXCLUDED.sleeve_styling,
            sleeve_length = EXCLUDED.sleeve_length,
            fabric = EXCLUDED.fabric,
            neck = EXCLUDED.neck
        """
        
        execute_values(cur, insert_query, upsert_data)
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        print(f"Error upserting data: {e}")
        raise

def insert_data_parallel(conn, model, bm25, data, batch_size=200, num_threads=20):
    """Insert data into PostgreSQL in parallel batches"""
    try:
        total_batches = len(data) // batch_size + int(len(data) % batch_size != 0)
        print(f"Total batches: {total_batches}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for batch_idx, i in enumerate(range(0, len(data), batch_size)):
                i_end = min(i + batch_size, len(data))
                data_batch = data.iloc[i:i_end]

                meta_dict = data_batch[['id', 'product_display_name', 'brand_name', 'master_category', 
                                        'sub_category', 'article_type', 'gender', 'color', 'season', 
                                        'occasion', 'is_jewellery', 'style_image']].to_dict(orient="records")

                cols_to_consider = ['product_display_name', 'master_category', 'sub_category', 'color', 
                                    'pattern', 'occasion', 'sleeve_styling', 'sleeve_length', 'fabric', 'neck']
                cols_for_query = data_batch[cols_to_consider]
                query_strings = [" ".join(str(val) for col, val in row.items() if val != 'none') 
                                for _, row in cols_for_query.iterrows()]

                # Load images
                img_batch = []
                for x in meta_dict:
                    currImageId = x["id"]
                    currImageName = f'./downloaded_images2/{currImageId}.jpg'
                    if os.path.exists(currImageName):
                        img = Image.open(currImageName)
                        img_batch.append(img)
                    else:
                        print(f"The image '{currImageName}' does not exist.")
                        img_batch.append(None)

                # Generate embeddings
                sparse_embeds = bm25.encode_documents([text for text in query_strings])
                
                # Filter out None images for encoding
                valid_images = [img for img in img_batch if img is not None]
                valid_indices = [i for i, img in enumerate(img_batch) if img is not None]
                
                if valid_images:
                    dense_embeds = model.encode(valid_images).tolist()
                else:
                    dense_embeds = []
                
                # Prepare data for insertion
                upsert_data = []
                dense_idx = 0
                for idx, (sparse, meta) in enumerate(zip(sparse_embeds, meta_dict)):
                    # Handle missing images
                    if idx in valid_indices:
                        dense = dense_embeds[dense_idx]
                        dense_idx += 1
                    else:
                        # Use zero vector if image is missing
                        dense = [0.0] * 1024
                    
                    # Convert dense vector to PostgreSQL vector format string
                    dense_vector_str = "[" + ",".join(map(str, dense)) + "]"
                    
                    # Convert sparse vector to JSONB format
                    sparse_json = json.dumps({
                        'indices': sparse.get('indices', []),
                        'values': sparse.get('values', [])
                    })
                    
                    # Prepare row data
                    row_data = (
                        str(meta["id"]),
                        dense_vector_str,
                        sparse_json,
                        meta.get("product_display_name"),
                        meta.get("brand_name"),
                        meta.get("master_category"),
                        meta.get("sub_category"),
                        meta.get("article_type"),
                        meta.get("gender"),
                        meta.get("color"),
                        meta.get("season"),
                        meta.get("occasion"),
                        meta.get("is_jewellery"),
                        meta.get("style_image"),
                        data_batch.iloc[idx].get("pattern") if "pattern" in data_batch.columns else None,
                        data_batch.iloc[idx].get("sleeve_styling") if "sleeve_styling" in data_batch.columns else None,
                        data_batch.iloc[idx].get("sleeve_length") if "sleeve_length" in data_batch.columns else None,
                        data_batch.iloc[idx].get("fabric") if "fabric" in data_batch.columns else None,
                        data_batch.iloc[idx].get("neck") if "neck" in data_batch.columns else None,
                    )
                    upsert_data.append(row_data)

                # Submit batch for parallel processing
                futures.append(executor.submit(parallel_upsert, conn, upsert_data))

                # Print progress
                print(f"Batch {batch_idx + 1}/{total_batches}: Embeddings generated: {len(upsert_data)}")

            # Wait for all upsert tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error upserting data: {e}")

    except Exception as e:
        print(f"Error inserting data: {e}")
        raise

def upsert_csv(csv_file, char_enc, conn, model, bm25):
    """Upsert data from CSV file to PostgreSQL"""
    try:
        total_start_time = time.time()

        # Timing point: Reading CSV
        read_csv_start_time = time.time()
        data = pd.read_csv(csv_file, encoding=char_enc)
        read_csv_end_time = time.time()
        read_csv_elapsed_time = read_csv_end_time - read_csv_start_time
        print("read_csv_elapsed_time:", read_csv_elapsed_time)
        
        data = data[['id', 'product_display_name', 'brand_name', 'color', 'master_category', 
                    'sub_category', 'article_type', 'gender', 'season', 'occasion', 'is_jewellery', 
                    'style_image', 'pattern', 'sleeve_styling', 'sleeve_length', 'fabric', 'neck']]

        # Timing point: Fitting BM25 model
        bm25_fit_start_time = time.time()
        bm25.fit(data['product_display_name'])
        bm25_fit_end_time = time.time()
        bm25_fit_elapsed_time = bm25_fit_end_time - bm25_fit_start_time
        print("bm25_fit_elapsed_time:", bm25_fit_elapsed_time)

        with open('bm25_choli.pkl', 'wb') as f:
            pickle.dump(bm25, f)

        # Timing point: Upserts
        upsert_start_time = time.time()
        insert_data_parallel(conn, model, bm25, data, batch_size=200, num_threads=20)
        upsert_end_time = time.time()
        upsert_elapsed_time = upsert_end_time - upsert_start_time
        print("upsert_elapsed_time:", upsert_elapsed_time)

        # Get count of records
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM products")
        count = cur.fetchone()[0]
        cur.close()
        print(f"Total records in database: {count}")

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        print("total_elapsed_time:", total_elapsed_time)

        # Print timing results
        print(f"Reading CSV took {read_csv_elapsed_time:.2f} seconds.")
        print(f"Fitting BM25 took {bm25_fit_elapsed_time:.2f} seconds.")
        print(f"Upserts took {upsert_elapsed_time:.2f} seconds.")
        print(f"Total time taken: {total_elapsed_time:.2f} seconds.")

    except Exception as e:
        print(f"Error occurred during upsert_csv: {e}")
        raise

def main():
    conn, model, bm25 = setup_postgres()
    startTime = time.time()
    csv_file = "../dataset/choli.csv"
    rawdata = open(csv_file, 'rb').read()
    result = chardet.detect(rawdata)
    char_enc = result['encoding']

    print(char_enc)
    upsert_csv(csv_file, char_enc, conn, model, bm25)
    endTime = time.time()
    print('final time to upsert', endTime - startTime)
    conn.close()

if __name__ == "__main__":
    main()

