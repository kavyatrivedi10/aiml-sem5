import os
import pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
import torch
import time
from dotenv import find_dotenv, load_dotenv
# Load environment variables from the root .env file
root_env_path = find_dotenv()
load_dotenv(root_env_path)

def initialize_pinecone() -> pinecone.Pinecone:
    try:
        # initialize connection to pinecone (get API key at app.pinecone.io)
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENVIRONMENT")
        # init connection to pinecone
        # pinecone.init(api_key=api_key, environment=env)
        PC = pinecone.Pinecone(api_key=api_key, environment=env)
        return PC
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

def create_index(index_name, PC: pinecone.Pinecone):
    try:
        if index_name not in PC.list_indexes().names(): # or should this be PC.list_indexes()?
            # create the index
            PC.create_index(
                index_name,
                dimension=1024,
                metric="cosine", # or "dotproduct"? or "euclidean"?
                # pod_type="s1",
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        return PC.Index(index_name)
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def setup_pinecone():
    start_time = time.time()

    try:
        print('Initializing Pinecone...')
        PC = initialize_pinecone()
        print('Initialization completed.')

        print('Getting CLIP and BM25 model...')
        model, bm25 = get_clip_and_bm25_model()
        print('Models obtained:')
        print('---- Model:', model)
        print('---- BM25:', bm25)

        print('Creating index...')
        index_name = "final-database"
        pinecone_index = create_index(index_name, PC)
        print('Index created:', pinecone_index)
        print('Setup completed.')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
        return pinecone_index, model, bm25

    except Exception as e:
        print(f"Error setting up Pinecone: {e}")
        raise




def get_clip_and_bm25_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
        bm25 = BM25Encoder()
        return model, bm25
    except Exception as e:
        print(f"Error getting CLIP and BM25 model: {e}")
        raise

