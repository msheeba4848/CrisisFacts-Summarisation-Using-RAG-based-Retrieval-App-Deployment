import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

PROCESSED_DIR = os.getenv('PROCESSED_DIR')
EMBEDDINGS_DIR = os.getenv('EMBEDDINGS_DIR')

def generate_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    processed_path = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    df = pd.read_csv(processed_path)
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")

