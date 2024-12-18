import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from backend.src.embedding import compute_query_embedding

def retrieve_top_events(query, bm25, embeddings, df, top_k=5, alpha=0.5):
    """
    Retrieve top-k events combining BM25 and Dense Embedding scores.

    Parameters:
    - query (str): Query text.
    - bm25 (BM25Okapi): BM25 model.
    - embeddings (ndarray): Dense embeddings for documents.
    - df (DataFrame): DataFrame containing the dataset.
    - top_k (int): Number of top results to retrieve.
    - alpha (float): Weight for BM25 scores.

    Returns:
    - List of top-k events ranked by relevance.
    """
    # Step 1: BM25 retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-8)  # Normalize BM25 scores

    # Step 2: Dense retrieval
    query_embedding = compute_query_embedding([query])  # Wrap query in a list
    dense_scores = cosine_similarity(query_embedding, embeddings).flatten()
    dense_scores = dense_scores / (np.max(dense_scores) + 1e-8)  # Normalize Dense scores

    # Step 3: Combine BM25 and Dense scores
    combined_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    # Step 4: Rank and retrieve top-k events
    top_indices = np.argsort(combined_scores)[::-1][:top_k]  # Descending order
    top_events = df.iloc[top_indices]['event'].tolist()  # Retrieve top events
    return top_events