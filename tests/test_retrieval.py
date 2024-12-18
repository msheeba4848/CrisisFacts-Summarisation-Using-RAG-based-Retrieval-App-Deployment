import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from backend.src.embedding import compute_query_embedding
from backend.src.retrieval import retrieve_top_events
import os


@pytest.fixture
def sample_data():
    """Fixture for sample data and BM25 setup."""
    # Load data from CSV
    file_path = "../backend/data/processed/all_data_cleaned.csv"
    if not os.path.exists(file_path):
        pytest.fail("Test requires 'all_data_cleaned.csv' to exist at the specified location.")

    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)

    # Filter out rows with empty 'cleaned_text'
    df = df[df['cleaned_text'].str.strip() != ""]

    # Add check for non-empty tokenized corpus
    tokenized_corpus = [text.split() for text in df['cleaned_text'] if len(text.split()) > 0]
    if len(tokenized_corpus) == 0:
        pytest.fail("Tokenized corpus is empty after filtering. Ensure 'cleaned_text' column has valid text.")

    bm25 = BM25Okapi(tokenized_corpus)

    # Generate embeddings for cleaned text
    embedding_path = "../backend/data/embeddings/embeddings.npy"
    if os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
    else:
        texts = df['cleaned_text'].tolist()
        embeddings = compute_query_embedding(texts)
        np.save(embedding_path, embeddings)

    return df, bm25, embeddings

def test_retrieve_top_events(sample_data):
    """Test for retrieve_top_events function using real data."""
    df, bm25, embeddings = sample_data

    # Define query and parameters
    query = "earthquake damage"
    top_k = 5
    alpha = 0.5

    # Call the function
    top_events = retrieve_top_events(query, bm25, embeddings, df, top_k=top_k, alpha=alpha)

    # Assertions
    assert len(top_events) <= top_k, "Top-k results exceed the specified limit."
    assert isinstance(top_events, list), "Output is not a list."
    assert all(isinstance(event, str) for event in top_events), "Not all elements in the output are strings."
    assert all(event in df['event'].values for event in top_events), "Some returned events do not exist in the dataset."

    print("Test passed: retrieve_top_events works as expected with real data!")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
