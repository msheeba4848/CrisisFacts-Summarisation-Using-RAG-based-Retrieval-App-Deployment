import os
import sys
import pytest
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

# Adjusting the path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from src.embedding import compute_query_embedding
from src.retrieval import retrieve_top_events


@pytest.fixture
def sample_data(tmp_path):
    """
    Fixture for sample data, BM25 setup, and embeddings generation.
    Ensures the test environment can operate without requiring existing files.
    """
    # Define file paths
    csv_file = tmp_path / "all_data_cleaned.csv"
    embedding_file = tmp_path / "embeddings.npy"

    # Mock dataset creation
    data = {
        'event': ['earthquake rescue', 'flood relief', 'wildfire control', 'earthquake damage'],
        'cleaned_text': [
            'rescue operations after earthquake',
            'relief for flood victims',
            'efforts to control wildfire spread',
            'damage caused by earthquake shaking'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    # Ensure tokenized corpus and BM25 are set up
    df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)
    tokenized_corpus = [text.split() for text in df['cleaned_text']]
    bm25 = BM25Okapi(tokenized_corpus)

    # Generate or load embeddings
    if embedding_file.exists():
        embeddings = np.load(embedding_file)
    else:
        texts = df['cleaned_text'].tolist()
        embeddings = compute_query_embedding(texts)
        np.save(embedding_file, embeddings)

    return df, bm25, embeddings


def test_retrieve_top_events(sample_data):
    """
    Test for retrieve_top_events function using mock data.
    Validates output structure, length, and existence of returned events.
    """
    df, bm25, embeddings = sample_data

    # Define query and parameters
    query = "earthquake damage"
    top_k = 3
    alpha = 0.5

    # Call the function
    top_events = retrieve_top_events(query, bm25, embeddings, df, top_k=top_k, alpha=alpha)

    # Assertions
    assert isinstance(top_events, list), "Output should be a list."
    assert len(top_events) <= top_k, f"Returned events exceed top_k={top_k} limit."
    assert all(isinstance(event, str) for event in top_events), "All events should be strings."
    assert all(event in df['event'].values for event in top_events), "Some returned events are not in the dataset."

    # Log results for clarity
    print(f"Query: {query}")
    print(f"Top-{top_k} Retrieved Events: {top_events}")
    print("Test passed: retrieve_top_events works as expected!")


if __name__ == "__main__":
    # Run tests with verbosity
    pytest.main(["-v", __file__])