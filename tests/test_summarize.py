import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from backend.src.embedding import compute_query_embedding
from backend.src.summarize import custom_query_summary

@pytest.fixture
def sample_data():
    """Fixture for sample data and BM25 setup."""
    # Create a small sample DataFrame for testing
    data = {
        "event": ["earthquake", "flood", "hurricane", "wildfire"],
        "class_label": ["natural disaster", "natural disaster", "natural disaster", "natural disaster"],
        "cleaned_text": [
            "An earthquake caused significant damage to the city.",
            "Heavy floods affected thousands of people.",
            "A hurricane destroyed several coastal areas.",
            "A wildfire burned down a large forest."
        ]
    }
    df = pd.DataFrame(data)

    # Tokenize the corpus and initialize BM25
    tokenized_corpus = [text.split() for text in df['cleaned_text']]
    bm25 = BM25Okapi(tokenized_corpus)

    # Generate embeddings
    embeddings = compute_query_embedding(df['cleaned_text'].tolist())

    return df, bm25, embeddings

def test_custom_query_summary(sample_data):
    """Test for custom_query_summary function."""
    df, bm25, embeddings = sample_data

    # Define a query that matches the sample data
    query = "earthquake damage"
    alpha = 0.5
    top_k = 2

    # Call the function
    summary = custom_query_summary(df, bm25, embeddings, query, alpha=alpha, top_k=top_k)

    # Assertions
    assert isinstance(summary, str), "Summary should be a string."
    assert len(summary) > 0, "Summary should not be empty."
    print("Test passed! Summary:", summary)

if __name__ == "__main__":
    pytest.main(["-v", __file__])