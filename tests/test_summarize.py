import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from backend.src.utils import clean_text, escape_special_chars, filter_relevant_rows, normalize_query
from backend.src.embedding import compute_query_embedding
from backend.src.summarize import custom_query_summary
import os

# Load the summarizer with CPU for testing
test_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

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

def test_custom_query_summary(sample_data):
    """Test for custom_query_summary function with summarization pipeline."""
    df, bm25, embeddings = sample_data

    # Define a sample query that exists in the data
    query = df['cleaned_text'].iloc[0].split()[0]  # Use first word from dataset to ensure match
    alpha = 0.5
    top_k = 5

    # Ensure BM25 scores are non-zero
    tokenized_query = normalize_query(query).split()
    bm25_scores = bm25.get_scores(tokenized_query)
    assert not np.all(bm25_scores == 0), "BM25 scores are all zero. Check the query or the tokenized corpus."

    # Call the function
    summary = custom_query_summary(df, bm25, embeddings, query, alpha=alpha, top_k=top_k)

    # Assertions
    assert isinstance(summary, str), "Summary should be a string."
    assert len(summary) > 0, "Summary should not be empty."

    print("Generated Summary:")
    print(summary)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
