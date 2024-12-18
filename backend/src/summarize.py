import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import clean_text, escape_special_chars, filter_relevant_rows, normalize_query
from src.embedding import compute_query_embedding

# Load the summarizer with GPU support
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def summarize_texts(filtered_df, max_length=100, min_length=30):
    """Summarize filtered texts."""
    combined_text = " ".join(filtered_df['cleaned_text'].tolist())
    combined_text = combined_text[:2000]  # Truncate for summarization
    try:
        summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {e}"

def summarize_top_relevant_texts(results, max_rows=3):
    """
    Summarize the top N most relevant rows, dynamically adjusting max_length.
    """
    top_texts = " ".join(clean_text(text) for text in results['cleaned_text'].head(max_rows))
    input_length = len(top_texts.split())
    max_length = min(100, input_length - 10)  # Dynamically adjust max_length
    min_length = min(30, max_length // 2)  # Adjust min_length proportionally

    try:
        summary = summarizer(top_texts, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {e}"

def custom_query_summary(df, bm25, embeddings, query, alpha=0.5, top_k=10):
    """
    Generate a summary for a custom query by searching across all events and class labels.
    """
    # Normalize the query
    normalized_query = normalize_query(query)

    # Step 1: BM25 scores
    tokenized_query = normalized_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores  # Normalize BM25 scores

    # Step 2: Dense scores
    query_embedding = compute_query_embedding(query)
    dense_scores = cosine_similarity(query_embedding, embeddings).flatten()
    dense_scores = dense_scores / np.max(dense_scores) if np.max(dense_scores) > 0 else dense_scores  # Normalize Dense scores

    # Step 3: Combine scores
    combined_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    # Step 4: Retrieve top rows
    top_results = df.iloc[top_indices]
    print(f"\nTop results based on BM25 and dense scores:")
    print(top_results[['event', 'class_label', 'cleaned_text']].head(5))

    # Step 5: Filter for relevant rows
    filtered_results = filter_relevant_rows(top_results, normalized_query)

    if not filtered_results.empty:
        print(f"\nFiltered relevant rows for query '{query}':")
        print(filtered_results[['event', 'class_label', 'cleaned_text']].head(5))
        summary = summarize_top_relevant_texts(filtered_results, max_rows=3)
        return summary
    else:
        # Fallback: Match directly with the event column
        matched_event_rows = df[df['event'].str.contains(escape_special_chars(normalized_query), case=False, na=False)]
        if not matched_event_rows.empty:
            print(f"\nRows found for event match '{query}':")
            print(matched_event_rows[['event', 'class_label', 'cleaned_text']].head(5))
            summary = summarize_top_relevant_texts(matched_event_rows, max_rows=3)
            return summary
        else:
            return f"No relevant data found for query: '{query}'."