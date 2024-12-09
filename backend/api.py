import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from rank_bm25 import BM25Okapi
from src.embedding import compute_query_embedding
from src.retrieval import retrieve_top_events
from src.summarize import summarize_texts, custom_query_summary
from src.utils import filter_by_class_label, save_summary, humanize_labels, humanize_events, normalize_input

app = Flask(__name__)

# Load data
df = pd.read_csv("data/processed/all_data_cleaned.csv")
df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)

# Define paths
embedding_path = "data/embeddings/embeddings.npy"

# Prepare embeddings
texts = df['cleaned_text'].tolist()
if os.path.exists(embedding_path):
    print("Loading existing embeddings...")
    embeddings = np.load(embedding_path)
else:
    print("Computing new embeddings...")
    embeddings = compute_query_embedding(texts)
    np.save(embedding_path, embeddings)

# Prepare BM25
tokenized_corpus = [text.split() for text in df['cleaned_text']]
bm25 = BM25Okapi(tokenized_corpus)

@app.route('/summarize_by_event', methods=['POST'])
def summarize_by_event():
    """Summarize based on a specific event and class label."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        normalized_query = normalize_input(query)
        top_events = retrieve_top_events(normalized_query, bm25, embeddings, df, top_k=5, alpha=0.5)

        if not top_events:
            return jsonify({"error": "No events found for the query"}), 404

        selected_event = top_events[0]
        available_labels = df[df['event'] == selected_event]['class_label'].unique()

        if not available_labels.size:
            return jsonify({"error": "No labels available for the selected event"}), 404

        selected_label = available_labels[0]
        filtered_df = filter_by_class_label(df, selected_event, selected_label)

        if filtered_df.empty:
            return jsonify({
                "event": humanize_events(selected_event),
                "label": humanize_labels(selected_label),
                "summary": "No relevant data found."
            })

        summary = summarize_texts(filtered_df)
        return jsonify({
            "event": humanize_events(selected_event),
            "label": humanize_labels(selected_label),
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize_custom_query', methods=['POST'])
def summarize_custom_query():
    """Generate a summary for a custom query."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        normalized_query = normalize_input(query)
        summary = custom_query_summary(df, bm25, embeddings, normalized_query, alpha=0.5, top_k=10)
        return jsonify({
            "query": query,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/select_event_label', methods=['POST'])
def select_event_label():
    """Allow users to select an event and label for summarization."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        normalized_query = normalize_input(query)
        top_events = retrieve_top_events(normalized_query, bm25, embeddings, df, top_k=5, alpha=0.5)

        if not top_events:
            return jsonify({"error": "No events found for the query"}), 404

        event_idx = int(data.get("event_index", 0))
        if event_idx < 0 or event_idx >= len(top_events):
            return jsonify({"error": "Invalid event index"}), 400

        selected_event = top_events[event_idx]
        available_labels = df[df['event'] == selected_event]['class_label'].unique()

        label_idx = int(data.get("label_index", 0))
        if label_idx < 0 or label_idx >= len(available_labels):
            return jsonify({"error": "Invalid label index"}), 400

        selected_label = available_labels[label_idx]
        filtered_df = filter_by_class_label(df, selected_event, selected_label)

        if filtered_df.empty:
            return jsonify({
                "event": humanize_events(selected_event),
                "label": humanize_labels(selected_label),
                "summary": "No relevant data found."
            })

        summary = summarize_texts(filtered_df)
        return jsonify({
            "event": humanize_events(selected_event),
            "label": humanize_labels(selected_label),
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
