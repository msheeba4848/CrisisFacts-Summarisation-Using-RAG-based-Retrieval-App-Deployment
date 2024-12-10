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

# Get cleaned texts
texts = df['cleaned_text'].tolist()

# Check if embeddings exist
if os.path.exists(embedding_path):
    embeddings = np.load(embedding_path)
else:
    embeddings = compute_query_embedding(texts)
    np.save(embedding_path, embeddings)

# Prepare BM25
tokenized_corpus = [text.split() for text in df['cleaned_text']]
bm25 = BM25Okapi(tokenized_corpus)

@app.route('/events', methods=['POST'])
def get_events():
    query = request.json.get('query', '')
    print(f"Received query: {query}")  # Debugging: Check the received query

    normalized_query = normalize_input(query)
    top_events = retrieve_top_events(normalized_query, bm25, embeddings, df, top_k=10, alpha=0.5)
    human_readable_events = [humanize_events(event) for event in top_events]
    print(f"Top events: {human_readable_events}")  # Debugging: Log the top events

    return jsonify({"events": human_readable_events})


@app.route('/labels', methods=['POST'])
def get_labels():
    # Receive the event from the request
    selected_event = request.json.get('event', '').strip()
    print(f"Received event: {selected_event}")  # Debugging log

    # Check if the event exists in the dataset
    filtered_df = df[df['event'] == selected_event]
    if filtered_df.empty:
        print(f"No data found for event: {selected_event}")  # Debugging log
        return jsonify({"error": f"No data found for event '{selected_event}'"}), 404

    # Retrieve unique class labels
    available_labels = filtered_df['class_label'].dropna().unique()
    print(f"Available labels for event '{selected_event}': {available_labels}")  # Debugging log

    # Convert labels to human-readable format
    human_readable_labels = [humanize_labels(label) for label in available_labels]

    # Prepare response
    return jsonify({"labels": human_readable_labels})



@app.route('/summarize', methods=['POST'])
def summarize():
    event = request.json.get('event', '')
    label = request.json.get('label', '')
    print(f"Received event: {event}, label: {label}")  # Debugging: Check inputs

    filtered_df = filter_by_class_label(df, event, label)
    if filtered_df.empty:
        print(f"No data found for event '{event}' and label '{label}'.")  # Debugging
        return jsonify({"error": "No relevant data found"}), 404

    summary = summarize_texts(filtered_df)
    print(f"Generated summary: {summary}")  # Debugging
    return jsonify({"summary": summary})


@app.route('/custom-summary', methods=['POST'])
def custom_summary():
    query = request.json.get('query', '')
    normalized_query = normalize_input(query)
    summary = custom_query_summary(df, bm25, embeddings, normalized_query, alpha=0.5, top_k=10)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
