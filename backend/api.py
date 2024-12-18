import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from src.retrieval import retrieve_top_events
from src.summarize import summarize_texts, custom_query_summary
from src.utils import filter_by_class_label, save_summary, normalize_input
from src.embedding import compute_query_embedding
import torch

app = Flask(__name__)

# ----------------- Configuration -----------------

# Paths
DATA_PATH = "data/processed/all_data_cleaned.csv"
EMBEDDING_PATH = "data/embeddings/embeddings.npy"
MODEL_NAME = "bert-base-uncased"  # Replace if using another model
MODEL_CACHE_DIR = "./models"

# ----------------- Embedding Function -----------------
def compute_query_embedding(texts, tokenizer, model):
    """
    Compute embeddings for a list of texts using a pre-trained tokenizer and model.
    Args:
        texts (list): List of input texts.
        tokenizer: HuggingFace tokenizer object.
        model: HuggingFace model object.
    Returns:
        numpy.ndarray: Embeddings for the input texts.
    """
    embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings.append(embedding.squeeze().numpy())

    return np.array(embeddings)

# ----------------- Load Data -----------------

print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
    df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)
    texts = df['cleaned_text'].tolist()
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Data file not found at {DATA_PATH}. Exiting.")
    raise e

# ----------------- Load or Generate Embeddings -----------------

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
print("Tokenizer and model loaded successfully.")

print("Loading embeddings...")
if os.path.exists(EMBEDDING_PATH):
    embeddings = np.load(EMBEDDING_PATH)
    print("Embeddings loaded from file.")
else:
    print("Embeddings not found. Generating embeddings...")
    embeddings = compute_query_embedding(texts)
    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)
    print("Embeddings generated and saved.")

# ----------------- Initialize BM25 -----------------

print("Preparing BM25...")
tokenized_corpus = [text.split() for text in df['cleaned_text']]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 initialized.")

# ----------------- Routes -----------------

@app.route('/events', methods=['POST'])
def get_events():
    query = request.json.get('query', '')
    print(f"Received query: {query}")

    if not query:
        print("Error: Query parameter is missing.")
        return jsonify({"error": "Query parameter is missing."}), 400

    try:
        normalized_query = normalize_input(query)
        print(f"Normalized query: {normalized_query}")
        
        top_events = retrieve_top_events(normalized_query, bm25, embeddings, df, top_k=10, alpha=0.5)
        print(f"Top events: {top_events}")

        return jsonify({"events": top_events})
    except Exception as e:
        print(f"Error during event retrieval: {str(e)}")
        return jsonify({"error": "Internal server error."}), 500


@app.route('/labels', methods=['POST'])
def get_labels():
    selected_event = request.json.get('event', '').strip()
    print(f"Received event: {selected_event}")

    if not selected_event:
        return jsonify({"error": "Event parameter is missing."}), 400

    filtered_df = df[df['event'] == selected_event]
    if filtered_df.empty:
        return jsonify({"error": f"No data found for event '{selected_event}'"}), 404

    available_labels = filtered_df['class_label'].dropna().unique()
    print(f"Available labels: {available_labels}")

    return jsonify({"labels": list(available_labels)})


@app.route('/summarize', methods=['POST'])
def summarize():
    event = request.json.get('event', '')
    label = request.json.get('label', '')
    print(f"Received event: {event}, label: {label}")

    if not event or not label:
        return jsonify({"error": "Event or label parameter is missing."}), 400

    filtered_df = filter_by_class_label(df, event, label)
    if filtered_df.empty:
        return jsonify({"error": f"No data found for event '{event}' and label '{label}'"}), 404

    summary = summarize_texts(filtered_df)
    print(f"Generated summary: {summary}")

    return jsonify({"summary": summary})


@app.route('/custom-summary', methods=['POST'])
def custom_summary():
    query = request.json.get('query', '')
    print(f"Received custom summary query: {query}")

    if not query:
        return jsonify({"error": "Query parameter is missing."}), 400

    normalized_query = normalize_input(query)
    summary = custom_query_summary(df, bm25, embeddings, normalized_query, alpha=0.5, top_k=10)
    print(f"Generated custom summary: {summary}")

    return jsonify({"summary": summary})

# ----------------- Main Entry Point -----------------

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5002, debug=True)