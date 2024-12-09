import os
import sys
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.src.embedding import compute_query_embedding
from backend.src.retrieval import retrieve_top_events
from backend.src.summarize import summarize_texts, custom_query_summary
from backend.src.utils import filter_by_class_label, save_summary, humanize_labels, humanize_events, normalize_input


def interactive_hybrid_summarization_with_improvements(df, bm25, embeddings):
    print("Welcome to the summarization system!")

    print("\nOptions:")
    print("1. Summarize based on specific event and class label")
    print("2. Generate a summary for a custom query across the dataset")

    option = input("Choose an option (1 or 2): ").strip()

    if option == "1":
        query = input("What event are you interested in? (e.g., 'earthquake relief', 'hurricane rescue')\n> ").strip()
        normalized_query = normalize_input(query)
        top_events = retrieve_top_events(normalized_query, bm25, embeddings, df, top_k=5, alpha=0.5)

        # Display human-readable events
        print("\nTop events related to your query:")
        human_readable_events = [humanize_events(event) for event in top_events]
        for idx, event in enumerate(human_readable_events, 1):
            print(f"{idx}. {event}")
        try:
            event_idx = int(input("\nSelect the number corresponding to your event: ").strip()) - 1
            if event_idx < 0 or event_idx >= len(top_events):
                raise ValueError("Invalid selection. Please choose a valid number.")
        except ValueError as e:
            print(e)
            return
        selected_event = top_events[event_idx]
        print(f"\nYou selected: {humanize_events(selected_event)}")

        # Retrieve available class labels
        available_labels = df[df['event'] == selected_event]['class_label'].unique()
        human_readable_labels = [humanize_labels(label) for label in available_labels]
        print("\nWhat label are you interested in? Here are the available options:")
        for idx, label in enumerate(human_readable_labels, 1):
            print(f"{idx}. {label}")
        try:
            label_idx = int(input("\nSelect the number corresponding to your label: ").strip()) - 1
            if label_idx < 0 or label_idx >= len(available_labels):
                raise ValueError("Invalid selection. Please choose a valid number.")
        except ValueError as e:
            print(e)
            return
        selected_label = available_labels[label_idx]
        print(f"\nYou selected: {humanize_labels(selected_label)}")

        # Summarize and save
        filtered_df = filter_by_class_label(df, selected_event, selected_label)
        if not filtered_df.empty:
            print("\nGenerating summary...")
            summary = summarize_texts(filtered_df)
            print(f"\nSummary for '{humanize_events(selected_event)}' and label '{humanize_labels(selected_label)}':\n{summary}")
            save_option = input("\nWould you like to save this summary? (yes/no): ").strip().lower()
            if save_option == "yes":
                save_summary(humanize_events(selected_event), humanize_labels(selected_label), summary)
        else:
            print(f"No relevant data found for event: '{humanize_events(selected_event)}' and label: '{humanize_labels(selected_label)}'.")

    elif option == "2":
        query = input("Enter your custom query (e.g., 'earthquake damage'):\n> ").strip()
        normalized_query = normalize_input(query)
        summary = custom_query_summary(df, bm25, embeddings, normalized_query, alpha=0.5, top_k=10)
        print(f"\nSummary for custom query '{query}':\n{summary}")
    else:
        print("Invalid option. Please restart the system.")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/processed/all_data_cleaned.csv")
    df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)

    # Define paths
    embedding_path = "../data/embeddings/embeddings.npy"

    # Get cleaned texts
    texts = df['cleaned_text'].tolist()

    # Check if embeddings exist
    if os.path.exists(embedding_path):
        print("Loading existing embeddings...")
        embeddings = np.load(embedding_path)
    else:
        print("Computing new embeddings...")
        embeddings = compute_query_embedding(texts)  # Assume this function is defined elsewhere
        np.save(embedding_path, embeddings)

    # Prepare BM25
    tokenized_corpus = [text.split() for text in df['cleaned_text']]
    bm25 = BM25Okapi(tokenized_corpus)

    # Perform hybrid summarization
    interactive_hybrid_summarization_with_improvements(df, bm25, embeddings)
