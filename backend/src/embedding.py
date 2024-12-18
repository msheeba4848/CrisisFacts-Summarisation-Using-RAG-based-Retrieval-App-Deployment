from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import os

# Device configuration: Prioritize MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU
if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend for Apple Silicon")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA backend")
else:
    device = torch.device("cpu")
    print("Using CPU backend")

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
print("Tokenizer and model loaded successfully.")


def compute_query_embedding(texts, batch_size=32):
    """
    Compute dense embeddings for a list of texts in batches to avoid memory issues.

    Args:
        texts (list of str): Input texts to compute embeddings for.
        batch_size (int): Number of texts to process in each batch. Default is 32.

    Returns:
        np.ndarray: Numpy array containing embeddings of shape (len(texts), embedding_dim).
    """
    embeddings = []
    print(f"Generating embeddings for {len(texts)} texts...")
    for i in range(0, len(texts), batch_size):
        print(f"Processing batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
        # Slice texts into batches
        batch_texts = texts[i:i + batch_size]

        # Check if batch is empty
        if not batch_texts:
            print("Batch is empty, skipping...")
            continue

        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        # Pass inputs through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract CLS token embeddings for the batch (first token)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move to CPU
        embeddings.append(batch_embeddings)

    # Combine all batch embeddings into a single numpy array
    final_embeddings = np.vstack(embeddings)
    print(f"Generated embeddings of shape: {final_embeddings.shape}")
    return final_embeddings


if __name__ == "__main__":
    # Define paths
    DATA_PATH = "../data/processed/all_data_cleaned.csv"
    EMBEDDING_PATH = "../data/embeddings/embeddings.npy"

    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    import pandas as pd
    df = pd.read_csv(DATA_PATH)

    if "cleaned_text" not in df.columns:
        raise ValueError("Column 'cleaned_text' not found in the data file.")

    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    print(f"Loaded {len(texts)} texts for embedding generation.")

    # Generate embeddings
    embeddings = compute_query_embedding(texts, batch_size=32)

    # Save embeddings
    print(f"Saving embeddings to {EMBEDDING_PATH}...")
    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)
    print(f"Embeddings successfully saved to {EMBEDDING_PATH}")