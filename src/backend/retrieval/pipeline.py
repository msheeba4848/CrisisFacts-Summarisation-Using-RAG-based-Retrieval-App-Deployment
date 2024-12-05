import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.backend.retrieval.bm25 import BM25Retriever
from src.backend.retrieval.transformer import TransformerRetrieverANN

import re
import torch
from transformers import AutoTokenizer, AutoModel

def preprocess_documents(text_list, model_name="bert-base-uncased", batch_size=64):
    """
    Fully GPU-accelerated text preprocessing and embedding generation:
    - Tokenizes and processes text in batches on the GPU.
    - Replaces CPU-bound Spacy operations.

    Args:
        text_list (list): A list of strings containing text data.
        model_name (str): Transformer model name for tokenization and embedding.
        batch_size (int): Number of documents to process per batch.

    Returns:
        list: Raw cleaned text (for FAISS).
        list: Tokenized text (for BM25).
        list: Embeddings for ANN retrieval.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    raw_texts = []  # For FAISS
    tokenized_texts = []  # For BM25
    embeddings = []  # For ANN

    # Preprocess text: Remove non-alphabetic characters
    def clean_text(line):
        if not isinstance(line, str):
            return ""
        return re.sub(r"[^a-zA-Z\s]", "", line).lower()

    # Clean all lines
    raw_texts = [clean_text(line) for line in text_list]

    # Tokenize and embed in batches
    for i in range(0, len(raw_texts), batch_size):
        batch_texts = raw_texts[i: i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)

        with torch.no_grad():
            # Generate embeddings on the GPU
            model_output = model(**inputs)
            batch_embeddings = model_output.last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings.extend(batch_embeddings.cpu().numpy())  # Move to CPU

        # Store tokenized text for BM25
        batch_tokens = [tokenizer.tokenize(text) for text in batch_texts]
        tokenized_texts.extend(batch_tokens)

    return raw_texts, tokenized_texts, embeddings


class TwoStagePipeline:
    def __init__(self, documents, model_name='bert-base-uncased'):
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents, model_name=model_name)
        self.TransformerRetrieverANN = TransformerRetrieverANN()

    def run(self, query, bm25_top_k=20, faiss_top_k=5):
        # Stage 1: BM25 Retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        top_docs = [doc[0] for doc in bm25_results]

        # Stage 2: Transformer Retriever ANN
        self.TransformerRetrieverANN.build_index(top_docs)
        return self.TransformerRetrieverANN.retrieve(query, top_k=faiss_top_k)
