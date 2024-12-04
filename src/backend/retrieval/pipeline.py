import os
import sys
import spacy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.backend.retrieval.bm25 import BM25Retriever
from src.backend.retrieval.transformer import TransformerRetrieverANN
import re
from transformers import AutoTokenizer, AutoModel

import torch
nlp = spacy.load("en_core_web_sm")


def preprocess_documents(text_series, model_name="bert-base-uncased", batch_size=32):
    """
    Preprocess a pandas Series of text with GPU acceleration:
    - Removes non-English words.
    - Tokenizes and lemmatizes.
    - Generates embeddings in batches using the GPU.

    Args:
        text_series (pd.Series): A pandas Series containing text data.
        model_name (str): Transformer model name for tokenization.
        batch_size (int): Number of documents to process per batch.

    Returns:
        list: A list of preprocessed raw text (for FAISS).
        list: A list of tokenized text (for BM25).
        list: A list of embeddings for ANN retrieval.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    raw_texts = []  # For FAISS
    tokenized_texts = []  # For BM25
    embeddings = []  # For ANN

    def preprocess_line(line):
        if not isinstance(line, str):
            return "", []
        # Remove non-alphabetic characters
        cleaned_line = re.sub(r"[^a-zA-Z\s]", "", line)

        # Process the text with Spacy for lemmatization
        doc = nlp(cleaned_line.lower())
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

        # Generate raw text and tokenized text
        raw_text = " ".join(lemmatized_tokens)
        tokenized_text = tokenizer.tokenize(raw_text)

        return raw_text, tokenized_text

    # Preprocess all lines
    preprocessed_data = [preprocess_line(line) for line in text_series]

    # Extract raw and tokenized text
    raw_texts = [raw for raw, _ in preprocessed_data]
    tokenized_texts = [tokens for _, tokens in preprocessed_data]

    # Batch process embeddings
    for i in range(0, len(raw_texts), batch_size):
        batch_texts = raw_texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)

        with torch.no_grad():
            model_output = model(**inputs)
            batch_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().tolist()

        embeddings.extend(batch_embeddings)

    return raw_texts, tokenized_texts, embeddings



class TwoStagePipeline:
    def __init__(self, documents, model_name='bert-base-uncased'):
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents, model_name=model_name)
        self.faiss_retriever = TransformerRetrieverANN()

    def run(self, query, bm25_top_k=20, faiss_top_k=5):
        # Stage 1: BM25 Retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        top_docs = [doc[0] for doc in bm25_results]

        # Stage 2: FAISS Retrieval
        self.faiss_retriever.build_index(top_docs)
        return self.faiss_retriever.retrieve(query, top_k=faiss_top_k)
