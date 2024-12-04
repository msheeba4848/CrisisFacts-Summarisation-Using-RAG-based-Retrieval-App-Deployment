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

def preprocess_documents(text_series, model_name='bert-base-uncased'):
    """
    Preprocess a pandas Series of text: remove non-English words, tokenize, and lemmatize.
    Prepares the output for use with TwoStagePipeline (raw text + tokenized text).

    Args:
        text_series (pd.Series): A pandas Series containing text data.
        model_name (str): Transformer model name for tokenization.

    Returns:
        list: A list of preprocessed raw text (for FAISS).
        list: A list of tokenized text (for BM25).
    """
    # Load tokenizer for BM25
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw_texts = []  # For FAISS
    tokenized_texts = []  # For BM25

    def preprocess_line(line):
        if not isinstance(line, str):
            return "", []
        try:
            # Remove non-alphabetic characters
            cleaned_line = re.sub(r'[^a-zA-Z\s]', '', line)

            # Process the text with Spacy for lemmatization
            doc = nlp(cleaned_line.lower())
            lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

            # Join lemmatized tokens to form the raw text
            raw_text = ' '.join(lemmatized_tokens)

            # Tokenize for BM25
            tokenized_text = tokenizer.tokenize(raw_text)

            return raw_text, tokenized_text
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")
            return "", []

    for line in text_series:
        raw_text, tokenized_text = preprocess_line(line)
        raw_texts.append(raw_text)
        tokenized_texts.append(tokenized_text)

    return raw_texts, tokenized_texts


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
