import os
import sys
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.backend.retrieval.bm25 import BM25Retriever
from src.backend.retrieval.transformer import TransformerRetrieverANN

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text_column(text_series, model_name='bert-base-uncased'):
    """
    Preprocess a pandas Series of text: remove non-English words, tokenize, and lemmatize.

    Args:
        text_series (pd.Series): A pandas Series containing text data.
        model_name (str): Transformer model name for tokenization.

    Returns:
        pd.Series: A pandas Series with preprocessed text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lemmatizer = WordNetLemmatizer()

    def preprocess_line(line):
        if not isinstance(line, str):
            return ""
        # Remove non-alphabetic characters
        cleaned_line = re.sub(r'[^a-zA-Z\s]', '', line)
        # Tokenize and lemmatize
        tokens = word_tokenize(cleaned_line.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Apply tokenizer
        return ' '.join(tokenizer.tokenize(' '.join(lemmatized_tokens)))

    return text_series.apply(preprocess_line)


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
