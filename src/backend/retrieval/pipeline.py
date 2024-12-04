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

def preprocess_text_column(text_series, model_name='allenai/longformer-base-4096'):
    """
    Preprocess a pandas Series of text: remove non-English words, tokenize, and lemmatize.
    Supports sequences up to the Longformer limit (4096 tokens) and uses GPU for processing.

    Args:
        text_series (pd.Series): A pandas Series containing text data.
        model_name (str): Transformer model name for tokenization.

    Returns:
        pd.Series: A pandas Series with processed text embeddings or tokenized output.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Longformer tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    def preprocess_line(line):
        if not isinstance(line, str):
            return ""
        try:
            # Remove non-alphabetic characters
            cleaned_line = re.sub(r'[^a-zA-Z\s]', '', line)

            # Process the text with Spacy for lemmatization
            doc = nlp(cleaned_line.lower())
            lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

            # Tokenize and move tensors to GPU
            inputs = tokenizer(
                ' '.join(lemmatized_tokens),
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(device)

            # Process with Longformer model
            outputs = model(**inputs)

            # Use the last hidden state as the processed output
            # Convert it back to CPU and reduce dimensions if needed
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")
            return None

    # Apply preprocessing line by line
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
