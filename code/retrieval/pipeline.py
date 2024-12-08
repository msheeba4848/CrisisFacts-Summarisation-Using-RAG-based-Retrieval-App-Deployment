import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code import BM25Retriever
from code import TransformerRetrieverANN

import torch
from sentence_transformers import SentenceTransformer


def preprocess_documents(documents, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Preprocess documents by generating embeddings using a Transformer model with GPU support.

    Args:
        documents (list of str): List of text documents to preprocess.
        model_name (str): Name of the Transformer model (default: MiniLM).

    Returns:
        embeddings (numpy.ndarray): Array of embeddings for the documents.
        model (SentenceTransformer): The loaded Transformer model (useful for later queries).
    """
    # Determine device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Transformer model
    model = SentenceTransformer(model_name, device=device)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)

    return embeddings, model


class TwoStagePipeline:
    def __init__(self, documents, model_name="bert-base-uncased"):
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents, model_name=model_name)
        self.ann_retriever = TransformerRetrieverANN(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def run(self, queries, bm25_top_k=20, faiss_top_k=5):
        bm25_results = [self.bm25_retriever.retrieve(query, top_k=bm25_top_k) for query in queries]
        bm25_candidates = [[doc for doc, _ in results] for results in bm25_results]

        # Flatten candidates and build FAISS index
        flat_candidates = [doc for candidates in bm25_candidates for doc in candidates]
        self.ann_retriever.build_index(flat_candidates)

        # Run FAISS retrieval
        faiss_results = self.ann_retriever.retrieve(queries, top_k=faiss_top_k)
        return faiss_results

