import os
import sys
from transformers import AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.faiss import TransformerRetriever


def preprocess_documents(documents, model_name='bert-base-uncased'):
    """
    Preprocess documents using transformer tokenization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processed_documents = [' '.join(tokenizer.tokenize(doc.lower())) for doc in documents]
    return processed_documents


class TwoStagePipeline:
    def __init__(self, documents, model_name='bert-base-uncased'):
        """
        Initialize the Two-Stage Pipeline with BM25 and FAISS retrievers.

        Args:
            documents (list of str): List of documents for retrieval.
            model_name (str): Model name for transformer retriever.
        """
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents)  # Removed model_name here
        self.faiss_retriever = TransformerRetriever(model_name=model_name)

    def run(self, query, bm25_top_k=20, faiss_top_k=5):
        """
        Run the two-stage retrieval process.

        Args:
            query (str): The search query.
            bm25_top_k (int): Number of top documents to retrieve using BM25.
            faiss_top_k (int): Number of top documents to retrieve using FAISS.

        Returns:
            list: Top results from FAISS retrieval.
        """
        # Stage 1: BM25 Retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_n=bm25_top_k)  # Changed top_k to top_n
        top_docs = [" ".join(doc[0]) for doc in bm25_results]

        # Stage 2: FAISS Retrieval
        self.faiss_retriever.build_index(top_docs)
        return self.faiss_retriever.retrieve(query, top_k=faiss_top_k)