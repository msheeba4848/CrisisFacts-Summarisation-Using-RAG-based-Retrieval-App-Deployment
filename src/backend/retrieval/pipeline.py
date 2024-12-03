import os
import sys
from transformers import AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.backend.retrieval.bm25 import BM25Retriever
from src.backend.retrieval.transformer import TransformerRetrieverANN


def preprocess_documents(documents, model_name='bert-base-uncased'):
    """
    Preprocess documents using transformer tokenization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processed_documents = [' '.join(tokenizer.tokenize(doc.lower())) for doc in documents]
    return processed_documents


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
