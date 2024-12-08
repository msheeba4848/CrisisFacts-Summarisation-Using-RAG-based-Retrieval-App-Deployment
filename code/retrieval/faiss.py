import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

class TransformerRetriever:
    """
    FAISS-based dense retriever using transformer embeddings.
    """
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.index = None
        self.document_embeddings = None
        self.documents = []

    def encode(self, texts):
        """
        Encode texts into dense vector representations.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling
        return embeddings.numpy()

    def build_index(self, documents):
        """
        Build FAISS index from a list of documents.
        """
        self.documents = documents
        self.document_embeddings = self.encode(documents)
        embedding_dim = self.document_embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.document_embeddings)

    def retrieve(self, query, top_k=5):
        """
        Retrieve top_k documents for a given query using FAISS.
        """
        if self.index is None:
            raise ValueError("The FAISS index is not built. Call `build_index` first.")

        query_embedding = self.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1]
        return results