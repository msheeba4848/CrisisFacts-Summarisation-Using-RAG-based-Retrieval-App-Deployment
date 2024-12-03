from transformers import AutoTokenizer, AutoModel
from sklearn.random_projection import SparseRandomProjection
import numpy as np
import torch

class TransformerRetrieverANN:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_hash_tables=10):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.documents = []
        self.embeddings = None
        self.lsh = SparseRandomProjection(n_components=num_hash_tables)

    def embed_texts(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**tokens)
        embeddings = self.mean_pooling(model_output, tokens['attention_mask'])
        return embeddings.cpu().numpy()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def build_index(self, documents):
        self.documents = documents
        self.embeddings = self.embed_texts(documents)
        self.lsh.fit(self.embeddings)  # Create LSH hash buckets

    def retrieve(self, query, top_k=5):
        query_embedding = self.embed_texts([query])[0]

        # Ensure embeddings are transformed in the same space as the query
        hashed_query = self.lsh.transform(query_embedding.reshape(1, -1))
        hashed_embeddings = self.lsh.transform(self.embeddings)

        # Perform similarity search in hashed space
        candidates = np.dot(hashed_embeddings, hashed_query.T).flatten()
        top_indices = candidates.argsort()[-top_k:][::-1]  # Get top-k matches
        return [(self.documents[i], candidates[i]) for i in top_indices]
