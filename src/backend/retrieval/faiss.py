from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class TransformerRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.documents = []
        self.embeddings = None

    def embed_texts(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**tokens)
        embeddings = self.mean_pooling(model_output, tokens['attention_mask'])
        return embeddings.cpu().numpy()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # Apply mean pooling to transformer outputs
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def build_index(self, documents):
        self.documents = documents
        self.embeddings = self.embed_texts(documents)

    def retrieve(self, query, top_k=5):
        query_embedding = self.embed_texts([query])[0]
        scores = np.dot(self.embeddings, query_embedding)  # Compute similarity (dot product)
        top_k_indices = np.argsort(scores)[-top_k:][::-1]  # Get top-k indices (sorted by score)
        return [(self.documents[i], scores[i]) for i in top_k_indices]


