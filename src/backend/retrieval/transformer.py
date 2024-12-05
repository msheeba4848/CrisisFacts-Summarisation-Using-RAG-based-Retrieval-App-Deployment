from transformers import AutoTokenizer, AutoModel
import torch

class TransformerRetrieverANN:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).cuda()  # Load the model on GPU
        self.documents = []
        self.embeddings = None

    def embed_texts(self, texts):
        """
        Convert a list of texts into embeddings using the transformer model.
        """
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model_output = self.model(**tokens)
        embeddings = self.mean_pooling(model_output, tokens['attention_mask'])
        return embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Perform mean pooling on the model output with attention mask.
        """
        token_embeddings = model_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def build_index(self, documents):
        """
        Build the index by embedding all documents.
        """
        self.documents = documents
        self.embeddings = self.embed_texts(documents)  # Keep embeddings on the GPU

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k documents for a given query using cosine similarity.
        """
        query_embedding = self.embed_texts([query])  # Query embedding on GPU

        # Compute cosine similarity between query and document embeddings
        dot_product = torch.matmul(self.embeddings, query_embedding.T).squeeze(1)
        query_norm = torch.norm(query_embedding, dim=1)
        doc_norms = torch.norm(self.embeddings, dim=1)
        cosine_similarities = dot_product / (doc_norms * query_norm + 1e-9)

        # Ensure top_k does not exceed the number of available documents
        top_k = min(top_k, cosine_similarities.size(0))

        # Get top-k results
        top_k_indices = torch.topk(cosine_similarities, top_k).indices
        results = [(self.documents[i], cosine_similarities[i].item()) for i in top_k_indices]
        return results
