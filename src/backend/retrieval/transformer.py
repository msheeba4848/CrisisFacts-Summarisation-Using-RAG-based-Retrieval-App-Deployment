from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class TransformerRetrieverANN:
    def __init__(self, model_name, n_neighbors=5):
        """
        Transformer-based ANN Retriever using scikit-learn's NearestNeighbors.
        """
        self.model = SentenceTransformer(model_name)
        self.n_neighbors = n_neighbors
        self.nn_model = None  # Placeholder for the nearest neighbors model

    def fit(self, corpus):
        """
        Generates embeddings for the corpus and fits the NearestNeighbors model.
        """
        print("Generating embeddings for the corpus...")
        self.embeddings = self.model.encode(corpus, show_progress_bar=True)
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine', algorithm='auto')
        self.nn_model.fit(self.embeddings)
        self.corpus = corpus

    def query(self, query, top_n=5):
        """
        Queries the ANN index for the top-N most similar documents to the query.
        """
        query_embedding = self.model.encode([query])
        distances, indices = self.nn_model.kneighbors(query_embedding, n_neighbors=top_n)
        results = [self.corpus[i] for i in indices[0]]  # Retrieve top-N documents
        return results, distances[0]