from faiss import IndexFlatL2
import numpy as np

# Mocked document embeddings
documents = [
    {"text": "Flooding in Jakarta has displaced thousands.", "type": "news"},
    {"text": "Relief efforts in Florida have increased after Hurricane Ian.", "type": "social"},
]
embeddings = np.random.rand(len(documents), 128)  # Mock embeddings

# FAISS Index
index = IndexFlatL2(128)
index.add(embeddings)

def retrieve_documents(query, filters):
    query_embedding = np.random.rand(1, 128)  # Mock query embedding
    distances, indices = index.search(query_embedding, 5)
    return [documents[i] for i in indices[0] if documents[i]["type"] in filters]
