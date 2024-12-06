import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus):
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, queries, top_n=1):
        """
        Queries BM25 for the top-N results for each query and returns indices.
        """
        results = []
        for query in queries:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            # Sort scores and retrieve the top-N indices
            top_indices = np.argsort(scores)[::-1][:top_n]
            results.append(top_indices.tolist())  # Convert to list for compatibility
        return results

