import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, corpus):
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, queries, top_n=1):
        results = []
        for query in queries:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_n]
            results.append(top_indices)
        return results

