import numpy as np
from rank_bm25 import BM25Okapi

import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus, model_name=None):
        """
        Initialize the BM25 retriever with a corpus of documents.

        Args:
            corpus (list of str): List of documents to index.
            model_name (str, optional): Transformer model name. Defaults to None.
        """
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.model_name = model_name  # Stored for compatibility, if needed

    def retrieve(self, query, top_n=1):
        """
        Retrieve the top-N most relevant documents for a given query.

        Args:
            query (str): The search query.
            top_n (int): The number of top documents to return.

        Returns:
            list of tuple: A list of (document, score) tuples for the top-N results.
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Return the documents and their scores
        return [(self.tokenized_corpus[i], scores[i]) for i in top_indices]
