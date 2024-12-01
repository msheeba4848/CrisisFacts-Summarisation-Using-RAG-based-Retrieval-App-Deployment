from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


class BM25Retriever:
    def __init__(self, documents):
        # Store the raw documents
        self.documents = documents
        # Tokenize documents
        self.tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        # Initialize BM25 model with tokenized documents
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a given query.

        :param query: The search query as a string.
        :param top_k: Number of top documents to retrieve.
        :return: List of tuples with document and score.
        """
        # Tokenize the query
        tokenized_query = word_tokenize(query.lower())
        # Get BM25 scores for the query
        scores = self.bm25.get_scores(tokenized_query)
        # Rank documents by score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        # Return the top-k documents with their scores
        return [(self.documents[i], scores[i]) for i in ranked_indices]
