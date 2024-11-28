from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query, top_k=10):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], scores[i]) for i in ranked_indices]
