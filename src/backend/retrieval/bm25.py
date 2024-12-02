from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

class BM25Retriever:
    def __init__(self, documents, model_name='bert-base-uncased'):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Tokenize documents using transformer tokenizer
        self.documents = documents
        self.tokenized_docs = [self._preprocess(doc) for doc in documents]
        # Initialize BM25 model with tokenized documents
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _preprocess(self, text):
        # Tokenize and return only input IDs as tokens (handles stopwords implicitly)
        return self.tokenizer.tokenize(text.lower())

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a given query.

        :param query: The search query as a string.
        :param top_k: Number of top documents to retrieve.
        :return: List of tuples with document and score.
        """
        tokenized_query = self._preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [(self.documents[i], scores[i]) for i in ranked_indices]
