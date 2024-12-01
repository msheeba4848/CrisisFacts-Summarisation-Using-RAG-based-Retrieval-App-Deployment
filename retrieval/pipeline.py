from bm25 import BM25Retriever
from faiss import FAISSRetriever
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_documents(documents): # Use the preprocessing function to clean and tokenize the text:
    stop_words = set(stopwords.words('english'))
    processed_documents = []
    for doc in documents:
        tokens = word_tokenize(doc.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        processed_documents.append(' '.join(filtered_tokens))
    return processed_documents


class TwoStagePipeline:
    def __init__(self, documents):
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents)
        self.faiss_retriever = FAISSRetriever()

    def run(self, query, bm25_top_k=20, faiss_top_k=5):
        # Stage 1: BM25 Retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        top_docs = [doc[0] for doc in bm25_results]

        # Stage 2: FAISS Retrieval
        self.faiss_retriever.build_index(top_docs)
        return self.faiss_retriever.retrieve(query, top_k=faiss_top_k)

# The TwoStagePipeline class combines two powerful information retrieval techniques, BM25 and FAISS, into a pipeline.
# It leverages BM25 for initial filtering of documents based on keyword matching,
# followed by FAISS for fine-grained semantic search on the top candidates. Here's how this works in detail:

