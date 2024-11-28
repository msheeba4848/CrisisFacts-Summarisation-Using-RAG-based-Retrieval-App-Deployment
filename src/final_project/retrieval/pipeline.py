from src.final_project.retrieval.bm25 import BM25Retriever
from src.final_project.retrieval.faiss import FAISSRetriever

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
