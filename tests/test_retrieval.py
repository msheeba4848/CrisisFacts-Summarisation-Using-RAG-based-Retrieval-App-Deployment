from backend.retrieval import retrieve_documents

def test_retrieve_documents():
    query = "Flooding"
    filters = ["news"]
    results = retrieve_documents(query, filters)
    assert len(results) > 0
    assert all(doc["type"] == "news" for doc in results)
