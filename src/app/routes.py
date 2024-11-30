from flask import request, jsonify
from app import app
from backend.retrieval import retrieve_documents
from backend.summarization import summarize_documents

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query = data.get("query", "")
    filters = data.get("filters", [])
    
    documents = retrieve_documents(query, filters)
    summary = summarize_documents(documents)
    
    return jsonify({
        "summary": summary,
        "documents": documents
    })
