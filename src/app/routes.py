from flask import Flask, request, jsonify
from src.backend.retrieval import retrieve_documents
from final_project.summarization import summarize_documents

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')
    filters = data.get('filters', [])
    top_k = data.get('top_k', 10)

    retrieved_docs = retrieve_documents(query, filters, top_k)
    summary = summarize_documents(retrieved_docs)

    return jsonify({"summary": summary, "documents": retrieved_docs})

if __name__ == '__main__':
    app.run(debug=True)
