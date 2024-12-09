from flask import Flask, request, jsonify
from code.final_project import TwoStagePipeline

app = Flask(__name__)

# Example dataset
documents = [
    "Relief efforts in Florida",
    "Flooding in Texas",
    "FEMA is responding to emergencies",
]

# Initialize the pipeline
pipeline = TwoStagePipeline(documents)

@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get('query')
    results = pipeline.run(query)
    return jsonify({"query": query, "results": results})

if __name__ == '__main__':
    app.run(debug=True)
