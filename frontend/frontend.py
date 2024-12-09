from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Frontend route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit_query', methods=['POST'])
def submit_query():
    try:
        # Extract form data
        query = request.form.get('query', '').strip()
        option = request.form.get('option', '1')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Determine the backend endpoint
        backend_url = (
            "http://backend:5002/summarize_by_event"
            if option == "1"
            else "http://backend:5002/summarize_custom_query"
        )

        # Send query to the backend API
        response = requests.post(backend_url, json={"query": query})

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Backend error", "details": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
