from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend URL
BACKEND_URL = "http://backend:5002"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-events', methods=['POST'])
def get_events():
    query = request.form.get('query')
    response = requests.post(f"{BACKEND_URL}/events", json={"query": query})
    if response.status_code == 200:
        events = response.json().get("events", [])
        return render_template('events.html', query=query, events=events)
    return render_template('error.html', message="Failed to fetch events.")


@app.route('/get-labels', methods=['POST'])
def get_labels():
    event = request.form.get('event')
    print(f"Frontend: Selected event: {event}")  # Debug log

    # Send the event to the backend
    response = requests.post(f"{BACKEND_URL}/labels", json={"event": event})
    print(f"Frontend: Backend response status code: {response.status_code}")  # Debug log
    print(f"Frontend: Backend response JSON: {response.json()}")  # Debug log

    if response.status_code == 200:
        labels = response.json().get("labels", [])
        return render_template('labels.html', event=event, labels=labels)

    # Handle errors
    error_message = response.json().get("error", "Failed to fetch labels.")
    print(f"Frontend: Error fetching labels: {error_message}")  # Debug log
    return render_template('error.html', message=error_message)


@app.route('/summarize', methods=['POST'])
def summarize():
    event = request.form.get('event')
    label = request.form.get('label')
    response = requests.post(f"{BACKEND_URL}/summarize", json={"event": event, "label": label})
    if response.status_code == 200:
        summary = response.json().get("summary", "")
        return render_template('summary.html', event=event, label=label, summary=summary)
    return render_template('error.html', message="Failed to generate summary.")


@app.route('/custom-summary', methods=['POST'])
def custom_summary():
    query = request.form.get('query')
    response = requests.post(f"{BACKEND_URL}/custom-summary", json={"query": query})
    if response.status_code == 200:
        summary = response.json().get("summary", "")
        return render_template('custom_summary.html', query=query, summary=summary)
    return render_template('error.html', message="Failed to generate custom summary.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
