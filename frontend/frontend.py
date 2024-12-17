from flask import Flask, render_template, request, jsonify
import requests
import re

app = Flask(__name__)

# Backend URL
BACKEND_URL = "http://backend:5002"

# Utility Functions (Frontend Only)
def humanize_labels(label):
    """Convert snake_case class labels to human-readable format."""
    return label.replace('_', ' ').title()

def humanize_events(event):
    """Convert snake_case events to human-readable format with specific disaster handling."""
    event = event.replace('_earthquake', ' Earthquake')
    event = event.replace('_hurricane', ' Hurricane')
    event = event.replace('_floods', ' Floods')
    event = event.replace('_typhoon', ' Typhoon')
    event = event.replace('_cyclone', ' Cyclone')
    event = event.replace('_bombings', ' Bombings')
    event = event.replace('_explosion', ' Explosion')
    event = event.replace('_train-crash', ' Train Crash')
    event = event.replace('_wildfires', ' Wildfires')
    event = event.replace('_volcano', ' Volcano')
    event = event.replace('_shootings', ' Shootings')
    event = event.replace('_syndrome', ' Syndrome')
    event = event.replace('_fire', ' Fire')
    event = event.replace('_building-collapse', ' Building Collapse')
    event = event.replace('_airport', ' Airport')
    event = event.replace('_refinery-explosion', ' Refinery Explosion')
    event = event.replace('_respiratory', ' Respiratory')
    event = event.replace('_', ' ').title()  # Convert remaining snake_case to Title Case
    return event

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-events', methods=['POST'])
def get_events():
    query = request.form.get('query')
    response = requests.post(f"{BACKEND_URL}/events", json={"query": query})
    if response.status_code == 200:
        # Humanize events before rendering
        events = response.json().get("events", [])
        humanized_events = [humanize_events(event) for event in events]
        return render_template('events.html', query=query, events=humanized_events)
    return render_template('error.html', message="Failed to fetch events.")


@app.route('/get-labels', methods=['POST'])
def get_labels():
    event = request.form.get('event')
    print(f"Frontend: Selected event: {event}")  # Debug log

    # Send the event to the backend
    response = requests.post(f"{BACKEND_URL}/labels", json={"event": event})
    if response.status_code == 200:
        # Humanize labels before rendering
        labels = response.json().get("labels", [])
        humanized_labels = [humanize_labels(label) for label in labels]
        return render_template('labels.html', event=humanize_events(event), labels=humanized_labels)

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
        return render_template('summary.html',
                               event=humanize_events(event),
                               label=humanize_labels(label),
                               summary=summary)
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
