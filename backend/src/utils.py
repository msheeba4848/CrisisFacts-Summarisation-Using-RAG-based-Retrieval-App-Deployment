import re


def filter_by_class_label(df, event, class_label):
    """Filter rows by event and class label."""
    return df[(df['event'] == event) & (df['class_label'].str.contains(class_label, case=False, na=False))]


def clean_text(text):
    """Remove unnecessary tokens, URLs, and mentions from text."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'rt ', '', text, flags=re.IGNORECASE)  # Remove "RT" for retweets
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def escape_special_chars(text):
    """Escape special characters for regex matching."""
    return re.escape(text)

def filter_relevant_rows(results, query):
    """
    Filter rows for relevance by prioritizing event matches.
    """
    normalized_query = escape_special_chars(query)
    # Exact match with `event` column
    event_matches = results[results['event'].str.contains(normalized_query, case=False, na=False)]
    if not event_matches.empty:
        return event_matches

    # Fallback to broader text-based relevance
    text_matches = results[
        results['cleaned_text'].str.contains('|'.join(normalized_query.split('_')), case=False, na=False)
    ]
    return text_matches


def save_summary(event, label, summary):
    """Save the generated summary to a file."""
    filename = f"summary_{event.replace(' ', '_')}_{label.replace(' ', '_')}.txt"
    with open(filename, "w") as file:
        file.write(f"Event: {event}\n")
        file.write(f"Label: {label}\n")
        file.write(f"Summary:\n{summary}")
    print(f"Summary saved to {filename}.")

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



def normalize_input(user_input):
    """Normalize user input to match dataset format."""
    return user_input.lower().replace(' ', '_')


def normalize_query(query):
    """Normalize the query to match the dataset format."""
    # Define filler words and phrases to remove
    fillers = [
        "what about ", "tell me about ", "show me ", "can you find ",
        "could you find ", "is there any ", "do you have ", "please find ",
        "any information on ", "anything about ", "how about ",
        "do you know ", "can I see ", "give me details on", "can you find information about",
        "what does"
    ]

    # Remove filler words
    query = query.lower()
    for filler in fillers:
        query = query.replace(filler, "")

    # Remove multiple question marks and extra spaces
    query = query.replace("?", "").strip()
    query = query.replace("????", "").strip()
    query = query.replace("!", "").strip()
    # Replace spaces with underscores
    query = query.replace(" ", "_")

    return query

