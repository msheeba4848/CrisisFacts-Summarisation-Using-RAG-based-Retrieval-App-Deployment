from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_documents(documents):
    concatenated_text = " ".join(doc["text"] for doc in documents)
    summary = summarizer(concatenated_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]
