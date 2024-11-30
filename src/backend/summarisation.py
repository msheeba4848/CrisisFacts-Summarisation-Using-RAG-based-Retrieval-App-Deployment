from transformers import pipeline

summarizer = pipeline('summarization', model="facebook/bart-large-cnn")

def summarize_documents(documents):
    texts = " ".join([doc['text'] for doc in documents])
    summary = summarizer(texts, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']
