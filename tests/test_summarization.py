import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.backend.summarization import summarize_documents


def test_summarize_documents():
    documents = [
        {"text": "Flooding in Jakarta has displaced thousands."},
        {"text": "Hurricane relief efforts in Florida are underway."},
    ]
    summary = summarize_documents(documents)
    assert len(summary) > 0
