import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.backend.app import app

import unittest
import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/all_data_cleaned.csv")
df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)

class TestApp(unittest.TestCase):
    def setUp(self):
        """
        Set up the Flask test client and prepare test data.
        """
        self.client = app.test_client()
        self.test_event = df['event'].iloc[0]  # Use the first event from the dataset
        self.test_label = df['class_label'].iloc[0]  # Use the first class label from the dataset
        self.test_query = "earthquake relief"  # A custom query for testing

    def test_summarize_event_label(self):
        """
        Test /summarize endpoint with event_label option.
        """
        response = self.client.post("/summarize", json={
            "event": self.test_event,
            "label": self.test_label,
            "option": "event_label"
        })

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("event", response.json)
        self.assertIn("label", response.json)
        self.assertIn("summary", response.json)
        self.assertEqual(response.json["event"], self.test_event.replace('_', ' ').title())
        self.assertEqual(response.json["label"], self.test_label.replace('_', ' ').title())

        # Print the output for manual inspection
        print("\nEvent Label Test Output:", response.json)

    def test_summarize_custom_query(self):
        """
        Test /summarize endpoint with custom query option.
        """
        response = self.client.post("/summarize", json={
            "query": self.test_query,
            "option": "custom"
        })

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("summary", response.json)

        # Print the output for manual inspection
        print("\nCustom Query Test Output:", response.json)

    def test_invalid_option(self):
        """
        Test /summarize endpoint with an invalid option.
        """
        response = self.client.post("/summarize", json={
            "query": self.test_query,
            "option": "invalid_option"
        })

        # Assertions
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        self.assertEqual(response.json["error"], "Invalid option")

        # Print the output for manual inspection
        print("\nInvalid Option Test Output:", response.json)

    def test_missing_query(self):
        """
        Test /summarize endpoint with a missing query.
        """
        response = self.client.post("/summarize", json={
            "option": "custom"
        })

        # Assertions
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        self.assertEqual(response.json["error"], "Query is required")

        # Print the output for manual inspection
        print("\nMissing Query Test Output:", response.json)


if __name__ == "__main__":
    unittest.main()
