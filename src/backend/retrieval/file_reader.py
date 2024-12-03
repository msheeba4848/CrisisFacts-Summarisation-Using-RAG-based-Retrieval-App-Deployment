# Utilities for reading documents from files
import os
import pandas as pd
import json
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def read_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


def read_csv(file_path, text_column=None):
    """
    Reads a CSV file and extracts the specified text column.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text.
                           If None, the function will auto-detect the first text-based column.

    Returns:
        pd.Series: The specified text column as a pandas Series.
    """
    # Load the CSV
    df = pd.read_csv(file_path)

    # Auto-detect text column if not provided
    if text_column is None:
        text_column = next((col for col in df.columns if df[col].dtype == 'object'), None)
        if text_column is None:
            raise ValueError("No suitable text column found in the dataset.")
        print(f"No text_column specified. Using '{text_column}' as the text column.")

    # Ensure the text column exists
    if text_column not in df.columns:
        raise ValueError(f"The column '{text_column}' is not found in the dataset.")

    return df[text_column]


def read_from_json(file_path, key='text'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [item[key] for item in data]


# # Example usage:
#
# directory_path = "path/to/your/documents"
# documents = read_documents_from_directory(directory_path)
# # documents = read_documents_from_csv(file_path, column_name='content')
# # documents = read_documents_from_json(file_path, key='text')
