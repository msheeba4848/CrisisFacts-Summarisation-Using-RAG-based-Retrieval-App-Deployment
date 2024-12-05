# Utilities for reading documents from files
import os
import pandas as pd
import json

def read_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


def read_csv(file_path, doc_column, label_column):
    """
    Reads a CSV file and removes rows with missing values in the document or label columns.

    Args:
        file_path (str): Path to the CSV file.
        doc_column (str): Name of the column containing documents.
        label_column (str): Name of the column containing labels.

    Returns:
        tuple: A tuple containing:
            - documents (list): List of document strings.
            - labels (list): List of corresponding labels.
    """
    # Load the CSV
    df = pd.read_csv(file_path)

    # Drop rows with missing values in the specified columns
    df = df.dropna(subset=[doc_column, label_column])

    # Extract documents and labels
    documents = df[doc_column].astype(str).tolist()
    labels = df[label_column].astype(str).tolist()

    return documents, labels



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
