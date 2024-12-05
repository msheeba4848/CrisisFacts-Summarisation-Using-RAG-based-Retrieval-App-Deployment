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
    Reads a CSV file and extracts documents and labels for use with train_test_split.

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

    # Ensure the specified columns exist
    if doc_column not in df.columns:
        raise ValueError(f"The column '{doc_column}' is not found in the dataset.")
    if label_column not in df.columns:
        raise ValueError(f"The column '{label_column}' is not found in the dataset.")

    # Extract documents and labels
    documents = df[doc_column].dropna().astype(str).tolist()
    labels = df[label_column].dropna().astype(str).tolist()

    # Ensure the lengths match
    if len(documents) != len(labels):
        raise ValueError("The document and label columns have mismatched lengths.")

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
