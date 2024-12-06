# Utilities for reading documents from files
import os
import pandas as pd
import json
import re

def read_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


def read_csv(file_path):
    '''
    Reads the data file, selects necessary columns, and cleans the text.
    '''
    df = pd.read_csv(file_path)
    df = df[['cleaned_text', 'class_label']].dropna()
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()))
    return df



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
