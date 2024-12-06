# Main script for running the pipeline
<<<<<<< HEAD
from retrieval.file_reader import (
    read_documents_from_directory,
    read_documents_from_csv,
    read_documents_from_json,
)
from final_project.pipeline import TwoStagePipeline, preprocess_documents
=======

from src.backend.retrieval.file_reader import read_from_csv
from src.backend.retrieval.pipeline import TwoStagePipeline, preprocess_csv
>>>>>>> f94a37a569d9fe33c215e53772bfbcd2f4bc6e08

# Step 1: Read documents
documents = read_from_csv('data/processed/all_data_cleaned.csv', column_name='cleaned_text')
# documents = read_documents_from_csv('documents/documents.csv', column_name='content')
# documents = read_documents_from_json('documents/documents.json', key='text')

# Step 2: Preprocess documents
documents = preprocess_csv(documents)

# Step 3: Initialize the pipeline
pipeline = TwoStagePipeline(documents)

# Step 4: Run a query
query = "flooding"
results = pipeline.run(query=query, bm25_top_k=20, faiss_top_k=5)

# Step 5: Print results
print("Query Results:")
for doc, score in results:
    print(f"Document: {doc}, Score: {score}")
