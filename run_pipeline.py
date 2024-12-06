# Main script for running the pipeline
from retrieval.file_reader import (
    read_documents_from_directory,
    read_documents_from_csv,
    read_documents_from_json,
)
from final_project.pipeline import TwoStagePipeline, preprocess_documents

# Step 1: Read documents
documents = read_documents_from_directory('documents/')
# documents = read_documents_from_csv('documents/documents.csv', column_name='content')
# documents = read_documents_from_json('documents/documents.json', key='text')

# Step 2: Preprocess documents
documents = preprocess_documents(documents)

# Step 3: Initialize the pipeline
pipeline = TwoStagePipeline(documents)

# Step 4: Run a query
query = "flooding"
results = pipeline.run(query=query, bm25_top_k=20, faiss_top_k=5)

# Step 5: Print results
print("Query Results:")
for doc, score in results:
    print(f"Document: {doc}, Score: {score}")
