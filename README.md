# final_project

A package for doing great things!

## Installation

```bash
pip install final_project
```

## Usage

### Retrieval Part: 

This part implements a two-stage document retrieval system for ranking and retrieving documents based on their relevance to a user query. It combines BM25 retrieval in Stage 1 with FAISS-based neural embeddings in Stage 2, providing a robust and efficient approach to information retrieval.

#### Purpose
This retrieval system is designed to efficiently retrieve and rank documents from large datasets using both lexical similarity (BM25) and semantic similarity (FAISS with SentenceTransformers). Key applications include:

- Search engines for textual data.
- Research tools to identify relevant documents in large corpora.
- Semantic retrieval for better contextual understanding of user queries.

#### System Architecture
##### Stage 1: BM25 Retrieval
<b> Purpose</b>: Quickly rank documents based on lexical similarity (e.g., overlapping keywords).<br>
<b> Method </b>: Uses the BM25 algorithm to retrieve the top k most relevant documents.<br>
<b> Implementation</b>: Implemented in the BM25Retriever class (src/final_project/retrieval/bm25.py), which:<br>
- Tokenizes input documents.<br>
- Computes BM25 relevance scores for a query.<br>
- Returns the top k ranked documents.<br>

##### Stage 2: FAISS + Neural Embeddings
<b>Purpose</b>: Refine and re-rank the results from Stage 1 using semantic embeddings.<br/>
<b> Method:</b> <br/>
FAISS: A library for efficient similarity search and clustering of dense vectors.
SentenceTransformer: Generates high-quality semantic embeddings for documents and queries.
<b>Implementation:</b>
Implemented in the FAISSRetriever class (src/final_project/retrieval/faiss.py), which: <br/>
- Encodes documents and queries into embeddings using a pre-trained SentenceTransformer model (all-MiniLM-L6-v2 by default).
- Builds an index of document embeddings using FAISS.
- Retrieves and ranks the top k documents based on cosine similarity.
- 
#### Pipeline
The TwoStagePipeline (src/final_project/retrieval/pipeline.py) orchestrates the retrieval process:
0. preprocess: Tokenizes and preprocesses input documents.
1. BM25 Stage: Retrieves a subset of documents (bm25_top_k) based on keyword similarity.<br/>
2. FAISS Stage: Further narrows down and re-ranks these documents (faiss_top_k) based on semantic similarity.

#### file_reader.py

Contains utility functions for reading documents from: <br/>
A directory of text files.
A CSV file with a specific column.
A JSON file with structured data.

#### Test Usage
```python
poetry install
```

The BM25Retriever requires the punkt tokenizer from NLTK. Run:

```bash
python setup_nltk.py
```

Use pytest to validate the retrieval pipeline:

```bash
pytest tests/test_pipeline.py
```

#### Running the script
    
```bash
python run_pipeline.py
``` 

## Viewing the Documentation

We produced Sphinx documentation, you can view it locally by opening the generated HTML files in your browser:

   ```bash
   open docs/_build/html/index.html
   ```

   - On **Linux**, use:
     ```bash
     xdg-open docs/_build/html/index.html
     ```

   - On **Windows**, use:
     ```cmd
     start docs\_build\html\index.html
     ```

---

### Notes:
- Ensure you have built the documentation successfully before attempting to open the HTML file.
- The main landing page for the documentation is `index.html`.

## Contributing

Clone and set up the repository with

```bash
git clone TODO && cd final_project
pip install -e ".[dev]"
```

Install pre-commit hooks with

```bash
pre-commit install
```

Run tests using

```
pytest -v tests
```



_------_ # Sheebas's Version


# Proposed File Structure

CRISISFacts/
├── data/
│   ├── raw/                    # Raw data files (e.g., social media, news, and official statements)
│   ├── processed/              # Preprocessed data ready for modeling
│   └── embeddings/             # Precomputed vector embeddings for ANN search
├── src/
│   ├── app/                    # Flask or FastAPI application code
│   │   ├── __init__.py
│   │   ├── routes.py           # API routes for query input and summarization
│   │   └── utils.py            # Helper functions for app
│   ├── backend/                # Backend logic for retrieval and summarization
│   │   ├── __init__.py
│   │   ├── retrieval.py        # ANN search (FAISS/ScaNN implementation)
│   │   ├── summarization.py    # Summarization logic using PEGASUS, T5, or BART
│   │   └── preprocessing.py    # Tokenization and data cleaning scripts
│   ├── frontend/               # Frontend integration for query input and result display
│   │   ├── static/             # Static files (CSS, JS)
│   │   ├── templates/          # HTML templates
│   │   └── app.js              # Frontend logic (if using React)
│   ├── evaluation/             # Scripts for model evaluation
│   │   ├── rouge_eval.py       # ROUGE metric evaluation
│   │   └── precision_at_k.py   # Precision@k computation
│   └── config.py               # Configuration variables (e.g., paths, constants)
├── tests/
│   ├── test_retrieval.py       # Unit tests for retrieval module
│   ├── test_summarization.py   # Unit tests for summarization module
│   ├── test_preprocessing.py   # Unit tests for preprocessing logic
│   ├── test_app.py             # Tests for API endpoints
│   └── test_integration.py     # Integration tests for end-to-end system
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis on the CRISISFacts Dataset
│   └── model_prototyping.ipynb # Prototyping summarization and retrieval models
├── docker/
│   ├── Dockerfile              # Dockerfile for backend services
│   ├── Dockerfile.frontend     # Dockerfile for frontend service
│   └── docker-compose.yml      # Compose file for multi-service setup
├── scripts/
│   ├── preprocess_data.py      # Script for batch data preprocessing
│   ├── generate_embeddings.py  # Script to compute vector embeddings
│   └── start_app.py            # Script to launch the app (used in Docker)
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # CI workflow for testing and linting
│   │   └── deploy.yml          # Deployment workflow
├── .env.example                # Example environment variables file
├── .gitignore                  # Files and directories to ignore in version control
├── poetry.lock                 # Poetry lock file for dependency resolution
├── pyproject.toml              # Poetry configuration file
├── README.md                   # Project overview and usage instructions
└── discussion.md               # Documentation of challenges and improvements

# CRISISFacts Project: Real-Time Crisis Summarization

## Project Overview
The CRISISFacts project focuses on efficiently summarizing and retrieving crisis-related information from diverse data sources, including social media, news articles, and official statements.

---

## Steps Completed

### 1. Define File Structure
We created a structured directory for the project to ensure modularity and maintainability:
- `src/`: Core application logic for backend and frontend.
- `data/`: Raw and processed data, including embeddings.
- `tests/`: Unit and integration tests.
- `notebooks/`: EDA and model prototyping.
- `docker/`: Docker configuration for containerization.
- `scripts/`: Standalone scripts for preprocessing and other utilities.

---

### 2. Create Backend Logic
We implemented the backend with the following components:
- **Data Retrieval**:
  - Used FAISS to perform Approximate Nearest Neighbor (ANN) search for document retrieval.
- **Summarization**:
  - Utilized HuggingFace's `transformers` library with models like PEGASUS and BART for abstractive summarization.
- **Preprocessing**:
  - Implemented functions for tokenization, cleaning, and filtering irrelevant data.

Key files:
- `src/backend/retrieval.py`
- `src/backend/summarization.py`
- `src/backend/preprocessing.py`

---

### 3. Set Up Frontend
Designed a basic frontend using Flask (or React as an optional framework). Features include:
- Query input box.
- Dropdown for filtering results by media type.
- Display section for summaries and links.

Key files:
- `src/app/__init__.py`
- `src/app/routes.py`

---

### 4. Add Unit Tests
Added tests to verify the functionality of the backend:
- Retrieval logic: Ensures documents are retrieved based on the query and filters.
- Summarization logic: Validates the generation of concise summaries.

Key files:
- `tests/test_retrieval.py`
- `tests/test_summarization.py`

---

### 5. Install Dependencies
Installed the necessary libraries for the project:
- Core libraries: Flask, Transformers, FAISS, NumPy, PyTorch.
- Development tools: pytest, black, flake8, isort.

Dependencies are managed through:
- `requirements.txt` for `pip` users.
- `pyproject.toml` for `Poetry` users.

---

### 6. Docker Integration
Set up Docker for containerized deployment:
- Created a `Dockerfile` for the backend.
- Configured `docker-compose.yml` to run multiple services, including the frontend and backend.

Key files:
- `docker/Dockerfile`
- `docker/docker-compose.yml`

---

### 7. Create a Virtual Environment
Set up a virtual environment for dependency isolation:
- Using `venv` or `virtualenv`:
  - Created and activated a virtual environment.
  - Installed dependencies from `requirements.txt`.
- Using `Poetry`:
  - Initialized the environment with `poetry install`.
  - Managed dependencies via `pyproject.toml`.

---

### 8. Added Documentation
Documented project details and usage:
- Wrote a detailed `README.md` to guide users on setup and usage.
- Included a `discussion.md` for challenges and potential improvements.

---

## Next Steps
1. Implement the frontend interface (React or Flask templates).
2. Extend functionality with geospatial visualization or multimodal data.
3. Optimize the summarization and retrieval pipelines for real-time performance.
4. Deploy the application and evaluate using real-world crisis datasets.

---

