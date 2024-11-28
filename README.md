# final_project

A package for doing great things!

## Installation

```bash
pip install final_project
```

## Usage

bm25.py

Contains the BM25Retriever class for the first-stage retrieval using statistical methods.

faiss.py

Contains the FAISSRetriever class for second-stage retrieval using neural embeddings.

pipeline.py

Implements the TwoStagePipeline class, integrating BM25 and FAISS to perform a two-stage retrieval process.

app.py

A Flask application exposing your project as an API.

This allows users to interact with your pipeline via HTTP requests.

tests/ Directory
This directory contains all test files.

test_pipeline.py

Unit tests for pipeline.py. Ensures the two-stage pipeline works as expected.

test_final_project.py
Integration tests for the project as a whole (e.g., testing the Flask API or the full pipeline)



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

