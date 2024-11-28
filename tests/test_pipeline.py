from src.final_project.retrieval.pipeline import TwoStagePipeline

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


def test_pipeline():
    documents = ["Flooding in Texas", "Emergency response by FEMA", "Hurricane relief efforts"]
    pipeline = TwoStagePipeline(documents)

    results = pipeline.run("Flooding")
    assert len(results) > 0
    assert all(isinstance(doc[0], str) for doc in results)
