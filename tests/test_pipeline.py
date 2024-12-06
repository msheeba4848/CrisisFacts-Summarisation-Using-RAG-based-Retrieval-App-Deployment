import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
<<<<<<< HEAD
from final_project.pipeline import TwoStagePipeline
=======
from src.backend.retrieval.pipeline import TwoStagePipeline
>>>>>>> f94a37a569d9fe33c215e53772bfbcd2f4bc6e08


def test_pipeline():
    documents = ["Flooding in Texas", "Emergency response by FEMA", "Hurricane relief efforts"]
    pipeline = TwoStagePipeline(documents)

    results = pipeline.run("Flooding")
    assert len(results) > 0
    assert all(isinstance(doc[0], str) for doc in results)
    print(results)
