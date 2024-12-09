# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from code.final_project.pipeline import TwoStagePipeline


# def test_pipeline():
#     documents = ["Flooding in Texas", "Emergency response by FEMA", "Hurricane relief efforts"]
#     pipeline = TwoStagePipeline(documents)

#     results = pipeline.run("Flooding")
#     assert len(results) > 0
#     assert all(isinstance(doc[0], str) for doc in results)
#     print(results)
