import pytest
from deepeval.dataset import EvaluationDataset, Golden
from agent import RAGAgent
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from metrics import relevancy
import os


# Pull the evaluation dataset for Confident AI cloud
dataset_alias = "RAG QA Agent Dataset"
dataset = EvaluationDataset()
dataset.pull(alias=dataset_alias)


# Initialise docs and Agent
base_path = "./datasets/Theranos"
documents = ["theranos.txt"]
document_paths = [os.path.join(base_path, document_name) for document_name in documents]
agent = RAGAgent(document_paths=document_paths)


# The following is an example of end to end LLM evaluation
# @pytest.mark.parametrize("golden", dataset.goldens)
# def test_ra_agent_components(golden):
#     answer, retrieved_docs = agent.answer(golden.input)
#     test_case = LLMTestCase(input=golden.input, actual_output=answer, retrieval_context=retrieved_docs)
#     # assert_test(golden=golden, observed_callback=agent.answer)
#     assert_test(test_case=test_case, metrics=[relevancy])


# The followin is an example of component level evaluation
@pytest.mark.parametrize("golden", dataset.goldens)
def test_ra_agent_components(golden: Golden):
    assert_test(golden=golden, observed_callback=agent.answer)


