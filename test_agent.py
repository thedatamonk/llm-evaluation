from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.metrics import GEval
from deepeval import evaluate


from agent import RAGAgent
import os


CREATE_SYNTHETIC_DATASET = False
dataset_alias = "RAG QA Agent Dataset"

if CREATE_SYNTHETIC_DATASET:
    synthesizer = Synthesizer()

    # generate synthetic dataset
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=["datasets/Theranos/theranos.txt"]
    )

    # now create an evaluation dataset using the goldens
    dataset = EvaluationDataset(goldens=goldens)

    dataset.push(alias=dataset_alias)

    del dataset
else:
    print (f"Loading existing dataset: {dataset_alias}")
    dataset = EvaluationDataset()
    dataset.pull(dataset_alias)


# Initialise docs and Agent
base_path = "./datasets/Theranos"
documents = ["theranos.txt"]
document_paths = [os.path.join(base_path, document_name) for document_name in documents]
agent = RAGAgent(document_paths=document_paths)


# Using LLMTestCase create test cases using goldens
test_cases = []
for golden in dataset.goldens:
    retrieved_docs = agent.retrieve(golden.input)
    response = agent.generate(query=golden.input, retrieved_docs=retrieved_docs)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(response),
        retrieval_context=retrieved_docs,
        expected_output=golden.expected_output
    )

    test_cases.append(test_case)


print (f"Created {len(test_cases)} test cases.")


# Define metrics
# Retriever metrics
# Contextual Relevancy — The retrieved context must be relevant to the query
# Contextual Recall — The retrieved context should be enough to answer the query
# Contextual Precision — The retrieved context should be precise and must not include unnecessary details

relevancy = ContextualRelevancyMetric()
recall = ContextualRecallMetric()
precision = ContextualPrecisionMetric()

# Generator metrics
# Answer Correctness — To evaluate only the answer from our json.
# Citation Accuracy — To evaluate the citations mentioned in the json.

answer_correctness = GEval(
    name="Answer Correctness",
    criteria="Evaluate if the actual output's 'answer' property is correct and complete from the input and retrieved context. If the answer is not correct or complete, reduce score.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
)

citation_accuracy = GEval(
    name="Citation Accuracy",
    criteria="Check if the citations in the actual output are correct and relevant based on input and retrieved context. If they're not correct, reduce score.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
)



metrics = [relevancy, recall, precision, answer_correctness, citation_accuracy]

# And finally we use the evaluate() function to evaluate all the test cases and log the metrics summary into Confident AI cloud

evaluate(test_cases=test_cases, metrics=metrics)
