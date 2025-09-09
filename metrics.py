from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval
)

from deepeval.test_case import LLMTestCaseParams


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



# metrics = [relevancy, recall, precision, answer_correctness, citation_accuracy]

__all__ = [
    "relevancy",
    "recall",
    "precision",
    "answer_correctness",
    "citation_accuracy"
]