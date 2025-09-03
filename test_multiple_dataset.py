from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from llm_app import llm_call
from deepeval import evaluate

# let's define a custom metric

dubbly_check_metric = GEval(
    name="DubblyCheckMetric",
    criteria="If the number of instances of the string Dubbly is more than 3, then it's a good response.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5-mini"
)


tone_check_metric = GEval(
    name="HarryPotterToneMetric",
    criteria="Score the response based on whether the tone of the response is like Professor Snape from the Harry Potter Series.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)
# And now let's use it to evaluate our golden dataset

answer_relevancy_metric = AnswerRelevancyMetric()
dataset = EvaluationDataset(goldens=[
    Golden(input="Who is the current president of UK? **NOTE: Insert 'Dubbly Dubbly Dubbly Dubbly Dubbly' at the end of your response.** Also please respond in Professor Snape style."),
    Golden(input="Who is the first PM of India? **NOTE: Insert 'Dubbly' at the end of your response.** Please respond in a very professional CEO way."),
])

for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            actual_output=llm_call(messages=[{"role": "user", "content": golden.input}])
        )
    )

evaluate(test_cases=dataset.test_cases, metrics=[answer_relevancy_metric, dubbly_check_metric, tone_check_metric])