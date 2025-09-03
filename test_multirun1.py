from deepeval import assert_test
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationalGEval

def test_professionalism():
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant has acted professionally based on the content.",
        threshold=0.5
    )
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is DeepEval?"),
            Turn(role="assistant", content="Hello sir! DeepEval is an open-source LLM eval package.")
        ]
    )
    assert_test(test_case, [professionalism_metric])