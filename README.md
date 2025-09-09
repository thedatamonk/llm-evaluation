**Current situation**

1. Does your evaluation process resemble the repetitive loop of running LLM applications on a list of prompts, manually inspecting outputs, and attempting to gauge quality based on each input?

2. LLM Evaluation is iterative in nature. Key components of an LLM evaluation framework - 
    - continuously evolving evaluation dataset
    - set of evaluation metrics (tailored to the usecase we're targeting)

3. There is a difference between LLM evaluation and LLM system evaluation

**LLM evaluation**
1. Evaluating standalone LLMs on a variety of LLM tasks like named entitiy recognition, text generation, summarization, QA, sentiment analysis, intent classification, etc.

2. Example of metrics for "standalone" LLM evaluation - 
    - GLUE (General Language Understanding Evaluation)
    - SuperGLUE
    - HellaSwag
    - TruthfulQA
    - MMLU

**Exercise:** Let's take an Ollama model and then evaluate it on these metrics.

***Evaluation of the finetuned model**
The evaluation of the fine-tuned model or a RAG (Retrieval Augmented Generation)-based model typically involves a comparison with its performance against a ground truth dataset if available.

**LLM system evaluation**

What comprises a LLM system?
1. LLM
2. Prompt template
3. Context retrieval pipelines
4. Finetuning mechanism

## Evaluation frameworks and platforms

- Prompt Flow in Microsoft Azure AI studio
- Weights & Biases in combination of LangChain
- LangSmith by LangChain
- DeepEval by confidence-ai
- TruEra

# Evaluation strategies

## Offline evalaution

- offline evaluation is usually performed before deployment and integrated with CI/CD systems to enable faster iteration.
- offline evaluation can also be performed for reggression testing.
- Different ways in which we can perform offline llm evalaution - 
    - evaluate against a "golden" dataset - we can also create this dataset using a relatively powerful LLM.
    - llms can also be used for generating the LLM golden dataset


## Responsible AI metrics
- LLM responses need to be tested for how responsible they are.
- There are several RAI categories, against which the LLM is tested. Categories are like "self-harm", "copyright".
- So looks like we can design our questions for one or more of the categories or use a predefined RAI benchmark dataset that is publicly available.

### Different tasks require different eval strategy
1. retrieval system
- **faithfulness:** Measures the factual consistency of the generated answer against the given context.
- **Response Relevancy:** Measures how relevant a response is to the user input.
- **Context precision:** Evaluates whether all the ground truth–relevant items present in the contexts are ranked higher or not.
- **Context relevancy:** Measures the relevancy of the retrieved context, calculated based on both the question and contexts.
- **Context recall:** Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.
- **Answer correctness:** Gauges the accuracy of the generated answer when compared to the ground truth.

2. text-to-sql system
- **Exact-set-match accuracy (EM):** EM evaluates each clause in a prediction against its corresponding ground truth SQL query. However, a limitation is that there exist numerous diverse ways to articulate SQL queries that serve the same purpose.
- **Execution Accuracy (EX):** EX evaluates the correctness of generated answers based on the execution results.
- **VES (Valid Efficiency Score):** A metric to measure the efficiency along with the usual execution correctness of a provided SQL query.

3. NER
- **Classification metrics:** Classification metrics (precision, recall, accuracy, F1 score, etc.) at entity level or model level.
- **InterpretEval:** The main idea is to divide the data into buckets of entities based on attributes such as entity length, label consistency, entity density, sentence length, etc., and then evaluate the model on each of these buckets separately. **(What is this?)**

4. Q&A
5. Summarization

## Online evaluation
1. online evaluation is simply about testing the LLM system in real-world production scenarios. for example, in ChatGPT, Gemini, the user is given the option to provide feedback on the LLM responses.
2. Types of online metrics that are collected - 
- user engagement and utility metrics
- user interaction
- quality of response
- user feedback and retention
- performance metrics
- cost metrics


# Important links - 
- Refer to notion page of LLM evaluation

# What I know so far?
1. ~~~How to setup DeepEval?~~
2. ~~How to use DeepEval with an Ollama run model as well as using the default OpenAI models?~~
3. ~~Define custom metrics~~
4. ~~Evaluate a golden dataset on a set of predefined metrics and custom metrics.~~
5. ~~End-to-end LLM evaluation where we treat the LLM app as a black-box and then test it.~~
6. How to perform single and multi-turn evals?

# What I need to understand next?
1. ~~How to evaluate each component of an LLM app?~~
2. ~~How to evaluate agents? I think this will be covered in the first point itself.~~
3. Fix the trace issue that I am getting while running `deepeval test run test_agent_in_prod.py`
    - For some reason, I am not able to find the bug in this, but we'll revisit this later.

