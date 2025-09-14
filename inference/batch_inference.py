"""
Batch inference: This is for asynchronous requests where immediate respones aren't required.
For Batch inference, we don't need to deploy the model to an endpoint. Instrad we can directly send the request to the model resource.
We send te request as a BatchPredictionJob resource directly to the Model resource.

Reference - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage#create-batch-job-python_genai_sdk

"""

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from dotenv import load_dotenv
import os
import time

load_dotenv(dotenv_path="./inference/.env")


OUTPUT_GCS_URI = os.getenv("BATCH_PRED_JOB_OUTPUT_GCS_URI")
print ("Project ID:", os.getenv("GOOGLE_CLOUD_PROJECT"))
print ("Location:", os.getenv("GOOGLE_CLOUD_LOCATION"))
print ("Use Vertex AI:", os.getenv("GOOGLE_GENAI_USE_VERTEX_AI"))
print ("Output GCS URI:", OUTPUT_GCS_URI)


# create genai client
client = genai.Client(
                    vertexai=bool(os.getenv("GOOGLE_GENAI_USE_VERTEX_AI")),
                    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
                    http_options=HttpOptions(api_version="v1")
                    )


# Create a batch prediction job
job = client.batches.create(
    model="gemini-2.5-flash",
    src="gs://maximal-brace-132923_finetuning-datasets/sample-llm-datasets/batch_requests_for_multimodal_input_2.jsonl",
    config=CreateBatchJobConfig(dest=OUTPUT_GCS_URI)
)

print(f"Job name: {job.name}")
print(f"Job state: {job.state}")


completed_states = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

while job.state not in completed_states:
    time.sleep(30)
    job = client.batches.get(name=job.name)
    print(f"Job state: {job.state}")

