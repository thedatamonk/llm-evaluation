"""
In this code, we will be finetuning an open-source model on a custom dataset on Vertex AI platform.

Specific task is to finetune a Llama model on MetaMathQA dataset using Vertex AI's managed finetuning service.

**Steps**
1. **Prepare the dataset**: Download the `MetaMathQA` dataset and convert it to JSONL format required by Vertex AI.
2. **Fine-Tune the Model**: Configure and launch a managed fine-tuning job on Vertex AI using a Llama 3.1 8B model.
3. **Deploy the tuned model**: Once the fine-tuning job is complete, deploy the model to an endpoint on Vertex AI and evaluate its performance.
4. **Compare (Optional)**: Compare our model's output with the official pre-trained MetaMath model from Hugging Face.
5. **Run Official Evaluation (Advanced)**: Download our tuned model and run the official evaluation scripts from the MetaMath repository.
"""
import os
import time
import vertexai
from vertexai.preview.tuning._tuning import SourceModel
from vertexai.preview.tuning import sft
from google.cloud.aiplatform_v1beta1.types import JobState
from utils import prepare_metamathqa_dataset
from pydantic import BaseModel, Field
from typing import Literal
import uuid

PROJECT_ID = "maximal-brace-132923"
PROJECT_NUMBER = "710805310428"
LOCATION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}_finetuning-datasets"
BUCKET_URI = f"gs://{BUCKET_NAME}"

DATASET_CREATION_NEEDED = False  # Set to False if dataset is already prepared and uploaded to GCS

# Initialize the Vertex AI SDK. This authenticates our session and sets the default project and location.
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)


# prepare dataset and upload to GCS
if DATASET_CREATION_NEEDED:
    print("Preparing MetaMathQA dataset...")
    datasets_gcs_uri = prepare_metamathqa_dataset()
else:
    print("Using existing MetaMathQA dataset in GCS...")
    datasets_gcs_uri = {
        "gcs_train_uri": f"gs://{BUCKET_NAME}/metamathqa/metamath_gsm8k_train.jsonl",
        "gcs_validation_uri": f"gs://{BUCKET_NAME}/metamathqa/metamath_gsm8k_validation.jsonl"
    }

# Configure the fine-tuning job
# This class groups all hyperparameters and provides documentation and default values.


class MetaMathTuningConfig(BaseModel):
    """Configuration settings for the MetaMath fine-tuning job."""

    base_model: str = Field(
        default="meta/llama3_1@llama-3.1-8b",
        description="The base model to fine-tune, corresponding to the 7B model in the paper.",
    )
    tuning_mode: Literal["FULL", "PEFT_ADAPTER"] = Field(
        default="FULL",
        description="The tuning mode. We use 'FULL' to replicate the paper's method for the 7B model.",
    )
    epochs: int = Field(
        default=2,
        description="Number of training epochs, as specified in the MetaMath paper.",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="The learning rate for the optimizer, matching the paper's value for full fine-tuning.",
    )

# Create an instance of the configuration class.
config = MetaMathTuningConfig()

# Specify output paths for storing model artifacts.
output_uri = f"{BUCKET_URI}/tuning-output/{uuid.uuid4()}"
model_artifacts_gcs_uri = os.path.join(
    output_uri, "postprocess/node-0/checkpoints/final"
)

print (f"Loading model {config.base_model} for fine-tuning from model garden in Vertex AI...")
source_model = SourceModel(base_model=config.base_model)
print (f"Model {config.base_model} loaded successfully.")

# Create finetuning job
sft_tuning_job = sft.preview_train(
    source_model=source_model,
    tuning_mode=config.tuning_mode,
    epochs=config.epochs,
    learning_rate=config.learning_rate,
    train_dataset=datasets_gcs_uri["gcs_train_uri"],
    validation_dataset=datasets_gcs_uri["gcs_validation_uri"],
    output_uri=output_uri,
)


print(
    "Monitoring job... This will take several hours. You can safely close this notebook and come back later."
)

while not sft_tuning_job.state in [
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_SUCCEEDED,
]:
    time.sleep(600)  # Check status every 10 minutes
    sft_tuning_job.refresh()
    print(f"Current job state: {str(sft_tuning_job.state.name)}")

print(f"Job finished with state: {sft_tuning_job.state.name}")
