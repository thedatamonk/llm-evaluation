from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import Conflict
import json
from datasets import load_dataset, concatenate_datasets
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="./finetuning/.env")

PROJECT_ID = os.getenv("PROJECT_ID", "")
LOCATION = os.getenv("LOCATION", "us-central1")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", f"{PROJECT_ID}_finetuning-datasets")

def create_gcs_bucket(bucket_name, project_id, location):
    """Creates a new bucket in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the bucket to create.
        project_id (str): The Google Cloud project ID. If not provided,
                          it will be inferred from the environment.
    """
    try:
        client = storage.Client(project=project_id)

        bucket = client.create_bucket(bucket_or_name=bucket_name, location=location)

        print (f"Bucket {bucket.name} created in {bucket.location} created successfully.")

        return bucket
    except DefaultCredentialsError as e:
        print("Authentication failed.")
        print("To authenticate, please run one of the following commands:")
        print("1. For user authentication: 'gcloud auth application-default login'")
        print("2. For service account key: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("   Example: 'export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/keyfile.json'")
        return None
    except Conflict:
        print(f"Bucket '{bucket_name}' already exists. Returning the existing bucket.")
        bucket = client.get_bucket(bucket_name)
        return bucket
    except Exception as e:
        print (f"Error creating bucket: {e}")
        return None

def upload_to_gcs(source_file_name, destination_blob_name, bucket_name=None):
    """Uploads a file to a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name or BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name} in bucket {bucket.name}.")

def save_to_jsonl(dataset, output_path):
    """Save dataset in JSONL format required by Vertex AI."""
    with open(output_path, 'w') as f:
        for example in dataset:
            json.dump(example, f)
            f.write("\n")
    print(f"Saved {len(dataset)} examples to {output_path}")

def prepare_metamathqa_dataset():
    # Load the MetaMathQA dataset from the Hugging Face Hub.
    dataset_dir = "./datasets/metamathqa"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset = load_dataset("meta-math/MetaMathQA")["train"]

    # We'll use the 'GSM8K' configuration, which is a key part of the paper's contribution.
    
    # Split the dataset into training and validation sets (80-20 split).
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # The result is a dictionary containing the two new splits.
    train_split = split_dataset["train"]
    validation_split = split_dataset["test"]

    # Limit validation dataset to less than 5000 rows for Vertex AI requirement
    # The remaining rows should be added to training dataset
    if len(validation_split) > 5000:
        validation_split = validation_split.shuffle(seed=42)
        validation_keep = validation_split.select(range(5000))
        validation_remainder = validation_split.select(range(5000, len(validation_split)))

        train_split = concatenate_datasets([train_split, validation_remainder])
        validation_split = validation_keep
        del validation_keep, validation_remainder
    
    print (f"Training set size: {len(train_split)}")
    print (f"Validation set size: {len(validation_split)}")

    # apply prompt template to each data sample
    train_split = apply_prompt_template(train_split)
    validation_split = apply_prompt_template(validation_split)

    train_file_path = f"{dataset_dir}/metamath_gsm8k_train.jsonl"
    validation_file_path = f"{dataset_dir}/metamath_gsm8k_validation.jsonl"

    train_dataset_gcs_blob_name = f"metamathqa/{os.path.basename(train_file_path)}"
    validation_dataset_gcs_blob_name = f"metamathqa/{os.path.basename(validation_file_path)}"

    
    # Write the formatted training data to a local JSONL file.
    save_to_jsonl(train_split, train_file_path)
    save_to_jsonl(validation_split, validation_file_path)

    # Upload the JSONL files to GCS
    upload_to_gcs(source_file_name=train_file_path,
                  destination_blob_name=train_dataset_gcs_blob_name)
    
    upload_to_gcs(source_file_name=validation_file_path,
                  destination_blob_name=validation_dataset_gcs_blob_name)
    
    return {
        "gcs_train_uri": f"gs://{BUCKET_NAME}/{train_dataset_gcs_blob_name}", 
        "gcs_validation_uri": f"gs://{BUCKET_NAME}/{validation_dataset_gcs_blob_name}"
    }


def apply_prompt_template(dataset):
    # MetaMath's instruction template
    METAMATH_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""

    def _format_template(example):
        query = example["query"]
        response = example["response"]

        instruction = METAMATH_TEMPLATE.format(instruction=query)

        # Important: Add space before response for proper tokenization
        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": f" {response}"},
            ]
        }
    

    formatted_dataset = dataset.map(_format_template, remove_columns=dataset.column_names, num_proc=os.cpu_count())
    print (f"Applied prompt template to {len(formatted_dataset)} examples.")
    return formatted_dataset

if __name__ == "__main__":
    # Example usage
    # TEST 1: Create a GCS bucket
    # PROJECT_NUMBER = "710805310428"
    # PROJECT_ID = "maximal-brace-132923"
    # LOCATION = "us-central1"
    # bucket = create_gcs_bucket(f"{PROJECT_ID}_finetuning-datasets", project_id=PROJECT_ID, location=LOCATION)

    # TEST 2: Prepare the MetaMathQA dataset and split into train and validation sets
    dataset_gcs_paths = prepare_metamathqa_dataset()