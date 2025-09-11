# How to implement red teaming?

# Use an off-the-shelf toxicity detector, martin-ha/toxic-comment-model

# Look at an existing red teaming dataset, the red team attempts from Anthropic’s hhrlhf

# Find system dialog turns that were rated toxic, and extract dialog turns in the conversations that led to that toxicity

# Train a 2019 gpt-2 to red-team based on this data

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TextClassificationPipeline
)

from datasets import load_dataset

import re
import os
import json
import tqdm


LOAD_HF_MODEL = True

if LOAD_HF_MODEL:
    toxicity_detector_model_path = "martin-ha/toxic-comment-model"
    detector_target_class = "toxic"
    # Initialise detector model and tokenizer
    detector_model = AutoModelForSequenceClassification.from_pretrained(toxicity_detector_model_path)
    detector_tokenizer = AutoTokenizer.from_pretrained(toxicity_detector_model_path)
    detector = TextClassificationPipeline(
        model=detector_model, tokenizer=detector_tokenizer
    )

DOWNLOAD_DATASET = True
RED_TEAMING_DATASET_NAME = "Anthropic/hh-rlhf"
SUBSET_NAME = "red-team-attempts"
LOCAL_DATASET_DIR = "./datasets"
local_file_path = os.path.join(LOCAL_DATASET_DIR, f"{SUBSET_NAME}.jsonl")

if DOWNLOAD_DATASET:
    if os.path.exists(local_file_path):
        print(f"File '{local_file_path}' already exists. Skipping download from Hugging Face.")
    else:
        print(f"File not found. Downloading dataset '{RED_TEAMING_DATASET_NAME}/{SUBSET_NAME}'...")

        os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

        dataset = load_dataset(RED_TEAMING_DATASET_NAME, data_dir=SUBSET_NAME)

        dataset['train'].to_json(f"./datasets/{SUBSET_NAME}.jsonl")

        print("Dataset successfully downloaded and converted to JSONL.")


print (f"Now converting dataset to a pretty looking format.")
attempts = []
with open(local_file_path, "r") as infile:
    for line in infile:
        try:
            attempts.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line: {e}")

outfile = open(f"{LOCAL_DATASET_DIR}/hhrlhf-toxic.jsonl", "w", buffering=1)


for attempt in tqdm.tqdm(attempts):

    # extracting transcript section
    transcript = re.split(r"\n\n([A-Z][a-z]+): ", attempt["transcript"])
    if not transcript[0].strip():
        transcript.pop(0)
    toxic = False
    turns = []

    # for each convo, extract the actor and the utterance and cleanup the utterance
    for i in range(0, len(transcript), 2):
        actor = transcript[i]
        utterance = (
            transcript[i + 1]
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2013", "-")
            .replace("  ", " ")
            .replace("\u2019", "'")
        )

        turns.append((actor, utterance))

    # We want to detect which system (Assistant) responses were toxic
    detector_results = detector([turn[1] for turn in turns if turn[0] != "Human"])
    # Why are we trying to check the detector label for the first turn only?
    if detector_results[0]["label"] == detector_target_class:
        toxic = True

    # Only if the detected label is toxic
    if toxic:
        for i in range(len(turns) - 1):
            if i == 0:
                outfile.write(
                    json.dumps({"prompt": "", "response": turns[i][1]}) + "\n"
                )
            if turns[i][0] == "Assistant":
                outfile.write(
                    json.dumps({"prompt": turns[i][1], "response": turns[i + 1][1]})
                    + "\n"
                )

with open(f"{LOCAL_DATASET_DIR}/ft_toxic_dataset.jsonl", "w") as tmp_json:
    for line in open(os.path.join(f"{LOCAL_DATASET_DIR}/hhrlhf-toxic.jsonl"), "r"):
        record = json.loads(line.strip())
        turn_resp = "<|input|>" + record["prompt"] + "<|response|>" + record["response"]
        tmp_json.write(json.dumps({"turn_resp": turn_resp}))

print (f"Finetuning dataset prepared.")

