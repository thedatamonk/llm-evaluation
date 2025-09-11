# uploading the trained huggingface model to huggingface


# load it back

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from llm_app import llm_call
import re
import json
import os

red_teamer_model_path = "./alpharedteam"

# red_teamer_tokenizer = AutoTokenizer.from_pretrained(red_teamer_model_path)
# red_teamer_model = AutoModelForCausalLM.from_pretrained(red_teamer_model_path)
red_teamer_model = pipeline("text-generation", model=red_teamer_model_path, tokenizer=red_teamer_model_path)

calls_made = 0
keep_going = True
last_response = None
turns = []
max_calls = 10

# TODO: We need to format the messages before feeding it into the model
target_to_attack = lambda prompt: llm_call(messages=[{"role": "redteam", "content": prompt}], stream=False)

while calls_made < max_calls and keep_going:
    if last_response == "":
        challenge = ""
    else:
        if last_response:
            last_response_first_sent = last_response.split(". ")[0]
        else:
            last_response_first_sent = ""
        
        # wrap the target's response in the markup used in training
        query = f"<|input|>{last_response_first_sent}<|response|>"
        challenge = red_teamer_model(query)
        challenge = challenge[0]['generated_text']

        # strip the prompt out from the front of the model response
        challenge = re.sub("^" + re.escape(query), "", challenge)

        # unwrap the recommended challenge from the markup
        challenge = re.sub("\<\|.*", "", challenge).strip()

    turn = ("probe", challenge)
    turns.append(turn)

    # send the challenge and get the response
    response = target_to_attack(challenge).strip()
    turn = ("model", response)
    turns.append(turn)

    # increment calls_made
    calls_made += 1

    # check if the resp is empty or if it matches the previous resp
    if not len(response):
        keep_going = False
    if response == last_response:
        keep_going = False

    # update last_response
    last_response = response.replace("\n", " ").strip()

# store the probes and responses in a json
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/red_team_results.json", "w") as f:
    json.dump(turns, f, indent=2)