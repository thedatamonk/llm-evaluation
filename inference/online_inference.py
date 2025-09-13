from google.cloud import aiplatform

PROJECT_ID = "710805310428"
location = "us-central1"
ENDPOINT_ID = "1229512385086095360"

# Initialise the AI platform client
aiplatform.init(
    project=PROJECT_ID,
    location=location
)

# Get the endpoint resource
endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{PROJECT_ID}/locations/{location}/endpoints/{ENDPOINT_ID}"
)

print (f"Endpoint:\n{endpoint}")

instances = [
        {
          "prompt": "What is Coulomb's law? Explain to a 5 yr old.",
          "max_tokens": 400
        },
        {
            "prompt": "Who was the father of electricity?",
            "max_tokens": 100
        }
    ]

predictions = endpoint.predict(instances=instances).predictions

for pred in predictions:
    print (pred["predictions"][0])
    print ("=="*50)

