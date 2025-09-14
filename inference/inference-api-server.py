from fastapi import FastAPI, HTTPException
from google.cloud import aiplatform
from models import PredictionInstance, PredictionRequest

# Replace with your actual project details
PROJECT_ID = "710805310428"
LOCATION = "us-central1"
ENDPOINT_ID = "1229512385086095360"

app = FastAPI()


# Initialize the AI Platform client outside the endpoint to reuse the connection
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
    )
except Exception as e:
    # Handle the case where the client cannot be initialized
    print(f"Error initializing Vertex AI client: {e}")
    endpoint = None

@app.post("/predict")
async def get_predictions(request_body: PredictionRequest):
    """
    Sends a prediction request to a Vertex AI Endpoint.
    """

    if not endpoint:
        raise HTTPException(
            status_code=503,
            detail="Service is unavailable. Vertex AI client failed to initialize."
        )

    try:
        instances = [instance.model_dump(exclude_none=True) for instance in request_body.instances]
        print (f"Total instances found: {len(instances)}")


        # Call the Vertex AI endpoint
        response = endpoint.predict(instances=instances)

        # Extract and return the predictions
        return {"predictions": response.predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health():
    return {"status": "ok"}