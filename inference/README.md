# Inference API Server for Vertex AI LLM

This folder contains code to serve an API for performing inference using a deployed Large Language Model (LLM) on a Google Vertex AI endpoint.

## Folder Structure

- **inference-api-server.py**  
  FastAPI server that exposes a `/predict` endpoint to send prediction requests to a Vertex AI endpoint and a `/health` endpoint for health checks.

- **models.py**  
  Pydantic models defining the request and instance schema for the API.

- **online_inference.py**  
  Example script for making direct prediction calls to the Vertex AI endpoint without using the API server.


## Setup Instructions

1. **Install dependencies**

   Make sure you have Python 3.8+ and install the required packages in a virtual environment:

   ```sh
   pip install -r requirements.txt
   ```

2. **Set up Google Cloud authentication**

   - Ensure you have access to the target Vertex AI endpoint.
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key JSON file:

     ```sh
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
     ```

3. **Configure project details**

   - Edit `PROJECT_ID`, `LOCATION`, and `ENDPOINT_ID` in [inference-api-server.py](inference/inference-api-server.py) and [online_inference.py](inference/online_inference.py) to match your Vertex AI deployment.

4. **Run the API server**

   ```sh
   uvicorn inference.inference-api-server:app --reload
   ```

   The server will be available at `http://127.0.0.1:8000`.

5. **Send a prediction request**

   - POST to `/predict` with a JSON body matching the `PredictionRequest` model defined in [models.py](inference/models.py).

6. **Health check**

   - GET `/health` returns `{"status": "ok"}` if the server is running.

## File Descriptions

- **inference-api-server.py**  
  Main FastAPI application. Handles initialization of the Vertex AI client and exposes endpoints for prediction and health checks.

- **models.py**  
  Contains Pydantic models:
  - `PredictionInstance`: Defines the schema for a single prediction input.
  - `PredictionRequest`: Defines the schema for the request body containing multiple instances.

- **online_inference.py**  
  Standalone script to send prediction requests directly to the Vertex AI endpoint for quick testing.

---

**Note:**  
Make sure your Google Cloud credentials and endpoint details are correct before running the server

## TODO:
- **Online inference**
1. Create a YAML file to build, push and deploy this API to Cloud Run. Basically fully automate the deployment process.
2. Optimise inferencing using vllm
3. Performance monitoring of the deployed API.
- **Batch inference**
