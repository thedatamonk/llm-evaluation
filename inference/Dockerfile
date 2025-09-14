# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Cloud Run injects a PORT environment variable, so listen on it
CMD ["uvicorn", "inference-api-server:app", "--host", "0.0.0.0", "--port", "8080"]