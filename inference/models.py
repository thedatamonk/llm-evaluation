from pydantic import BaseModel, Field
from typing import List

# Define the request body schema using Pydantic for validation
class PredictionInstance(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, description="Maximum number of tokens to generate.")
    
class PredictionRequest(BaseModel):
    instances: List[PredictionInstance]
