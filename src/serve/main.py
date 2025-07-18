from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import json
from src.predict.predictor import ModelPredictor

# Constants
MODEL_PATH = "models/random_forest.joblib"
SAMPLE_PATH = "sample_input.json"

# Load expected columns from sample input
try:
    with open(SAMPLE_PATH, "r") as f:
        sample_input = json.load(f)
    expected_columns = list(sample_input[0].keys())
except Exception as e:
    raise RuntimeError(f"Failed to load sample input from {SAMPLE_PATH}: {e}")

# Initialize model predictor (loaded once)
model_predictor = ModelPredictor(MODEL_PATH, expected_columns)

# FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="A simple API that uses a RandomForest model to predict exoplanets",
    version="1.0.0",
)

# Input schema
class InputData(BaseModel):
    input: List[Dict] = Field(..., json_schema_extra={"example": sample_input})

# Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        return model_predictor.predict(data.input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
