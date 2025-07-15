# 🚀 FastAPI Inference API for Exoplanet Detection

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import joblib
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="A simple API that uses a RandomForest model to predict exoplanets",
    version="1.0.0",
)

# Define paths for the model and sample input
MODEL_PATH = "models/random_forest.joblib"
SAMPLE_PATH = "sample_input.json"

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Load sample input to extract feature order
try:
    with open(SAMPLE_PATH, "r") as f:
        sample_input = json.load(f)
    expected_columns = list(sample_input[0].keys())
except Exception as e:
    raise RuntimeError(f"Failed to load sample input from {SAMPLE_PATH}: {e}")

# Define input schema using Pydantic
class InputData(BaseModel):
    input: List[Dict] = Field(..., example=sample_input)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data.input)

        # Ensure correct column order for prediction
        df = df[expected_columns]

        # Generate predictions
        predictions = model.predict(df)

        # Map numeric predictions to labels
        label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
        labels = [label_map[int(pred)] for pred in predictions]

        return {
            "predictions": predictions.tolist(),
            "labels": labels
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
