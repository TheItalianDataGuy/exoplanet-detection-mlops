from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
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

# Load trained model
MODEL_PATH = "models/random_forest.joblib"
model = joblib.load(MODEL_PATH)

# Load sample to get correct feature order
SAMPLE_PATH = "sample_input.json"
with open(SAMPLE_PATH) as f:
    sample_input = json.load(f)

expected_columns = list(sample_input[0].keys())

# Dynamically create the input schema using BaseModel
class InputData(BaseModel):
    input: list[dict] = Field(..., example=sample_input)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API"}

# Predict endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data.input)

        # Enforce column order to match training
        df = df[expected_columns]

        # Make predictions
        predictions = model.predict(df)
        label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
        labels = [label_map[int(pred)] for pred in predictions]

        return {
            "predictions": predictions.tolist(),
            "labels": labels,
        }

    except Exception as e:
        return {"error": str(e)}
