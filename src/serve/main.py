from fastapi import FastAPI, HTTPException
import json
from src.predict.predictor import ModelPredictor
from src.serve.input_schema import InputData


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
predictor = ModelPredictor("models:/RandomForestExoplanet@production")

# FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="A simple API that uses a RandomForest model to predict exoplanets",
    version="1.0.0",
)


# Root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API"}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        return predictor.predict(data.input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
