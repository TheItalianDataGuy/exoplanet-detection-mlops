from fastapi import FastAPI, HTTPException, Body
import json
from src.predict.predictor import ModelPredictor
from src.serve.input_schema import InputData

# Constants
MODEL_PATH = "models/random_forest.joblib"
COLUMNS_PATH = "models/expected_columns.json"
SAMPLE_PATH = "sample_input.json"

# Load example for docs
try:
    with open(SAMPLE_PATH, "r") as f:
        sample_input = json.load(f)
        example = {"input": sample_input[:1]}
except Exception as e:
    print(f"Failed to load sample input: {e}")
    example = {"input": [{"error": "failed to load"}]}

# Initialize predictor
predictor = ModelPredictor(
    "models:/RandomForestExoplanet@production", columns_path=COLUMNS_PATH
)

app = FastAPI(
    title="Exoplanet Detection API",
    description="A simple API that uses a RandomForest model to predict exoplanets",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API"}


@app.post("/predict", response_model=dict)
def predict(data: InputData = Body(..., example=example)):
    try:
        return predictor.predict(data.input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}
