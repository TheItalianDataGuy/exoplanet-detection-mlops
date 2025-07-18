from fastapi import FastAPI, HTTPException
from src.predict.predictor import ModelPredictor
from src.serve.input_schema import InputData
import logging
import os

# Read environment setting (local or prod)
ENV = os.getenv("ENV", "local")

# Set model path depending on environment
MODEL_URI = (
    "models/random_forest.joblib"
    if ENV == "local"
    else "models:/RandomForestExoplanet@production"
)
COLUMNS_PATH = "models/expected_columns.json"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load model
try:
    predictor = ModelPredictor(model_uri=MODEL_URI, columns_path=COLUMNS_PATH)
    logging.info(f"ModelPredictor initialized in '{ENV}' mode.")
except Exception as e:
    logging.error(f"Failed to initialize ModelPredictor: {e}")
    raise RuntimeError("Failed to load model or expected columns.")

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="A simple API that uses a RandomForest model to predict exoplanets",
    version="1.0.0",
)


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Exoplanet Detection API"}


# Prediction endpoint
@app.post("/predict", response_model=dict)
def predict(data: InputData):
    try:
        result = predictor.predict(data.input)
        return result
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}
