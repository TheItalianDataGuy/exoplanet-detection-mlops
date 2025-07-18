from fastapi import FastAPI, HTTPException, Body
from src.predict.predictor import ModelPredictor
from src.serve.input_schema import InputData
import logging

# Configuration
MODEL_URI = "models:/RandomForestExoplanet@production"
COLUMNS_PATH = "models/expected_columns.json"

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the predictor
try:
    predictor = ModelPredictor(model_uri=MODEL_URI, columns_path=COLUMNS_PATH)
    logging.info("ModelPredictor initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize ModelPredictor: {e}")
    raise RuntimeError("Failed to load model or expected columns.")

# Create FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="A simple API that uses a RandomForest model to predict exoplanets",
    version="1.0.0",
)


# Root endpoint
@app.get("/")
def read_root():
    """Health check at the root"""
    return {"message": "Welcome to the Exoplanet Detection API"}


# Prediction endpoint
@app.post("/predict", response_model=dict)
def predict(data: InputData = Body(...)):
    """
    Predicts exoplanet classification based on input features.
    """
    try:
        result = predictor.predict(data.input)
        return result
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Health check endpoint
@app.get("/health")
def health_check():
    """Simple health check"""
    return {"status": "ok"}
