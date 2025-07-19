import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from src.predict.predictor import ModelPredictor
from src.serve.input_schema import InputData

# Configure logging
logging.basicConfig(level=logging.INFO)

# Determine environment from ENV variable, default 'local'
ENV = os.getenv("ENV", "local")


def get_model_uri(env: str) -> str:
    """
    Determine model URI based on environment.
    """
    if env == "local":
        return "models/random_forest.joblib"
    else:
        return "models:/RandomForestExoplanet@production"


# Load model once at module load time
MODEL_URI = get_model_uri(ENV)
COLUMNS_PATH = "models/expected_columns.json"

try:
    predictor = ModelPredictor(model_uri=MODEL_URI, columns_path=COLUMNS_PATH)
    logging.info(f"ModelPredictor initialized with model_uri='{MODEL_URI}'")
except Exception as e:
    logging.error(f"Failed to initialize ModelPredictor: {e}")
    raise RuntimeError("Failed to load model or expected columns.")


def get_predictor() -> ModelPredictor:
    """
    Dependency that returns the already loaded ModelPredictor instance.
    """
    return predictor


# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="API to predict exoplanets using a RandomForest model",
    version="1.0.0",
)


@app.get("/")
def read_root():
    """
    Root endpoint returning a welcome message.
    """
    return {"message": "Welcome to the Exoplanet Detection API"}


@app.post("/predict", response_model=dict)
def predict(data: InputData, predictor: ModelPredictor = Depends(get_predictor)):
    """
    Prediction endpoint using the shared ModelPredictor instance.
    """
    try:
        result = predictor.predict(data.input)
        return result
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9696)
