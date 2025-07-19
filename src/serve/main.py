import logging
from fastapi import FastAPI, HTTPException, Depends

from src.predict.predictor import ModelPredictor
from src.serve.input_schema import InputData
from config.settings import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_model_uri(env: str) -> str:
    """
    Resolve the model URI depending on the execution environment.

    Args:
        env (str): Environment type (e.g. 'local', 'prod', 'staging').

    Returns:
        str: Path or URI to the trained model artifact.
    """
    if env == "local":
        return settings.model_path
    elif env == "prod":
        return "models:/RandomForestExoplanet@production"
    elif env == "staging":
        return "models:/RandomForestExoplanet@staging"
    else:
        raise ValueError(f"Unsupported ENV: {env}")


# Initialize paths to model and expected columns
model_uri = get_model_uri(settings.env)
columns_path = settings.model_path.replace(
    "random_forest.joblib", "expected_columns.json"
)

# Load model and expected input schema
try:
    predictor = ModelPredictor(model_uri=model_uri, columns_path=columns_path)
    logging.info(f"ModelPredictor initialized with model_uri='{model_uri}'")
except Exception as e:
    logging.error(f"Failed to initialize ModelPredictor: {e}")
    raise RuntimeError("Failed to load model or expected columns.")


def get_predictor() -> ModelPredictor:
    """
    FastAPI dependency injection for the shared ModelPredictor instance.

    Returns:
        ModelPredictor: The loaded and validated model instance.
    """
    return predictor


# Initialize FastAPI app
app = FastAPI(
    title="🪐 Exoplanet Detection API",
    description="API to predict exoplanets using a RandomForest model",
    version="1.0.0",
)


@app.get("/")
def read_root():
    """
    Root endpoint for connectivity check.
    """
    return {"message": "Welcome to the Exoplanet Detection API"}


@app.post("/predict", response_model=dict)
def predict(data: InputData, predictor: ModelPredictor = Depends(get_predictor)):
    """
    Predict the presence of an exoplanet given the input features.

    Args:
        data (InputData): Features in the correct format for prediction.
        predictor (ModelPredictor): Injected instance of the predictor.

    Returns:
        dict: Prediction result.
    """
    try:
        return predictor.predict(data.input)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    """
    Health check endpoint to confirm service availability.

    Returns:
        dict: Status OK.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve.main:app", host="127.0.0.1", port=8000, reload=False)
