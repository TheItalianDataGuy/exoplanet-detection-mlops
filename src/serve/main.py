# src/serve/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()

# Load model once at startup
MODEL_PATH = os.path.join("mlruns", "0", "737778685699943073", "artifacts", "model")
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Define input schema
class ExoplanetInput(BaseModel):
    features: list[float]  # a list of 40 features

@app.get("/")
def read_root():
    return {"message": "Exoplanet Detection API is running!"}

@app.post("/predict")
def predict(data: ExoplanetInput):
    # Convert input list to DataFrame
    input_df = pd.DataFrame([data.features])
    
    # Predict
    prediction = model.predict(input_df)
    
    return {"prediction": int(prediction[0])}
