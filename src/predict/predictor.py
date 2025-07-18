import pandas as pd
import mlflow.sklearn
import json
import os


class ModelPredictor:
    def __init__(
        self, model_uri: str, columns_path: str = "models/expected_columns.json"
    ):
        """
        model_uri: example -> 'models:/RandomForestExoplanet@production'
        columns_path: path to the JSON file containing expected input column names
        """
        # Load model from MLflow model registry
        self.model = mlflow.sklearn.load_model(model_uri)

        # Load expected input columns from JSON file
        if not os.path.exists(columns_path):
            raise FileNotFoundError(
                f"Expected columns file not found at {columns_path}"
            )
        with open(columns_path, "r") as f:
            self.expected_columns = json.load(f)

        # Map model output to human-readable labels
        self.label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

    def predict(self, input_data: list[dict]) -> dict:
        # Convert input list of dicts to DataFrame
        df = pd.DataFrame(input_data)

        # Ensure the input matches expected columns
        try:
            df = df[self.expected_columns]
        except KeyError as e:
            raise ValueError(f"Missing or unexpected columns: {e}")

        # Predict using the loaded model
        predictions = self.model.predict(df)
        labels = [self.label_map[int(p)] for p in predictions]

        return {"predictions": predictions.tolist(), "labels": labels}
