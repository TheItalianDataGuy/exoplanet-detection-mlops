import pandas as pd
import mlflow.sklearn
import json
import os


class ModelPredictor:
    def __init__(
        self, model_uri: str, columns_path: str = "models/expected_columns.json"
    ):
        """
        Initialize the ModelPredictor.

        Parameters:
        - model_uri: MLflow model URI. Can be a local path or a model registry URI (e.g., "models:/MyModel@Production")
        - columns_path: Path to the JSON file listing expected input columns for the model
        """

        # If it's a local model, check that the path exists
        if not model_uri.startswith("models:/") and not os.path.exists(model_uri):
            raise FileNotFoundError(f"Local model path does not exist: {model_uri}")

        # Load model using MLflow's loader
        self.model = mlflow.sklearn.load_model(model_uri)

        # Load expected input feature names
        if not os.path.exists(columns_path):
            raise FileNotFoundError(f"Expected columns file not found: {columns_path}")
        with open(columns_path, "r") as f:
            self.expected_columns = json.load(f)

        # Mapping from model output class (int) to human-readable labels
        self.label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

    def predict(self, input_data: list[dict]) -> dict:
        """
        Make predictions on input data.

        Parameters:
        - input_data: A list of dictionaries where each dictionary represents one input sample.

        Returns:
        - A dictionary with raw predictions and corresponding class labels.
        """

        # Optional: Restrict batch size for performance and safety
        if len(input_data) > 1000:
            raise ValueError("Batch size too large. Max 1000 records.")

        # Convert input to DataFrame
        df = pd.DataFrame(input_data)

        # Check for missing or extra columns compared to expected schema
        missing = [col for col in self.expected_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in self.expected_columns]

        if missing or extra:
            raise ValueError(
                f"Column mismatch:\nMissing: {missing}\nUnexpected: {extra}"
            )

        # Reorder and subset columns to match model expectations
        df = df[self.expected_columns]

        # Generate predictions
        predictions = self.model.predict(df)
        labels = [self.label_map[int(p)] for p in predictions]

        return {
            "predictions": predictions.tolist(),  # Raw model output
            "labels": labels,  # Human-readable class labels
        }
