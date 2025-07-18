import pandas as pd
import joblib
import json
import os

# Try importing mlflow
try:
    import mlflow.pyfunc
except ImportError:
    mlflow = None


class ModelPredictor:
    def __init__(
        self, model_uri: str, columns_path: str = "models/expected_columns.json"
    ):
        """
        Initialize the predictor with a model and expected columns.

        Supports loading from local joblib or MLflow registry URI.
        """
        # Load expected columns
        if not os.path.exists(columns_path):
            raise FileNotFoundError(f"Expected columns file not found: {columns_path}")
        with open(columns_path, "r") as f:
            self.expected_columns = json.load(f)

        # Check if URI starts with MLflow format
        if model_uri.startswith("models:/"):
            if mlflow is None:
                raise ImportError(
                    "mlflow is not installed but MLflow model URI was provided."
                )
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.is_mlflow_model = True
        else:
            if not os.path.exists(model_uri):
                raise FileNotFoundError(f"Model file does not exist: {model_uri}")
            self.model = joblib.load(model_uri)
            self.is_mlflow_model = False

        # Labels only for scikit-learn model
        self.label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

    def predict(self, input_data: list[dict]) -> dict:
        """
        Run prediction on the input list of dictionaries.
        """
        if len(input_data) > 1000:
            raise ValueError("Batch size too large. Max 1000 records.")

        df = pd.DataFrame(input_data)

        # Check columns
        missing = [col for col in self.expected_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in self.expected_columns]
        if missing or extra:
            raise ValueError(
                f"Column mismatch:\nMissing: {missing}\nUnexpected: {extra}"
            )

        df = df[self.expected_columns]

        # Predict
        predictions = self.model.predict(df)
        labels = [self.label_map[int(p)] for p in predictions]

        return {
            "predictions": predictions.tolist(),
            "labels": labels,
        }
