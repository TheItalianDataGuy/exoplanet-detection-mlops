import pandas as pd
import joblib


class ModelPredictor:
    def __init__(self, model_path: str, expected_columns: list[str]):
        self.model = joblib.load(model_path)
        self.expected_columns = expected_columns
        self.label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

    def predict(self, input_data: list[dict]) -> dict:
        df = pd.DataFrame(input_data)
        df = df[self.expected_columns]

        predictions = self.model.predict(df)
        labels = [self.label_map[int(p)] for p in predictions]

        return {"predictions": predictions.tolist(), "labels": labels}
