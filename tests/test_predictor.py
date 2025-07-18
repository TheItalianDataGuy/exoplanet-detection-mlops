import pytest
import json
from src.predict.predictor import ModelPredictor

SAMPLE_INPUT_PATH = "sample_input.json"
MODEL_PATH = "models/random_forest.joblib"

@pytest.fixture(scope="module")
def expected_columns():
    with open(SAMPLE_INPUT_PATH, "r") as f:
        sample = json.load(f)
    return list(sample[0].keys())

@pytest.fixture(scope="module")
def predictor(expected_columns):
    return ModelPredictor(MODEL_PATH, expected_columns)

@pytest.fixture
def sample_input():
    with open(SAMPLE_INPUT_PATH, "r") as f:
        return json.load(f)[:2]  # Use first 2 rows for test

def test_predictor_returns_valid_output(predictor, sample_input):
    result = predictor.predict(sample_input)

    assert isinstance(result, dict)
    assert "predictions" in result
    assert "labels" in result
    assert len(result["predictions"]) == len(sample_input)
    assert all(isinstance(label, str) for label in result["labels"])
