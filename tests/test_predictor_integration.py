import json
import os
import pytest

from src.predict.predictor import ModelPredictor

MODEL_PATH = "models/random_forest.joblib"
SAMPLE_INPUT_PATH = "sample_input.json"

# Use the following marker to run integration tests selectively:
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sample_input():
    with open(SAMPLE_INPUT_PATH, "r") as f:
        return json.load(f)[:3]  # use only first 3 rows for speed


@pytest.fixture(scope="module")
def expected_columns(sample_input):
    return list(sample_input[0].keys())


@pytest.fixture(scope="module")
def predictor(expected_columns):
    assert os.path.exists(MODEL_PATH), f"Model file not found at: {MODEL_PATH}"
    return ModelPredictor(MODEL_PATH, expected_columns)


def test_predictor_with_real_model(sample_input, predictor):
    result = predictor.predict(sample_input)

    assert isinstance(result, dict)
    assert "predictions" in result
    assert "labels" in result
    assert len(result["predictions"]) == len(sample_input)
    assert all(isinstance(label, str) for label in result["labels"])
