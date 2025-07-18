import pytest
import json
from src.predict.predictor import ModelPredictor

# Constants for paths
SAMPLE_INPUT_PATH = "sample_input.json"
MODEL_PATH = "models/random_forest.joblib"
COLUMNS_PATH = "models/expected_columns.json"


@pytest.fixture(scope="module")
def predictor():
    """
    Fixture to initialize and return the ModelPredictor once per test module.
    """
    return ModelPredictor(MODEL_PATH, columns_path=COLUMNS_PATH)


@pytest.fixture
def sample_input():
    """
    Fixture to load and return a small sample input from JSON for testing.
    """
    with open(SAMPLE_INPUT_PATH, "r") as f:
        return json.load(f)[:2]  # Use first 2 rows for test


def test_predictor_returns_valid_output(predictor, sample_input):
    """
    Test that the predictor returns a valid dictionary with expected keys and values.
    """
    result = predictor.predict(sample_input)

    # Basic checks on structure and content
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "labels" in result
    assert len(result["predictions"]) == len(sample_input)
    assert all(isinstance(label, str) for label in result["labels"])
