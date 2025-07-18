import json
import os
import pytest
from src.predict.predictor import ModelPredictor

# Paths to model and sample input
MODEL_PATH = "models/random_forest.joblib"
COLUMNS_PATH = "models/expected_columns.json"
SAMPLE_INPUT_PATH = "sample_input.json"

# Mark these tests as integration tests (can be run selectively with `-m integration`)
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sample_input():
    """
    Load a subset of sample input data for integration testing.
    """
    with open(SAMPLE_INPUT_PATH, "r") as f:
        return json.load(f)[:3]  # Use only first 3 rows for performance


@pytest.fixture(scope="module")
def predictor():
    """
    Load the model and initialize the predictor with expected columns file path.
    """
    assert os.path.exists(MODEL_PATH), f"Model file not found at: {MODEL_PATH}"
    assert os.path.exists(COLUMNS_PATH), f"Columns file not found at: {COLUMNS_PATH}"
    return ModelPredictor(MODEL_PATH, COLUMNS_PATH)


def test_predictor_with_real_model(sample_input, predictor):
    """
    Test that the predictor returns valid predictions and labels when using the real model.
    """
    result = predictor.predict(sample_input)

    # Assertions on prediction structure and validity
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "labels" in result
    assert len(result["predictions"]) == len(sample_input)
    assert all(isinstance(label, str) for label in result["labels"])
