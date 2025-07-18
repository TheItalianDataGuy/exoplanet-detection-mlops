import json
import os
import pytest

from src.predict.predictor import ModelPredictor

# Paths to model and sample input
MODEL_PATH = "models/random_forest.joblib"
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
def expected_columns(sample_input):
    """
    Infer expected column names from the sample input.
    """
    return list(sample_input[0].keys())


@pytest.fixture(scope="module")
def predictor(expected_columns):
    """
    Load the model and initialize the predictor with expected column names.
    """
    assert os.path.exists(MODEL_PATH), f"Model file not found at: {MODEL_PATH}"
    return ModelPredictor(MODEL_PATH, expected_columns)


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
