import pytest
import json
from typing import List, Dict, Any
from src.predict.predictor import ModelPredictor

# Constants for file paths
SAMPLE_INPUT_PATH = "sample_input.json"
MODEL_PATH = "models/random_forest.joblib"
COLUMNS_PATH = "models/expected_columns.json"


@pytest.fixture(scope="module")
def predictor() -> ModelPredictor:
    """
    Fixture to initialize and return the ModelPredictor instance once per test module.
    """
    return ModelPredictor(model_uri=MODEL_PATH, columns_path=COLUMNS_PATH)


@pytest.fixture
def sample_input() -> List[Dict[str, Any]]:
    """
    Fixture to load and return a small sample input list from JSON for testing.
    Reads first 2 samples to keep tests fast.
    """
    with open(SAMPLE_INPUT_PATH, "r") as f:
        data = json.load(f)
    return data[:2]


def test_predictor_returns_valid_output(
    predictor: ModelPredictor, sample_input: List[Dict[str, Any]]
):
    """
    Test that the predictor returns a dictionary with expected keys and
    correct length outputs given valid input.
    """
    result = predictor.predict(sample_input)

    # Verify result is a dictionary with expected keys
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "predictions" in result, "'predictions' key missing in result"
    assert "labels" in result, "'labels' key missing in result"

    # Verify the length of predictions matches input samples
    assert len(result["predictions"]) == len(
        sample_input
    ), "Mismatch in predictions count"

    # Verify all labels are strings
    assert all(
        isinstance(label, str) for label in result["labels"]
    ), "All labels should be strings"


def test_predictor_raises_on_invalid_input(predictor: ModelPredictor):
    """
    Test that the predictor raises an exception or handles invalid input properly.
    """
    invalid_input: Any = "this is not a valid input"

    with pytest.raises(Exception):
        predictor.predict(invalid_input)
