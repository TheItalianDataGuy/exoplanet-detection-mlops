import pytest
from predict.predictor import ModelPredictor

# Replace with the same expected columns used in your training pipeline
EXPECTED_COLUMNS = ["feature1", "feature2", "feature3", "feature4"]

@pytest.fixture(scope="module")
def predictor():
    return ModelPredictor("models/random_forest.joblib", EXPECTED_COLUMNS)

@pytest.fixture
def sample_input():
    return [
        {"feature1": 0.1, "feature2": 2.3, "feature3": 1.1, "feature4": 0.4},
        {"feature1": 0.3, "feature2": 1.9, "feature3": 3.5, "feature4": 0.1}
    ]

def test_predictor_returns_valid_output(predictor, sample_input):
    result = predictor.predict(sample_input)

    assert isinstance(result, dict)
    assert "predictions" in result
    assert "labels" in result
    assert len(result["predictions"]) == len(sample_input)
    assert len(result["labels"]) == len(sample_input)
    assert all(isinstance(label, str) for label in result["labels"])
