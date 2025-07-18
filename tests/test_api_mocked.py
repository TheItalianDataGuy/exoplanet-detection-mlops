from fastapi.testclient import TestClient
from unittest.mock import patch
from src.serve.main import app
from src.predict.predictor import ModelPredictor

# Use FastAPI test client with startup event
client = TestClient(app)


@patch.object(ModelPredictor, "predict")
def test_predict_endpoint_with_mock(mock_predict):
    """
    Unit test: Mock the predictor to test the /predict endpoint independently of model logic.
    """
    # Arrange: Mock return value
    mock_predict.return_value = {
        "predictions": [1, 2],
        "labels": ["CANDIDATE", "CONFIRMED"],
    }

    # Sample input (dummy features)
    request_payload = {
        "input": [
            {"feature1": 0.1, "feature2": 2.3, "feature3": 1.1, "feature4": 0.4},
            {"feature1": 0.3, "feature2": 1.9, "feature3": 3.5, "feature4": 0.1},
        ]
    }

    # Act: call /predict
    response = client.post("/predict", json=request_payload)

    # Assert: check status and content
    assert response.status_code == 200
    assert response.json() == {
        "predictions": [1, 2],
        "labels": ["CANDIDATE", "CONFIRMED"],
    }
