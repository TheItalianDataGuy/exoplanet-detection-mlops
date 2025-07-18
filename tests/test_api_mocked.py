from fastapi.testclient import TestClient
from unittest.mock import patch
from src.serve.main import app

client = TestClient(app)

@patch("src.serve.main.model_predictor.predict")
def test_predict_endpoint_with_mock(mock_predict):
    # Arrange
    mock_predict.return_value = {
        "predictions": [1, 2],
        "labels": ["CANDIDATE", "CONFIRMED"]
    }

    request_payload = {
        "input": [
            {"feature1": 0.1, "feature2": 2.3, "feature3": 1.1, "feature4": 0.4},
            {"feature1": 0.3, "feature2": 1.9, "feature3": 3.5, "feature4": 0.1}
        ]
    }

    # Act
    response = client.post("/predict", json=request_payload)

    # Assert
    assert response.status_code == 200
    response_json = response.json()
    assert response_json == {
        "predictions": [1, 2],
        "labels": ["CANDIDATE", "CONFIRMED"]
    }
