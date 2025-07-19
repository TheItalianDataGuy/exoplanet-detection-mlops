import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.serve.main import app

client = TestClient(app)


@pytest.fixture
def mock_predictor():
    """
    Patch the global predictor instance in src.serve.main.
    """
    with patch("src.serve.main.predictor") as mock:
        yield mock


def test_root_endpoint():
    """
    Test the root GET endpoint returns expected message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Exoplanet Detection API"}


def test_health_endpoint():
    """
    Test the health check GET endpoint returns 'ok'.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input(mock_predictor):
    """
    Test POST /predict with valid input returns mocked prediction.
    """
    mock_predictor.predict.return_value = {
        "predictions": ["planet"],
        "labels": ["planet"],
    }

    payload = {
        "input": [
            {
                "koi_period": 1.0,
                "koi_duration": 2.5,
            }  # Your InputData example features here
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"predictions": ["planet"], "labels": ["planet"]}
    mock_predictor.predict.assert_called_once_with(payload["input"])


def test_predict_invalid_input(mock_predictor):
    """
    Test POST /predict that raises an exception in predictor returns HTTP 400.
    """
    mock_predictor.predict.side_effect = Exception("Model failure")

    payload = {"input": [{"koi_period": 1.0, "koi_duration": 2.5}]}

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Prediction failed" in response.json()["detail"]
    mock_predictor.predict.assert_called_once_with(payload["input"])
