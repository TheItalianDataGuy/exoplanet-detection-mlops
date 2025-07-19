import pytest
import json
from fastapi.testclient import TestClient
from src.serve.main import app  # Make sure PYTHONPATH includes src

# Mark these tests as integration tests
pytestmark = pytest.mark.integration

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_input():
    """
    Load a subset of sample input data for API integration testing.
    """
    with open("sample_input.json", "r") as f:
        return json.load(f)[:3]  # Use first 3 rows for performance


def test_root_endpoint():
    """
    Test GET / returns welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Exoplanet Detection API"}


def test_health_endpoint():
    """
    Test GET /health returns status OK.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid(sample_input):
    """
    Test POST /predict with valid input returns predictions and labels.
    """
    response = client.post("/predict", json={"input": sample_input})
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert "labels" in result
    assert len(result["predictions"]) == len(sample_input)


def test_predict_endpoint_invalid():
    """
    Test POST /predict with invalid input returns HTTP 422 error.
    """
    response = client.post("/predict", json={"input": "invalid input"})
    assert response.status_code == 422

    json_response = response.json()
    # Validation errors include a 'detail' key with a list of errors
    assert "detail" in json_response
    assert isinstance(json_response["detail"], list)
