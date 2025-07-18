from fastapi.testclient import TestClient
from src.serve.main import app

# Initialize FastAPI test client
client = TestClient(app)


def test_predict_valid_input():
    """
    Unit test: Ensure that /predict returns 200 OK and valid keys for proper input.
    """
    import json

    with open("sample_input.json") as f:
        sample_input = json.load(f)

    response = client.post("/predict", json={"input": sample_input})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "labels" in response.json()


def test_predict_invalid_input():
    """
    Unit test: Ensure that invalid input returns a 422 Unprocessable Entity error.
    """
    response = client.post("/predict", json={"not": "valid"})
    assert response.status_code == 422


def test_predict_sample_input_file():
    """
    Integration test: Send real sample input to the /predict endpoint and verify response.
    """
    import json

    with open("sample_input.json") as f:
        input_data = json.load(f)

    response = client.post("/predict", json={"input": input_data})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "labels" in response.json()
