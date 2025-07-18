from fastapi.testclient import TestClient
from src.serve.main import app


# Unit test: valid input should return 200 and predictions
def test_predict_valid_input():
    import json

    with open("sample_input.json") as f:
        sample_input = json.load(f)
    with TestClient(app) as client:
        response = client.post("/predict", json={"input": sample_input})
        assert response.status_code == 200
        assert "predictions" in response.json()
        assert "labels" in response.json()


# Unit test: invalid input should return 422
def test_predict_invalid_input():
    with TestClient(app) as client:
        response = client.post("/predict", json={"not": "valid"})
        assert response.status_code == 422


# Integration test: real sample input should work
def test_predict_sample_input_file():
    import json

    with open("sample_input.json") as f:
        input_data = json.load(f)
    with TestClient(app) as client:
        response = client.post("/predict", json={"input": input_data})
        assert response.status_code == 200
        assert "predictions" in response.json()
        assert "labels" in response.json()
