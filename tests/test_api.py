from fastapi.testclient import TestClient
from src.serve.main import app


client = TestClient(app)


# Unit test: Check that /predict returns 200 OK with valid input
def test_predict_valid_input():
    import json

    with open("sample_input.json") as f:
        sample_input = json.load(f)

    response = client.post("/predict", json={"input": sample_input})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "labels" in response.json()


# Unit test: Check that invalid input returns 422 Unprocessable Entity
def test_predict_invalid_input():
    response = client.post("/predict", json={"not": "valid"})
    assert response.status_code == 422


# Integration test: Load real sample input file and hit the endpoint
def test_predict_sample_input_file():
    import json

    with open("sample_input.json") as f:
        input_data = json.load(f)

    response = client.post("/predict", json={"input": input_data})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "labels" in response.json()
