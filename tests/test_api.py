import pytest
from fastapi.testclient import TestClient
from src.serve.main import app

client = TestClient(app)

# Unit test: Check that /predict returns 200 OK with valid input
def test_predict_valid_input():
    sample_input = [{
        "koi_period": 10.5,
        "koi_time0bk": 135.6,
        "koi_impact": 0.1,
        "koi_duration": 5.5,
        "koi_depth": 1500.0,
        "koi_prad": 1.2,
        "koi_teq": 500,
        "koi_insol": 100,
        "koi_model_snr": 12.0,
        "koi_slogg": 4.4,
        "koi_srad": 0.9,
        "ra": 296.5,
        "dec": 43.9
    }]

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

# Unit test: Check that invalid input returns 422 Unprocessable Entity
def test_predict_invalid_input():
    response = client.post("/predict", json={"not": "valid"})
    assert response.status_code == 422

# Integration test: Load real sample input file and hit the endpoint
def test_predict_sample_input_file():
    import json
    with open("sample_input.json") as f:
        input_data = json.load(f)

    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)
