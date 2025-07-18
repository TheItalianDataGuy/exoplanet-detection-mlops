import importlib
import pytest


# Test different ENV values to check if the correct model path is selected
@pytest.mark.parametrize(
    "env_value, expected_model_uri",
    [
        ("local", "models/random_forest.joblib"),  # Local: load model from joblib
        (
            "prod",
            "models:/RandomForestExoplanet@production",
        ),  # Prod: load model from MLflow registry
        (None, "models/random_forest.joblib"),  # No ENV set: fallback to local joblib
    ],
)
def test_model_path_selection(monkeypatch, env_value, expected_model_uri):
    """
    Check that the model path selected matches the expected value
    based on the ENV environment variable.
    """

    # Set or remove the ENV variable
    if env_value is None:
        monkeypatch.delenv("ENV", raising=False)  # Remove ENV if None
    else:
        monkeypatch.setenv("ENV", env_value)  # Set ENV to the test value

    # Reload the main module to apply the new ENV
    import src.serve.main as main

    importlib.reload(main)

    # Check if MODEL_URI matches what we expect for that ENV
    assert main.MODEL_URI == expected_model_uri
