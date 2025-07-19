import pytest
from src.serve.main import get_model_uri


@pytest.mark.parametrize(
    "env_value, expected_uri",
    [
        ("local", "models/random_forest.joblib"),  # Local environment uses local joblib
        (
            "prod",
            "models:/RandomForestExoplanet@production",
        ),  # Prod uses MLflow registry path
        (None, "models/random_forest.joblib"),  # Default fallback to local joblib
    ],
)
def test_get_model_uri(env_value, expected_uri):
    """
    Test that get_model_uri returns the correct path depending on environment.
    This is a pure function test — no environment variable or module reload needed.
    """
    # When env_value is None, simulate default 'local' environment
    env = env_value if env_value is not None else "local"
    uri = get_model_uri(env)
    assert uri == expected_uri
