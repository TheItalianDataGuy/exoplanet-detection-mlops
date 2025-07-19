from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for the Exoplanet Detection project.
    This class uses Pydantic to manage configuration settings, allowing for easy
    loading from environment variables or a .env file.
    """

    # Runtime environment: 'local' or 'prod'
    env: str = "local"

    # Path to local model file (used when env = 'local')
    model_path: str = "models/random_forest.joblib"

    # Path to sample input file
    sample_path: str = "sample_input.json"

    # MLflow tracking configuration
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "exoplanet-detection"

    # Load from .env automatically
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # Configuration variables
    model_name: str = "RandomForestExoplanet"
    primary_metric: str = "f1_score"

    # Python path for module imports
    pythonpath: str = "src"


# Singleton instance to import across the app
settings = Settings()
