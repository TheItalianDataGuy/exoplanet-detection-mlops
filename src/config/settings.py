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
    tracking_uri: str = "http://localhost:5001"
    experiment_name: str = "exoplanet-detection"

    # Load from .env automatically
    model_config = SettingsConfigDict(env_file=".env")


# Singleton instance to import across the app
settings = Settings()
