from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings for the Exoplanet Detection project."""

    # Runtime environment: 'local' or 'prod'
    env: str = "local"

    # Paths to scripts
    train_script_path: str = "models/train_baseline.py"
    register_script_path: str = "models/register_best_model.py"

    # Path to sample input file
    sample_path: str = "sample_input.json"

    # MLflow tracking configuration
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "exoplanet-detection"

    # Configuration variables
    model_name: str = "RandomForestExoplanet"
    primary_metric: str = "f1_score"

    # Python path for module imports
    pythonpath: str = "src"

    # Load from .env automatically
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # Data paths
    data_path: str = "data/kepler_exoplanet_data.csv"

    # Metrics paths
    metrics_path: str = "models/metrics.json"

    # Model save directory (added type annotation)
    model_save_dir: Path = Path("models")

    # Success threshold for model evaluation
    success_threshold: float = 0.85


# Singleton instance to import across the app
settings = Settings()
