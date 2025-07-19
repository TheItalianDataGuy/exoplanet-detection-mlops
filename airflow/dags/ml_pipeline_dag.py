from airflow.decorators import dag, task
from datetime import datetime
import os
import sys
import subprocess
import logging
from typing import cast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "models/train_baseline.py")
REGISTER_SCRIPT = os.path.join(PROJECT_ROOT, "models/register_best_model.py")
EVALUATION_METRICS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "metrics.txt")


@dag(
    dag_id="ml_pipeline_dag",
    description="Train, evaluate, and register ML model for exoplanet classification",
    start_date=datetime(2025, 7, 19),
    schedule=None,
    catchup=False,
    tags=["mlops", "mlflow", "exoplanet"],
)
def ml_pipeline():

    @task()
    def load_data() -> str:
        """Mocked data loading step."""
        logger.info("Loading data (mocked).")
        # In production: load from S3, GCS, or local and return actual path
        return "/tmp/data.csv"

    @task()
    def train_model(data_path: str) -> str:
        """Train ML model and return model path."""
        logger.info(f"📦 Training model using data at {data_path}")
        try:
            subprocess.run(["python", TRAIN_SCRIPT], check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Training script failed.")
            raise e
        return "/tmp/model.pkl"

    @task()
    def evaluate_model(model_path: str) -> bool:
        """Evaluate the model based on metrics from a file."""
        logger.info(f"Evaluating model at {model_path}")
        try:
            with open(EVALUATION_METRICS_PATH, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "accuracy" in line.lower():
                        accuracy = float(line.strip().split("=")[-1])
                        logger.info(f"Accuracy: {accuracy}")
                        return accuracy >= 0.9
        except Exception as e:
            logger.error("Failed to evaluate model.")
            raise e
        return False

    @task()
    def register_model(evaluation_passed: bool) -> None:
        """Register the model if evaluation passed."""
        if not evaluation_passed:
            raise ValueError("Evaluation failed. Aborting registration.")
        logger.info("Registering model...")
        try:
            subprocess.run(["python", REGISTER_SCRIPT], check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Registration script failed.")
            raise e

    # DAG task dependencies
    data_path = load_data()
    model_path = train_model(cast(str, data_path))
    eval_result = evaluate_model(cast(str, model_path))
    register_model(cast(bool, eval_result))


ml_pipeline()
