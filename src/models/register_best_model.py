import os
import logging
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load values from .env
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "exoplanet_baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "RandomForestExoplanet")
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "f1_score")

# MLflow setup
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    logging.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
    exit(1)

# Find best run
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
)

if not runs:
    logging.warning("No runs found.")
    exit(0)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_metric_value = best_run.data.metrics.get(PRIMARY_METRIC)
logging.info(f"Best run ID: {best_run_id} with {PRIMARY_METRIC} = {best_metric_value}")

# Check current production version
try:
    prod_version = client.get_model_version_by_alias(
        name=MODEL_NAME, alias="production"
    )
    if prod_version.run_id == best_run_id:
        logging.info("Best model already in production.")
        exit(0)
    else:
        logging.info(f"Current production run ID: {prod_version.run_id}")
except Exception:
    logging.info("No production alias yet. Proceeding with registration.")

# Register the new version
model_uri = f"runs:/{best_run_id}/model"

try:
    model_version = client.create_model_version(
        name=MODEL_NAME, source=model_uri, run_id=best_run_id
    )
    logging.info(f"Model version {model_version.version} registered.")

    client.set_registered_model_alias(
        name=MODEL_NAME, alias="production", version=model_version.version
    )
    logging.info(f"Alias 'production' set for version {model_version.version}.")

except Exception as e:
    logging.error(f"Model registration failed: {str(e)}")
    exit(1)
