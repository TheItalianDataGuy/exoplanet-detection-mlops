from pathlib import Path
from airflow.decorators import task, dag
from datetime import timedelta, datetime
import logging
from sklearn.model_selection import train_test_split
from src.config.settings import settings
import joblib
import pandas as pd
from src.models.train_baseline import (
    load_data,
    train_model,
    evaluate_model,
    save_metrics,
    save_artifacts,
    register_model_with_alias,
)
from airflow.operators.email import EmailOperator
from airflow.models import Variable


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from settings (now correctly as Path objects)
EVALUATION_METRICS_PATH = settings.metrics_path
SUCCESS_THRESHOLD = settings.success_threshold


# Helper function to check if a value exists in XCom
def fetch_xcom_value(task_instance, key: str):
    """Fetch XCom value from previous task."""
    return task_instance.xcom_pull(task_ids=key)


# Notify in case of failure (Using email)
email_address = Variable.get("email_address", default_var="default@example.com")


def notify_failure(task_id: str):
    return EmailOperator(
        task_id=f"send_email_on_{task_id}_failure",
        to=email_address,
        subject=f"Task {task_id} Failed",
        html_content=f"Task {task_id} failed. Please check the logs for more information.",
    )


# Default arguments to be used by each task
default_args = {
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": notify_failure,
}


@dag(
    dag_id="ml_pipeline_dag",
    description="Train, evaluate, and register ML model for exoplanet classification",
    schedule_interval="@daily",
    start_date=datetime(2025, 7, 19),
    catchup=False,
    tags=["mlops", "mlflow", "exoplanet"],
    default_args=default_args,
)
def ml_pipeline():

    @task()
    def load_and_preprocess_data() -> pd.DataFrame:
        """Load and preprocess the data."""
        logger.info("Loading data and preprocessing.")
        data_path = settings.data_path
        return load_data(data_path)

    @task()
    def split_data(df: pd.DataFrame) -> tuple:
        """Split the data into training and test sets."""
        X = df.drop(columns=["koi_disposition"])
        y = df["koi_disposition"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @task()
    def train_and_save_model(X_train: pd.DataFrame, y_train: pd.Series) -> Path:
        """Train the model and save it."""
        logger.info("Training and saving model.")
        rf_params = {"n_estimators": 150, "max_depth": 15, "random_state": 42}
        clf, model_save_path = train_model(X_train, y_train, rf_params)
        return model_save_path  # Now returning Path instead of str

    @task()
    def evaluate_and_save_metrics(
        model_path: Path, X_test: pd.DataFrame, y_test: pd.Series
    ) -> bool:
        """Evaluate the model and save metrics."""
        logger.info("Evaluating model.")
        # Load the model from the saved path
        model = joblib.load(model_path)

        # Call evaluate_model with the model object
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, settings.metrics_path)

        if metrics["accuracy"] >= SUCCESS_THRESHOLD:
            logger.info("Model evaluation passed.")
            return True
        else:
            logger.warning("Model evaluation failed: Accuracy below threshold.")
            return False

    @task()
    def save_artifacts_to_disk(X_train: pd.DataFrame, model_path: Path) -> None:
        """Save artifacts like expected columns and sample input."""
        # Use Path for expected_cols_path from settings.py
        expected_cols_path = settings.expected_cols_path

        # Call save_artifacts from train_baseline.py with the correct arguments
        save_artifacts(
            X_train, model_path, expected_cols_path
        )  # Now passing Path objects

    @task()
    def register_model(evaluation_passed: bool, model_path: Path) -> None:
        """Register the model if evaluation passes."""
        if not evaluation_passed:
            logger.error("Evaluation failed, not registering model.")
            raise ValueError("Model evaluation failed.")

        logger.info("Registering the model.")
        model_uri = f"file://{model_path}"  # Change as necessary
        register_model_with_alias(model_uri, "RandomForestExoplanet")

    # Define task dependencies
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(df)  # type: ignore
    model_path = train_and_save_model(X_train, y_train)  # type: ignore
    eval_result = evaluate_and_save_metrics(model_path, X_test, y_test)  # type: ignore
    save_artifacts_to_disk(X_train, model_path)  # type: ignore
    register_model(eval_result, model_path)  # type: ignore


# Instantiate the DAG
ml_pipeline()
