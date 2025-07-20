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

# Constants
EVALUATION_METRICS_PATH = settings.metrics_path
SUCCESS_THRESHOLD = settings.success_threshold
email_address = Variable.get("email_address", default_var="default@example.com")


# Notify on failure
def notify_failure(task_id: str):
    return EmailOperator(
        task_id=f"send_email_on_{task_id}_failure",
        to=email_address,
        subject=f"Task {task_id} Failed",
        html_content=f"Task {task_id} failed. Please check the logs for more information.",
    )


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
        logger.info("Loading data and preprocessing.")
        return load_data(settings.data_path)

    @task()
    def split_data(df: pd.DataFrame):
        X = df.drop(columns=["koi_disposition"])
        y = df["koi_disposition"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    @task()
    def train_and_save_model(X_train: pd.DataFrame, y_train: pd.Series) -> Path:
        logger.info("Training and saving model.")
        clf, model_save_path = train_model(
            X_train, y_train, {"n_estimators": 150, "max_depth": 15, "random_state": 42}
        )
        return model_save_path

    @task()
    def evaluate_and_save_metrics(
        model_path: Path, X_test: pd.DataFrame, y_test: pd.Series
    ) -> bool:
        logger.info("Evaluating model.")
        model = joblib.load(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, settings.metrics_path)
        return metrics["accuracy"] >= SUCCESS_THRESHOLD

    @task()
    def save_artifacts_to_disk(X_train: pd.DataFrame, model_path: Path):
        save_artifacts(X_train, model_path, settings.expected_cols_path)

    @task()
    def register_model(evaluation_passed: bool, model_path: Path):
        if not evaluation_passed:
            logger.warning("Evaluation failed, not registering model.")
            raise ValueError("Model evaluation failed.")
        register_model_with_alias(f"file://{model_path}", "RandomForestExoplanet")

    # DAG structure
    df = load_and_preprocess_data()
    splits = split_data(df)  # type: ignore

    X_train = splits["X_train"]  # type: ignore
    X_test = splits["X_test"]  # type: ignore
    y_train = splits["y_train"]  # type: ignore
    y_test = splits["y_test"]  # type: ignore

    model_path = train_and_save_model(X_train, y_train)
    eval_result = evaluate_and_save_metrics(model_path, X_test, y_test)  # type: ignore
    save_artifacts_to_disk(X_train, model_path)  # type: ignore
    register_model(eval_result, model_path)  # type: ignore


ml_pipeline()
