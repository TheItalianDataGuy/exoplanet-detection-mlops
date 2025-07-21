from pathlib import Path
from airflow.decorators import task, dag
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from datetime import timedelta, datetime
import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from src.config.settings import settings
from src.models.train_baseline import (
    load_data,
    train_model,
    evaluate_model,
    save_metrics,
    save_artifacts,
    register_model_with_alias,
)

email_address = Variable.get("email_address", default_var="default@example.com")


def notify_failure(context):
    task_id = context.get("task_instance").task_id
    dag_id = context.get("dag").dag_id
    execution_date = context.get("execution_date")
    log_url = context.get("task_instance").log_url

    email_task = EmailOperator(
        task_id="email_on_failure",
        to=email_address,
        subject=f"Airflow Alert: Task {task_id} Failed in DAG {dag_id}",
        html_content=f"""
            <h3>Task Failure Notification</h3>
            <p><strong>Task:</strong> {task_id}</p>
            <p><strong>DAG:</strong> {dag_id}</p>
            <p><strong>Execution Time:</strong> {execution_date}</p>
            <p><a href="{log_url}">View Logs</a></p>
        """,
    )
    email_task.execute(context)


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
        logging.info("Loading and preprocessing data")
        return load_data(settings.data_path)

    @task(multiple_outputs=True)
    def split_data(df: pd.DataFrame):
        logging.info("Splitting data into train and test sets")
        X = df.drop(columns=["koi_disposition"])
        y = df["koi_disposition"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )

        output_dir = Path("/opt/airflow/data/splits")
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train_path = output_dir / "X_train.pkl"
        X_test_path = output_dir / "X_test.pkl"
        y_train_path = output_dir / "y_train.pkl"
        y_test_path = output_dir / "y_test.pkl"

        X_train.to_pickle(X_train_path)
        X_test.to_pickle(X_test_path)
        y_train.to_pickle(y_train_path)
        y_test.to_pickle(y_test_path)

        return {
            "X_train_path": str(X_train_path),
            "X_test_path": str(X_test_path),
            "y_train_path": str(y_train_path),
            "y_test_path": str(y_test_path),
        }

    @task()
    def train_and_save_model(X_train_path: str, y_train_path: str) -> str:
        logging.info("Training model and saving artifact")
        X_train = pd.read_pickle(X_train_path)
        y_train = pd.read_pickle(y_train_path)

        clf, model_path = train_model(
            X_train,
            y_train,
            params={"n_estimators": 150, "max_depth": 15, "random_state": 42},
        )
        return str(model_path)

    @task()
    def evaluate_and_save_metrics(
        model_path: Path, X_test_path: str, y_test_path: str
    ) -> bool:
        logging.info("Evaluating model")
        X_test = pd.read_pickle(X_test_path)
        y_test = pd.read_pickle(y_test_path)

        model = joblib.load(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, settings.metrics_path)

        return metrics["accuracy"] >= settings.success_threshold

    @task()
    def save_artifacts_to_disk(X_train_path: str, model_path: Path):
        logging.info("Saving artifacts to disk")
        X_train = pd.read_pickle(X_train_path)
        save_artifacts(X_train, model_path, settings.expected_cols_path)

    @task()
    def register_model(evaluation_passed: bool, model_path: Path):
        if not evaluation_passed:
            logging.warning("Evaluation failed, skipping model registration")
            raise ValueError("Model evaluation did not meet threshold")
        logging.info("Registering model with MLflow alias")
        register_model_with_alias(f"file://{model_path}", "RandomForestExoplanet")

    df = load_and_preprocess_data()
    split_paths = split_data(df)  # type: ignore

    X_train_path = split_paths["X_train_path"]  # type: ignore
    X_test_path = split_paths["X_test_path"]  # type: ignore
    y_train_path = split_paths["y_train_path"]  # type: ignore
    y_test_path = split_paths["y_test_path"]  # type: ignore

    model_path = train_and_save_model(X_train_path, y_train_path)
    evaluation_passed = evaluate_and_save_metrics(model_path, X_test_path, y_test_path)  # type: ignore
    save_artifacts_to_disk(X_train_path, model_path)  # type: ignore
    register_model(evaluation_passed, model_path)  # type: ignore


ml_pipeline()
