import json
from airflow.decorators import task, dag
from datetime import timedelta, datetime
import subprocess
import logging

from sklearn.model_selection import train_test_split
from config.settings import settings
from airflow.providers.smtp.operators.email import EmailOperator  # type: ignore
from airflow.models import Variable
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from settings
TRAIN_SCRIPT = settings.train_script_path
REGISTER_SCRIPT = settings.register_script_path
EVALUATION_METRICS_PATH = settings.metrics_path
SUCCESS_THRESHOLD = settings.success_threshold


# Helper function to check if a value exists in XCom
def fetch_xcom_value(task_instance, key: str):
    """Fetch XCom value from previous task."""
    return task_instance.xcom_pull(task_ids=key)


# Notify in case of failure (Using email)
email_address = Variable.get("email_address")


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
    def load_data() -> pd.DataFrame:
        """Load data from source."""
        logger.info("Loading data from source.")
        data_path = settings.data_path
        df = pd.read_csv(data_path)

        # Preprocessing steps
        drop_cols = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition"]
        df.drop(columns=drop_cols, errors="ignore", inplace=True)

        error_cols = [col for col in df.columns if "_err1" in col or "_err2" in col]
        df.drop(columns=error_cols, inplace=True)

        # Encode the target labels
        disposition_map = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
        df["koi_disposition"] = df["koi_disposition"].map(disposition_map)

        df = df.select_dtypes(include="number").astype("float64")
        df.drop(columns=["koi_score"], errors="ignore", inplace=True)

        return df

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
    def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> str:
        """Train the model using the provided data."""
        logger.info("Training model using data.")
        try:
            # Run the training script
            subprocess.run(["python", TRAIN_SCRIPT], check=True)

            # Save the model path dynamically based on settings
            model_save_dir = Path(settings.model_save_dir)
            model_save_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists
            model_save_path = model_save_dir / "random_forest.joblib"
            logger.info(f"Model saved to: {model_save_path}")

            return str(model_save_path)  # Return the full path of the saved model
        except subprocess.CalledProcessError as e:
            logger.error("Training failed.", exc_info=True)
            raise e

    @task()
    def evaluate_model(
        model_path: str, X_test: pd.DataFrame, y_test: pd.Series
    ) -> bool:
        """Evaluate the model's performance."""
        logger.info(f"Evaluating model at {model_path}.")
        try:
            model = joblib.load(model_path)  # Load the trained model
            logger.info(f"Model loaded successfully from {model_path}")

            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)

            # Log the evaluation metrics
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"F1 Score: {f1}")
            logger.info(f"Confusion Matrix: \n{conf_matrix}")
            logger.info(f"Classification Report: \n{class_report}")

            # Save metrics to a JSON file
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
            }
            save_metrics(metrics, settings.metrics_path)

            if accuracy >= SUCCESS_THRESHOLD:
                logger.info("Model evaluation passed.")
                return True
            else:
                logger.warning("Model evaluation failed: Accuracy below threshold.")
                return False

        except Exception as e:
            logger.error("Failed to evaluate model.", exc_info=True)
            raise e

    @task()
    def register_model(evaluation_passed: bool) -> None:
        """Register the model if evaluation passes."""
        if not evaluation_passed:
            logger.error("Evaluation failed, not registering model.")
            raise ValueError("Model evaluation failed.")

        logger.info("Registering the model.")
        try:
            subprocess.run(["python", REGISTER_SCRIPT], check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Registration failed.", exc_info=True)
            raise e

    @task()
    def save_metrics(metrics: dict, metrics_path: str):
        """Save the model evaluation metrics to a file."""
        try:
            Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            logger.info(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)
            raise

    # Define task dependencies
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)  # type: ignore
    model_path = train_model(X_train, y_train)
    eval_result = evaluate_model(model_path, X_test, y_test)  # type: ignore
    register_model(eval_result)  # type: ignore


# Instantiate the DAG
ml_pipeline()
