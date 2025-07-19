import os
import json
import logging
from pathlib import Path

import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient


def configure_logging() -> None:
    """Configure global logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the Kepler exoplanet dataset.

    Args:
        data_path: Path to the CSV dataset.

    Returns:
        Preprocessed pandas DataFrame.
    """
    df = pd.read_csv(data_path)

    # Drop irrelevant columns that do not contribute to the model
    drop_cols = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Remove error-related columns
    error_cols = [col for col in df.columns if "_err1" in col or "_err2" in col]
    df.drop(columns=error_cols, inplace=True)

    # Encode target labels numerically
    disposition_map = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
    df["koi_disposition"] = df["koi_disposition"].map(disposition_map)

    # Retain only numeric columns for model input
    df = df.select_dtypes(include="number").astype("float64")

    # Remove potential data leakage columns
    df.drop(columns=["koi_score"], errors="ignore", inplace=True)

    return df


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier with given parameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        params: Hyperparameters for RandomForestClassifier.

    Returns:
        Trained RandomForestClassifier instance.
    """
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Evaluate the model on test data and return relevant metrics.

    Args:
        clf: Trained classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with accuracy and weighted F1-score.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return {"accuracy": accuracy, "f1_score": f1}


def save_artifacts(
    X_train: pd.DataFrame, model_path: Path, expected_cols_path: Path
) -> None:
    """
    Save expected columns and sample input for downstream validation and FastAPI documentation.

    Args:
        X_train: Training features DataFrame.
        model_path: Path where the trained model will be saved.
        expected_cols_path: Path to save the expected input columns JSON.
    """
    # Save expected input columns to the models/ directory
    expected_cols_path.parent.mkdir(parents=True, exist_ok=True)
    with expected_cols_path.open("w") as f:
        json.dump(X_train.columns.tolist(), f)
    logging.info(f"Expected columns saved to: {expected_cols_path}")

    # Save sample input to the root directory for FastAPI documentation
    sample_input_path = Path("sample_input.json")
    X_train.sample(1, random_state=42).to_json(sample_input_path, orient="records")
    logging.info(f"Sample input saved to: {sample_input_path}")

    # Log both artifacts to MLflow
    mlflow.log_artifact(str(expected_cols_path))
    mlflow.log_artifact(str(sample_input_path))


def register_model_with_alias(model_uri: str, model_name: str):
    """
    Register a logged model to the MLflow Model Registry and assign an alias.

    Args:
        model_uri (str): URI of the logged model (e.g., "runs:/<run_id>/model").
        model_name (str): Name to register the model under.
    """
    client = MlflowClient()

    # Register the model (creates a new version under the given name)
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Optional: Set an alias (e.g., 'production')
    client.set_registered_model_alias(
        name=model_name, alias="production", version=result.version
    )


def main():
    configure_logging()
    load_dotenv()

    try:
        # Set MLflow tracking URI and experiment
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "exoplanet-detection")
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        logging.info(f"MLflow experiment set to: {experiment_name}")

        # Load and preprocess data
        data_path = "data/kepler_exoplanet_data.csv"
        df = load_data(data_path)
        logging.info(f"Data loaded and preprocessed with shape: {df.shape}")

        # Prepare features and labels
        X = df.drop(columns=["koi_disposition"])
        y = df["koi_disposition"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )
        logging.info(
            f"Train/test split completed: {X_train.shape[0]} train, {X_test.shape[0]} test"
        )

        # Train model
        rf_params = {"n_estimators": 150, "max_depth": 15, "random_state": 42}
        clf = train_model(X_train, y_train, rf_params)

        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)
        logging.info(f"Model evaluation metrics: {metrics}")

        # Save model locally
        model_save_path = Path("models/random_forest.joblib")
        joblib.dump(clf, model_save_path)
        logging.info(f"Trained model saved to: {model_save_path}")

        # Log to MLflow
        with mlflow.start_run(run_name="baseline_rf_150trees") as run:
            mlflow.log_params(rf_params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.sklearn.log_model(  # type: ignore
                sk_model=clf,
                artifact_path="model",
                input_example=X_train.sample(1, random_state=42),
                signature=infer_signature(X_train, clf.predict(X_train)),
            )

            save_artifacts(
                X_train, model_save_path, Path("models/expected_columns.json")
            )

            mlflow.set_tags({"stage": "baseline", "model_type": "RandomForest"})
            model_uri = f"runs:/{run.info.run_id}/model"
            register_model_with_alias(
                model_uri=model_uri, model_name="RandomForestExoplanet"
            )

        logging.info("Training, logging, and registration complete.")

    except Exception:
        logging.error("An error occurred during training or logging", exc_info=True)


if __name__ == "__main__":
    main()
