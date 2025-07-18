import os
import json
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set experiment name in MLflow
EXPERIMENT_NAME = "exoplanet_baseline"
mlflow.set_experiment(EXPERIMENT_NAME)

# Ensure local directory exists for model artifacts
os.makedirs("models", exist_ok=True)

# Load and clean dataset
df = pd.read_csv("data/kepler_exoplanet_data.csv")

# Drop irrelevant or redundant columns
df = df.drop(
    columns=["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition"],
    errors="ignore",
)

# Drop error columns (e.g., *_err1, *_err2)
error_cols = [col for col in df.columns if "_err1" in col or "_err2" in col]
df = df.drop(columns=error_cols)

# Encode target variable numerically
df["koi_disposition"] = df["koi_disposition"].map(
    {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
)

# Keep only numeric features, cast to float64
df = df.select_dtypes(include="number").astype("float64")

# Drop target leakage column if present
df = df.drop(columns=["koi_score"], errors="ignore")

# Split features and target
X = df.drop(columns=["koi_disposition"])
y = df["koi_disposition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Define model hyperparameters
params = {"n_estimators": 150, "max_depth": 15, "random_state": 42}

# Train the Random Forest classifier
clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)

# Evaluate model performance
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Start MLflow tracking
with mlflow.start_run(run_name="baseline_rf_150trees") as run:
    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

    # Save model locally
    joblib.dump(clf, "models/random_forest.joblib")

    # Log model with input example and signature
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=X_train.sample(1, random_state=42),
        signature=infer_signature(X_train, clf.predict(X_train)),
    )

    # Save sample input for FastAPI documentation
    sample_input_path = "sample_input.json"
    X_train.sample(1, random_state=42).to_json(
        sample_input_path, orient="records", lines=False
    )
    mlflow.log_artifact(sample_input_path)

    # Save expected columns for input validation in FastAPI
    expected_cols_path = "models/expected_columns.json"
    with open(expected_cols_path, "w") as f:
        json.dump(X_train.columns.tolist(), f)
    mlflow.log_artifact(expected_cols_path)

    # Add tags and register model
    mlflow.set_tags({"stage": "baseline", "type": "RandomForest"})
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="RandomForestExoplanet")

# Log completion
logging.info("Model trained, evaluated, logged, and registered.")

# Trigger model selection pipeline (optional automation)
os.system("python src/models/register_best_model.py")
