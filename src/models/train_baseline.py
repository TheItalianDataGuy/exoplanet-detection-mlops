import os
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set MLflow experiment
EXPERIMENT_NAME = "exoplanet_baseline"
mlflow.set_experiment(EXPERIMENT_NAME)

# Ensure model output directory exists
os.makedirs("models", exist_ok=True)

# Load and clean dataset
df = pd.read_csv("data/kepler_exoplanet_data.csv")
df = df.drop(
    columns=["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition"],
    errors="ignore",
)
error_cols = [col for col in df.columns if "_err1" in col or "_err2" in col]
df = df.drop(columns=error_cols)

# Map target variable to numeric values
df["koi_disposition"] = df["koi_disposition"].map(
    {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
)

# Ensure all numeric columns are of type float64
df = df.select_dtypes(include="number").astype("float64")

# Drop 'koi_score' if it exists, as it's not needed for training
df = df.drop(columns=["koi_score"], errors="ignore")


# Split features and target
X = df.drop(columns=["koi_disposition"])
y = df["koi_disposition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Define model parameters
params = {"n_estimators": 150, "max_depth": 15, "random_state": 42}

# Train model
clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Log to MLflow
with mlflow.start_run(run_name="baseline_rf_150trees") as run:
    # Log hyperparameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

    # Save model locally
    model_path = "models/random_forest.joblib"
    joblib.dump(clf, model_path)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=X_train.sample(1, random_state=42),
        signature=infer_signature(X_train, clf.predict(X_train)),
    )

    # Save a sample input for use in FastAPI
    sample_input_path = "sample_input.json"
    X_train.sample(1, random_state=42).to_json(
        sample_input_path, orient="records", lines=False
    )
    mlflow.log_artifact(sample_input_path)

    # Save and log expected column names
    import json

    expected_cols_path = "models/expected_columns.json"
    with open(expected_cols_path, "w") as f:
        json.dump(X_train.columns.tolist(), f)
    mlflow.log_artifact(expected_cols_path)

    # Add custom tags to the run
    mlflow.set_tags({"stage": "baseline", "type": "RandomForest"})

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="RandomForestExoplanet")

# Confirm success
logging.info("Model trained, evaluated, logged, and registered.")

# Trigger the model selection script
os.system("python src/models/register_best_model.py")
