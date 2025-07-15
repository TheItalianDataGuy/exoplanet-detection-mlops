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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow experiment name
mlflow.set_experiment("exoplanet_baseline")

# Create necessary directories
os.makedirs("models", exist_ok=True)

# Load and clean dataset
df = pd.read_csv("data/kepler_exoplanet_data.csv")
df = df.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition'], errors='ignore')
df = df.loc[:, df.isnull().sum() < 500].dropna()
df['koi_disposition'] = df['koi_disposition'].map({
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1,
    'CONFIRMED': 2
})
df = df.select_dtypes(include='number').astype('float64')

# Split data into features and target
X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Define model parameters
param_grid = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 15},
    {"n_estimators": 200, "max_depth": None},
]

for params in param_grid:
    with mlflow.start_run():
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

# Train Random Forest model
clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)

# Generate predictions
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Start MLflow run
with mlflow.start_run(run_name="baseline_rf_100trees") as run:
    # Log hyperparameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

    # Save and log model
    joblib.dump(clf, "models/random_forest.joblib")
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=X_train.sample(1, random_state=42),
        signature=infer_signature(X_train, clf.predict(X_train))
    )

    # Log input sample
    X_train.sample(1, random_state=42).to_json("sample_input.json", orient="records", lines=False)
    mlflow.log_artifact("sample_input.json")

    # Add tags
    mlflow.set_tags({
        "stage": "baseline",
        "type": "RandomForest"
    })

    # Register model
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="RandomForestExoplanet")

    # Call external script to register best model
    os.system("python src/models/register_best_model.py")

logger.info("Model training, logging, and registration completed.")
