import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models.signature import infer_signature
import joblib

DATA_PATH = "data/kepler_exoplanet_data.csv"
EXPERIMENT_NAME = "kepler_exoplanet_baseline"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load and clean data
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition'], errors='ignore')
df = df.loc[:, df.isnull().sum() < 500].dropna()
df['koi_disposition'] = df['koi_disposition'].map({'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2})
df = df.select_dtypes(include='number').astype('float64')

X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(clf, artifact_path="model", signature=signature, input_example=X_train.sample(5))

    print(f"Logged model with accuracy {acc:.4f}")
    print(classification_report(y_test, clf.predict(X_test), target_names=["False Positive", "Candidate", "Confirmed"]))

# Save model locally
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/random_forest.joblib")
X_train.sample(1, random_state=42).to_json("sample_input.json", orient="records", lines=False)

sample = X.sample(1, random_state=42)
sample.to_json("sample_input.json", orient="records", lines=False)