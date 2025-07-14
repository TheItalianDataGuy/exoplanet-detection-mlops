import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models.signature import infer_signature
import joblib

# Load the dataset
df = pd.read_csv('data/kepler_exoplanet_data.csv')

# Drop irrelevant ID and text columns
df = df.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition'])

# Map target labels to integers
df['koi_disposition'] = df['koi_disposition'].map({
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1,
    'CONFIRMED': 2
})

# Drop rows with missing target
df = df.dropna(subset=['koi_disposition'])

# Remove columns with too many missing values
mostly_complete = df.columns[df.isnull().sum() < 500]
df = df[mostly_complete]

# Drop rows with any remaining missing values
df = df.dropna()

# Drop all non-numeric columns
non_numeric_cols = df.select_dtypes(exclude='number').columns
df = df.drop(columns=non_numeric_cols)

# Separate features and target
X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']

# Debug output
print(f"Shape of features (X): {X.shape}")
print(f"Number of target labels (y): {len(y)}")
print("Target value counts:\n", y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set up MLflow experiment
mlflow.set_experiment("kepler_exoplanet_baseline")

# Track experiment
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Inferred model signature (after training)
    signature = infer_signature(X_train, clf.predict(X_train))
    input_example = X_train.iloc[:2]

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        clf,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print("\nModel trained and tracked with MLflow")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["False Positive", "Candidate", "Confirmed"]))

# Save model locally for deployment
joblib.dump(clf, "models/random_forest.joblib")
