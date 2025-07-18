# 🌌 Exoplanet Detection with MLOps

This project applies **end-to-end MLOps best practices** to build, deploy, and monitor a machine learning model for detecting exoplanets using NASA's Kepler data.

---

## 🚀 Objective

Classify whether a signal observed by the Kepler telescope corresponds to an **exoplanet** or a **false positive**, using features derived from light curves.

---

## 📊 Dataset

- **Source**: [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
- Cleaned and preprocessed for supervised learning.

---

## ✅ Features

- 📦 Model training with **RandomForestClassifier**
- 📈 Experiment tracking via **MLflow**
- 🧪 Unit & integration testing with **pytest**
- 📁 Model versioning via **MLflow Model Registry**
- 🔁 Batch inference support
- 🐳 Docker-based deployment
- 📡 REST API with **FastAPI**
- 🧹 Code quality with **black**, **ruff**, and **flake8**
- ✅ Pre-commit hooks for consistency
- 🧪 Monitoring with **Evidently**
- 🧱 Workflow orchestration with **Airflow** or **Prefect**
- ⚙️ Reproducibility via `Makefile`, `requirements.txt`, `.env.example`

---

## 📁 Project Structure

```
├── data/                          # Input dataset
├── models/                        # Trained model artifacts
├── notebooks/                     # EDA and exploration
├── src/
│   ├── features/                  # Feature engineering
│   ├── models/                   # Training scripts
│   ├── pipelines/                # (Optional) pipelines
│   ├── predict/                  # Predictor logic
│   └── serve/                    # FastAPI app
├── tests/                         # Unit and integration tests
├── .env.example                   # Template for environment variables
├── Makefile                       # Make commands for reproducibility
└── README.md                      # Project overview
```

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/exoplanet-detection-mlops.git
cd exoplanet-detection-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

---


## 🧪 Makefile Commands

You can use the `Makefile` to simplify common tasks:

| Command              | Description                            |
|----------------------|----------------------------------------|
| `make format`        | Format code with Black and Ruff        |
| `make lint`          | Run linting checks                     |
| `make test`          | Run unit and integration tests         |
| `make run`           | Launch the FastAPI server locally      |
| `make train`         | Train the baseline RandomForest model  |
| `make register-best` | Register the best model to MLflow      |

## 🌐 API Usage

Once the app is running locally:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check:
  ```bash
  curl http://localhost:8000/
  ```
- Prediction:
  ```bash
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d @sample_input.json
  ```

---

## 🧪 Example API Request

POST `/predict`

```json
{
  "input": [
    {
      "koi_fpflag_nt": 0,
      "koi_fpflag_ss": 0,
      "koi_fpflag_co": 0,
      "koi_fpflag_ec": 0,
      "koi_period": 41.079,
      "koi_time0bk": 134.5,
      "koi_impact": 0.11
    }
  ]
}
```

---

## 📦 Coming Soon

- Dockerized deployment
- CI/CD pipeline with GitHub Actions
- Live monitoring with Evidently
- Airflow orchestration for batch jobs

---

## 🧑‍💻 Author

Andrea — MSc Data Science Student & Aspiring MLOps Engineer  
Feel free to connect on [LinkedIn](https://www.https://www.linkedin.com/in/andrea-marella/) or check out my [GitHub](https://github.com/TheItalianDataGuy).

---

## 🤝 License

This project is licensed under the MIT License.