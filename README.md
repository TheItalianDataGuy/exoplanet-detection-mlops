# 🌌 Exoplanet Detection with MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://github.com/TheItalianDataGuy/exoplanet-detection-mlops/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-enabled-blue)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Airflow](https://img.shields.io/badge/Airflow-scheduled-9cf)

---

## 🚀 Objective

Classify whether a signal observed by the Kepler telescope corresponds to an **exoplanet** or a **false positive**, using features derived from light curves.

This project is a stepping stone toward contributing to the analysis of vast datasets produced by modern observatories like the **James Webb Space Telescope**, as part of my passion for astrophysics and scientific discovery.

---

## 📊 Dataset

- **Source**: [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
- Cleaned and preprocessed for supervised learning.

---

## ✅ Features

- 🧠 Model training with **RandomForestClassifier**
- 📈 Experiment tracking via **MLflow**
- 🔁 Model versioning and aliasing via **MLflow Model Registry**
- ⚙️ Batch pipeline orchestration with **Airflow**
- 🚀 REST API with **FastAPI**
- 🧪 Unit & integration testing with **pytest**
- 🐳 Full **Docker-based deployment**
- 🧹 Code quality with **black**, **ruff**, and **flake8**
- 🔒 Pre-commit hooks for consistency
- 📦 Reproducibility via `Makefile`, `requirements.txt`, `.env.example`
- 🧠 Future-proof: Architecture built for monitoring & CI/CD expansion

---

## 📁 Project Structure

```
├── airflow/                     # Airflow DAGs and variables
├── data/                        # Input dataset
├── models/                      # Trained model artifacts
├── notebooks/                   # EDA and exploration
├── src/
│   ├── config/                  # Environment and path settings
│   ├── features/                # Feature engineering (optional)
│   ├── models/                  # Training and evaluation scripts
│   ├── pipelines/              # Workflow or batch modules (optional)
│   ├── predict/                # Model prediction logic
│   └── serve/                  # FastAPI app
├── tests/                       # Unit and integration tests
├── .env.example                 # Template for environment variables
├── Makefile                     # Make commands for reproducibility
└── README.md                    # Project overview
```

---

## 🧪 Makefile Commands

You can use the `Makefile` to simplify common tasks:

| Command               | Description                                  |
|-----------------------|----------------------------------------------|
| `make format`         | Format code with Black and Ruff              |
| `make lint`           | Run linting checks                           |
| `make test`           | Run unit and integration tests               |
| `make train`          | Train the baseline RandomForest model        |
| `make register-best`  | Register the best model in MLflow Registry   |
| `make run`            | Run FastAPI app locally (dev mode)           |
| `make run-prod`       | Run FastAPI app in production mode           |
| `make mlflow-ui`      | Launch MLflow tracking UI                    |
| `make airflow-init`   | Initialize Airflow database                  |
| `make start-airflow`  | Start Airflow webserver and scheduler        |
| `make stop-airflow`   | Stop all Airflow services                    |
| `make import-airflow-vars` | Import Airflow variables from JSON     |
| `make check-ports`    | Check if ports 5001, 8000, 8080 are occupied |
| `make build`          | Build Docker containers                      |
| `make start`          | Start Docker containers + import vars        |
| `make stop`           | Stop Docker containers                       |
| `make logs`           | View logs for all Docker containers          |
| `make fastapi-logs`   | View logs for FastAPI container              |
| `make mlflow-logs`    | View logs for MLflow container               |
| `make airflow-logs`   | View logs for Airflow container              |
| `make restart`        | Restart all Docker containers                |
| `make clean`          | Remove all Docker containers and volumes     |
| `make train-docker`   | Train model from within FastAPI container    |
| `make check-docker`   | Ensure Docker daemon is running              |
| `make set-env-local`  | Set `.env` to local environment              |
| `make set-env-docker` | Set `.env` to docker environment             |
| `make set-env-staging`| Set `.env` to staging environment            |
| `make set-env-prod`   | Set `.env` to production environment         |

---

## 🌐 API Usage

Once the app is running locally:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check:
  ```bash
  curl http://localhost:8000/
  ```
- Prediction:
  ```bash
  curl -X POST http://localhost:8000/predict        -H "Content-Type: application/json"        -d @sample_input.json
  ```

---

## 🧪 Example API Request

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

## 💫 Future Plans

- Add live monitoring with **Evidently**
- Auto-retraining pipelines with model drift detection
- Integrate data from JWST or future exoplanet missions

---

## 👨‍🚀 Author

**Andrea** — MSc Data Science Student & Aspiring MLOps Engineer  
Driven by a passion for space exploration and the scientific process, this project is a small but concrete contribution to the future of astrophysical research.

[LinkedIn](https://www.linkedin.com/in/andrea-marella/) | [GitHub](https://github.com/TheItalianDataGuy)

---

## 📄 License

This project is licensed under the **MIT License**.