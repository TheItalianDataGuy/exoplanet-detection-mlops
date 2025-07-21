# 🌌 Exoplanet Detection with MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://github.com/TheItalianDataGuy/exoplanet-detection-mlops/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-enabled-blue)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Airflow](https://img.shields.io/badge/Airflow-scheduled-9cf)
![API Swagger UI Screenshot](images/swagger-ui.png)
![CI](https://github.com/TheItalianDataGuy/actions/workflows/ci.yml/badge.svg)

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

- 📦 Model training with **RandomForestClassifier**
- 📈 Experiment tracking via **MLflow**
- 📁 Model versioning via **MLflow Model Registry**
- 🔁 Batch inference support
- 📡 REST API with **FastAPI**
- 🧪 Unit & integration testing with **pytest**
- 🧹 Code quality with **black**, **ruff**, and **flake8**
- ✅ Pre-commit hooks for consistency
- ✅ CI/CD with **GitHub Actions**
- 🐳 Docker-based deployment
- 🧪 Monitoring with **Evidently**
- 🧱 Workflow orchestration with **Airflow**
- 📬 Email alerts on task failure via Airflow `EmailOperator`
- ⚙️ Reproducibility via `Makefile`, `requirements.txt`, `.env.example`

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

### 🛠 Makefile Commands

| Command                 | Description                                         |
|-------------------------|-----------------------------------------------------|
| `make format`           | Format code with **Black** and **Ruff**             |
| `make lint`             | Run linting checks with **Ruff** and **Flake8**     |
| `make test`             | Run all unit tests using **pytest**                 |
| `make train`            | Train the baseline model                            |
| `make register-best`    | Register the best model to **MLflow**               |
| `make run`              | Start FastAPI app in development mode               |
| `make run-prod`         | Start FastAPI app in production mode                |
| `make mlflow-ui`        | Launch MLflow tracking UI on port 5001              |
| `make airflow-init`     | Initialize Airflow metadata DB                      |
| `make start-airflow`    | Start Airflow webserver and scheduler               |
| `make stop-airflow`     | Stop all Airflow processes                          |
| `make import-airflow-vars` | Import Airflow variables from JSON file          |
| `make backfill-dag`     | Run a backfill on `ml_pipeline_dag`                 |
| `make check-ports`      | Check and free ports 5001, 8000, 8080               |
| `make build`            | Build Docker containers                             |
| `make start`            | Start Docker containers and import Airflow vars     |
| `make stop`             | Stop all Docker containers                          |
| `make logs`             | Tail logs from all Docker services                  |
| `make fastapi-logs`     | Tail logs from the FastAPI container                |
| `make mlflow-logs`      | Tail logs from the MLflow container                 |
| `make airflow-logs`     | Tail logs from the Airflow container                |
| `make restart`          | Stop and restart all Docker containers              |
| `make clean`            | Remove unused Docker images and volumes             |
| `make train-docker`     | Run training script inside Docker container         |
| `make check-docker`     | Verify Docker daemon is running                     |
| `make set-env-local`    | Set `.env` to use local environment                 |
| `make set-env-docker`   | Set `.env` to use Docker environment                |
| `make set-env-staging`  | Set `.env` to use staging environment               |
| `make set-env-prod`     | Set `.env` to use production environment            |

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


## 🧱 Run with Docker

This project uses **Docker Compose** to orchestrate **Airflow**, **MLflow**, and **FastAPI** services for end-to-end machine learning workflows.

---

### 🔧 Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)

---

### 🚀 Start the Stack

To start all services (Airflow, MLflow, FastAPI) and import Airflow variables automatically:

```bash
make start
```

This will:
- Check if ports 5001, 8000, and 8080 are available
- Start the containers
- Import Airflow variables from `airflow/airflow_variables.json`

---

### 🌀 Airflow

- Access the Airflow UI: [http://localhost:8080](http://localhost:8080)
- Default user credentials are loaded from `.env`
- Run the DAG manually via:

```bash
make backfill-dag
```

---

## 📬 Alerting

The Airflow pipeline includes built-in failure alerts:

- ✅ **Email notifications** using `EmailOperator` with detailed task/log info.
- 🔜 Easy extension to Slack alerts.

Example:
```html
Subject: Airflow Alert: Task split_data Failed in DAG ml_pipeline_dag

Task: split_data  
DAG: ml_pipeline_dag  
Execution Time: 2025-07-21  
View Logs: http://localhost:8080/log?dag_id=ml_pipeline_dag&task_id=split_data
```
---

### 🔭 MLflow

- Access the MLflow Tracking UI: [http://localhost:5001](http://localhost:5001)

---

### ⚙️ FastAPI

- Access the API docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Run manually in development mode:

```bash
make run
```

- Run in production mode:

```bash
make run-prod
```

---

### 🛑 Stop the Stack

To stop all running containers:

```bash
make stop
```

---

### 🧼 Clean Up Docker System

To remove containers, networks, images, and volumes:

```bash
make clean
```

---

## 📦 CI/CD

This project uses **GitHub Actions** for continuous integration:

- Run code linting (`ruff`, `flake8`)
- Execute unit and integration tests
- Prevent merge on failure
- Auto-build on push to main/dev branches

Badge:  
![CI](https://github.com/TheItalianDataGuy/exoplanet-detection-mlops/actions/workflows/ci.yml/badge.svg)

---

## 💫 Future Plans

- Add live monitoring with **Evidently**
- Auto-retraining pipelines with model drift detection

---

## 👨‍🚀 Author

**Andrea** — MSc Data Science Student & Aspiring MLOps Engineer  
Driven by a passion for space exploration and the scientific process, this project is a small but concrete contribution to the future of astrophysical research.

[LinkedIn](https://www.linkedin.com/in/andrea-marella/) | [GitHub](https://github.com/TheItalianDataGuy)

---

## 📄 License

This project is licensed under the **MIT License**.