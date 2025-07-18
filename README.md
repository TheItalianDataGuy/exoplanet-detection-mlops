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

## 🧪 Project Structure

```
.
├── data/                      # Data files
├── models/                    # Saved models
├── src/                       # Source code
│   ├── train/                 # Training scripts
│   ├── predict/               # Prediction logic
│   └── serve/                 # FastAPI app
├── tests/                     # Unit and integration tests
├── .env.example               # Environment variable template
├── Makefile                   # Command shortcuts
├── requirements.txt           # Python dependencies
└── README.md
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

## 🛠️ Usage

### Train the model

```bash
make train
```

### Run the API locally

```bash
make run
```

Visit: `http://127.0.0.1:8000/docs` to access Swagger UI.

### Run tests

```bash
make test
```

### Format and lint code

```bash
make format
make lint
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

Andrea — MSc Data Science Candidate & Aspiring MLOps Engineer  
Feel free to connect on [LinkedIn](https://www.https://www.linkedin.com/in/andrea-marella/) or check out my [GitHub](https://github.com/TheItalianDataGuy).

---

## 📄 License

MIT License.
