.PHONY: format lint test run train register-best run-airflow start-dag stop-dag mlflow-ui

# Format code using Black and Ruff
format:
	black .
	ruff check . --fix

# Lint code using Ruff and Flake8
lint:
	ruff check .
	flake8 .

# Run tests
test:
	PYTHONPATH=$(pwd)/src python -m pytest tests/

# Start FastAPI app
run:
	uvicorn src.serve.main:app --reload

# Train baseline model
train:
	python src/models/train_baseline.py

# Register best model to MLflow registry
register-best:
	python src/models/register_best_model.py

# Run FastAPI app in production mode
run-prod:
	ENV=prod uvicorn src.serve.main:app --host 0.0.0.0 --port 8000

# Start MLflow UI
mlflow-ui:
	@mkdir -p mlflow/mlruns
	mlflow server \
		--backend-store-uri sqlite:///mlflow/mlruns.db \
		--default-artifact-root ./mlflow/mlruns \
		--host 0.0.0.0 \
		--port 5001

# Start Airflow Web Server
start-airflow-web:
	airflow webserver --port 8080

# Start Airflow Scheduler
start-airflow-scheduler:
	airflow scheduler

# Initialize Airflow Database (first-time setup)
airflow-init:
	airflow db init

# Trigger the ML pipeline DAG manually
trigger-ml-pipeline:
	airflow dags trigger ml_pipeline_dag

# List all running DAGs in Airflow
list-dags:
	airflow dags list

# Start Airflow Web Server and Scheduler (combined)
start-airflow:
	@echo "Starting Airflow web server and scheduler..."
	airflow webserver --port 8080 &
	airflow scheduler &

# Stop Airflow services
stop-airflow:
	@echo "Stopping Airflow..."
	kill $(lsof -t -i:8080) # Kill the webserver
	kill $(lsof -t -i:8793) # Kill the scheduler

