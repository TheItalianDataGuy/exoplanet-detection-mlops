.PHONY: format lint test run run-prod train register-best \
        mlflow-ui airflow-init start-airflow stop-airflow \
        trigger-ml-pipeline list-dags docker-down clean \
        start

# -------------------
# Code Quality & Testing
# -------------------

format:
	@echo "Formatting code with Black and Ruff"
	black .
	ruff check . --fix

lint:
	@echo "Linting code with Ruff and Flake8"
	ruff check .
	flake8 .

test:
	@echo "Running unit tests"
	PYTHONPATH=$(pwd)/src python -m pytest tests/

# -------------------
# Model Operations
# -------------------

train:
	@echo "Training baseline model"
	python src/models/train_baseline.py

register-best:
	@echo "Registering best model to MLflow"
	python src/models/register_best_model.py

# -------------------
# App Server (FastAPI)
# -------------------

run:
	@echo "Starting FastAPI app (development)"
	uvicorn src.serve.main:app --reload

run-prod:
	@echo "Starting FastAPI app (production)"
	ENV=prod uvicorn src.serve.main:app --host 0.0.0.0 --port 8000

# -------------------
# MLflow UI
# -------------------

mlflow-ui:
	@echo "Starting MLflow UI"
	@mkdir -p mlflow/mlruns
	mlflow server \
		--backend-store-uri sqlite:///mlflow/mlruns.db \
		--default-artifact-root ./mlflow/mlartifacts \
		--host 0.0.0.0 \
		--port 5001

# -------------------
# Airflow Commands
# -------------------

airflow-init:
	@echo "Initializing Airflow database"
	airflow db migrate

start-airflow:
	@echo "Starting Airflow webserver and scheduler"
	airflow webserver --port 8080 &
	airflow scheduler &

stop-airflow:
	@echo "Stopping Airflow"
	@kill $$(lsof -t -i:8080) || true
	@kill $$(lsof -t -i:8793) || true

import-airflow-vars:
	@echo "Importing Airflow variables from JSON"
	docker cp airflow/airflow_variables.json mlops-airflow:/opt/airflow/airflow_variables.json
	docker exec -it mlops-airflow airflow variables import /opt/airflow/airflow_variables.json

# -------------------
# Docker Commands
# -------------------

PORTS := 5001 8000 8080

check-ports:
	@echo "Checking for running processes on ports $(PORTS)"
	@for port in $(PORTS); do \
		pid=$$(lsof -ti tcp:$$port); \
		if [ -n "$$pid" ]; then \
			echo "Port $$port is in use by PID $$pid. Attempting to terminate..."; \
			kill -9 $$pid || echo "Failed to kill process on port $$port. Please stop it manually."; \
		else \
			echo "Port $$port is free."; \
		fi \
	done

build:
	docker compose build

start: check-ports
	docker compose up -d
	$(MAKE) import-airflow-vars

stop:
	docker compose down

logs:
	docker compose logs -f

fastapi-logs:
	docker compose logs -f fastapi

mlflow-logs:
	docker compose logs -f mlflow

airflow-logs:
	docker compose logs -f airflow

restart: stop start

clean:
	docker system prune -af
	docker volume prune -f

train-docker:
	@echo "Training model inside Docker container"
	docker compose exec fastapi python src/models/train_baseline.py

check-docker:
	@docker info > /dev/null 2>&1 || (echo "Docker daemon is not running!" && exit 1)

# -------------------
# Environment Helpers
# -------------------

set-env-local:
	@echo "ENV=local" > .env && echo "Set environment to LOCAL"

set-env-docker:
	@echo "Setting environment to 'docker'"
	@sed -i '' 's/^ENV=.*/ENV=docker/' .env || echo "ENV=docker" >> .env

set-env-staging:
	@echo "ENV=staging" > .env && echo "Set environment to STAGING"

set-env-prod:
	@echo "ENV=prod" > .env && echo "Set environment to PROD"

