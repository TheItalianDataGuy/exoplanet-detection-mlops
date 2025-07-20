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

# -------------------
# Docker Commands
# -------------------

free-ports:
	@echo "Checking and freeing ports 5001, 8000, and 8080"
	@for port in 5001 8000 8080; do \
		pids=$$(lsof -ti tcp:$$port); \
		if [ -n "$$pids" ]; then \
			echo "Port $$port is in use by PID(s): $$pids"; \
			for pid in $$pids; do \
				echo "Trying to kill PID $$pid..."; \
				kill -9 $$pid 2>/dev/null || echo "Could not kill PID $$pid. You may need to close it manually."; \
			done; \
		else \
			echo "Port $$port is free."; \
		fi; \
	done

start:
	@echo "Checking for running processes on ports 5001, 8000, and 8080"
	@bash -c '\
		for port in 5001 8000 8080; do \
			if lsof -i :$$port >/dev/null 2>&1; then \
				echo "Port $$port is already in use. Please free it before continuing."; \
				exit 1; \
			fi; \
		done'
	@echo "Starting Docker Compose services"
	docker-compose up --build

docker-down:
	@echo "Stopping and removing Docker Compose containers"
	docker-compose down --remove-orphans

clean:
	@echo "Cleaning Docker system (containers, images, volumes, and cache)"
	docker system prune -af --volumes
