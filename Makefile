.PHONY: format lint test run train register-best

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
	PYTHONPATH=. pytest tests/

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