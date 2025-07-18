.PHONY: format lint test run train

format:
	black .
	ruff check . --fix

lint:
	ruff check .
	flake8 .

test:
	PYTHONPATH=. pytest tests/

run:
	uvicorn src.serve.main:app --reload

train:
	python src/models/train_baseline.py

register-best:
	python src/models/register_best_model.py