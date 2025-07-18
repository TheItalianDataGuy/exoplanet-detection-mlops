.PHONY: format lint test run train register-best

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
	python src/train/train_baseline.py