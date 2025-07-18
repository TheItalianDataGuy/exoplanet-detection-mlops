.PHONY: format lint test run

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
