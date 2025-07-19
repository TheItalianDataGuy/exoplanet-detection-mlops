# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install mlflow uvicorn apache-airflow

# Copy the entire src folder into the Docker image
COPY ./src /app/src

# Copy sample_input.json and expected_columns.json into the Docker image
COPY ./models/sample_input.json /app/sample_input.json
COPY ./models/expected_columns.json /app/models/expected_columns.json
COPY ./models/random_forest.joblib /app/models/random_forest.joblib

# Expose necessary ports
EXPOSE 8000 5000 8080

# Start the FastAPI app and Airflow web server
CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
