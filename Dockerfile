# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MLflow, FastAPI, Uvicorn, and Airflow
RUN pip install mlflow uvicorn apache-airflow

# Copy the rest of your application files
COPY . .

# Expose ports for FastAPI (8000), MLflow (5001), and Airflow (8080)
EXPOSE 8000
EXPOSE 5001
EXPOSE 8080

# Start FastAPI, MLflow, and Airflow webserver in the same container
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5001 & airflow webserver --port 8080 & uvicorn serve.main:app --host 0.0.0.0 --port 8000"]
