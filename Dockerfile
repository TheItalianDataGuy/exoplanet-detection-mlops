# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies explicitly
RUN pip install scikit-learn mlflow uvicorn apache-airflow
RUN mkdir -p /var/log/airflow
RUN chmod -R 777 /var/log/airflow


# Copy source code, models, and configuration files into the container
COPY ./src /app/src
COPY ./models /app/models
COPY ./src/config /opt/airflow/config

# Copy supervisord configuration
COPY ./docker/supervisord.conf /etc/supervisord.conf

# Expose necessary ports for FastAPI (8000), Airflow (8080), and MLflow (5001)
EXPOSE 8000 8080 5001

# Command to start services using supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
