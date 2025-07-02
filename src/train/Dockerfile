# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY ../../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and relevant folders
COPY ../../src ./src
COPY ../../data ./data
COPY ../../models ./models

# Run training script
CMD ["python", "src/train/train.py"]

# Provision a volume for output files to persist after container execution
VOLUME ["/app/data", "/app/models", "/app/logs", "/app/outputs"]
