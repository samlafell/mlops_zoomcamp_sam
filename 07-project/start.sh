#!/bin/bash

# Start the MLFlow server
mlflow server -h 0.0.0.0 -p 5001 --backend-store-uri postgresql://mlflow:9xh2Om9ryDbpPWLoFe8z@mlflow-project-db-ubuntu-2.cdflealfthrn.us-east-2.rds.amazonaws.com:5432/mlflow_db --default-artifact-root s3://sal-wine-quality

# Run your Flask application
python app.py