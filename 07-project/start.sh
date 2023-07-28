#!/bin/bash

# Start the MLFlow server
mlflow server -h 0.0.0.0 -p 5001 --backend-store-uri $MLFLOW_URI --default-artifact-root s3://sal-wine-quality &

# Run your Flask application
python app.py