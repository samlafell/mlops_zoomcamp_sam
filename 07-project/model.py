import os
import json
import base64

import boto3
import mlflow

def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')
    if model_location:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlops-week4')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
    model_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/model'
    return model_location


def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    return model
