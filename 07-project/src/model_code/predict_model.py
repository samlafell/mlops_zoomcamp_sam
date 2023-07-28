import hashlib
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

import boto3
import mlflow
import pandas as pd
import polars as pl
from mlflow.tracking import MlflowClient

# Set the tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:5000")  # Replace with your tracking URI
mlflow_client = MlflowClient()


class ModelService:
    def __init__(self, model_name, data=None, model=None, model_stage="Production"):
        self.model_name = model_name
        self.model = model
        self.model_stage = model_stage
        self.data = data
        self.model_version = None

    # Get the latest version of the model in the 'Production' stage
    def get_model_version(self):
        try:
            self.model_version = mlflow_client.get_latest_versions(
                self.model_name, stages=[self.model_stage]
            )[0]
            logger.info(
                f"Found model version of {self.model_version.name} at {self.model_version.source}"
            )
            return self
        except:
            logger.error("Could not find model version, check model name and stage")
            return None

    def load_model(self):
        try:
            self.model = mlflow.pyfunc.load_model(self.model_version.source)
            logger.info("Loaded model")
            return self
        except:
            logger.error("Could not load model, check model name")
            return None

    def predict(self, data):
        logger.info("Making Predictions")
        preds = self.model.predict(data)
        return preds


class Data:
    def __init__(self, project_dir):
        self.data_dir = project_dir / "data" / "processed"

    def load_features(self):
        features = pl.read_csv(self.data_dir / "X_test.csv")
        return features

    def load_labels(self):
        labels = pl.read_csv(self.data_dir / "y_test.csv")
        return labels


# Hashing function
def create_hash(row):
    """
    Creating a hash for each unique row in the input data so when we make predictions we know how to pair up to the rows
    """
    # create a hash object
    hash_object = hashlib.md5(row.encode())
    # get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


def main():
    # Load Data
    data = Data(project_dir).load_features()

    data_df = data.to_pandas()  # Convert to pandas dataframe

    # Concatenate all columns to a single string and create a hash for each row
    data_df["id"] = data_df.apply(
        lambda row: create_hash("".join(row.values.astype(str))), axis=1
    )

    # Convert pandas dataframe back to Polars
    data = pl.from_pandas(data_df)

    # Load Model
    model_service = ModelService(model_name="BestWineDatasetModel")
    model_service.get_model_version().load_model()

    # Predict
    preds = model_service.predict(data.drop("id"))

    # Add these predictions back to the original dataframe (which includes the 'id' column)
    data_df["predictions"] = preds

    # Output predictions to csv
    preds_df = data_df[["id", "predictions"]]
    csv_buffer = StringIO()
    preds_df.to_csv(csv_buffer, index=False)

    # Output to S3
    pred_date = datetime.strftime(datetime.today().date(), "%Y%m%d")
    s3_resource = boto3.resource("s3")
    s3_resource.Object(
        "sal-wine-quality",
        f"models/{model_service.model_name}/preds/{pred_date}/predictions.csv",
    ).put(Body=csv_buffer.getvalue())
    return preds


# You can now use model.predict() to generate predictions
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()
