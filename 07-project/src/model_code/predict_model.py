import hashlib
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

import boto3
import mlflow
import pandas as pd
import polars as pl
import polars.selectors as cs
from mlflow.tracking import MlflowClient

# Set the tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:5001")  # Replace with your tracking URI
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
        except Exception as e:
            logger.error(f"Could not find model version, check model name: {e}")
            return None

    def load_model(self):
        try:
            self.model = mlflow.pyfunc.load_model(self.model_version.source)
            logger.info("Loaded model")
            return self
        except Exception as e:
            logger.error("Could not load model, check model name: {e}")
            return None

    def predict(self, data):
        logger.info("Making Predictions")
        return self.model.predict(data)

class DataLoader:
    def __init__(self, project_dir):
        self.data_dir = project_dir / "data" / "processed"

    def data(self):
        self.data = pl.read_csv(self.data_dir / "X_test.csv")
        return self

def date_iter_var(data):
    return data.select(pl.col("date")).unique().to_series().to_list()

def main():
    # Load Data
    data_obj = DataLoader(project_dir).load_features()
    model_data = data_obj.data.select(cs.ends_with("_std"), "date", "Id")
    
    # Load Model
    model_service = ModelService(model_name="BestWineDatasetModel")
    model_service.get_model_version().load_model()

    dates_in_df = date_iter_var(data_obj.data)
    for date in dates_in_df:
        model_data_loop = model_data.filter(pl.col("date") == date)
        preds = model_service.predict(model_data_loop.select(cs.ends_with("_std")))
        model_data_loop = model_data_loop.with_columns(pl.Series(preds).alias("predictions"))
        
        # Output predictions to csv
        preds_df = model_data_loop.select('Id', 'predictions').to_pandas()
        csv_buffer = StringIO()
        preds_df.to_csv(csv_buffer, index=False)

        # Output to S3
        dt = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
        file_date = dt.strftime("%Y%m%d")
        s3_resource = boto3.resource("s3")
        s3_resource.Object(
            "sal-wine-quality",
            f"models/{model_service.model_name}/preds/{file_date}/predictions.csv",
        ).put(Body=csv_buffer.getvalue())
        
    return preds


# You can now use model.predict() to generate predictions
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()
