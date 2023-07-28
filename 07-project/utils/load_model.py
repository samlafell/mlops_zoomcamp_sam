import json
import logging

import mlflow
import polars as pl
import polars.selectors as cs
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

mlflow.set_tracking_uri("http://0.0.0.0:5001")
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
            logger.error(f"Could not load model, check model name: {e}")
            return None
        
    def get_columns(self):
        try:
            # Get the run ID of the model
            run_id = self.model_version.run_id

            # Build the columns file path and download the file
            columns_file_path = mlflow_client.download_artifacts(run_id, "columns.json")

            # Load the columns from the file
            with open(columns_file_path, "r") as f:
                self.original_columns = json.load(f)

            logger.info('Loaded columns file')
            return self
        
        except Exception as e:
            logger.error(f"Could not load columns file: {e}")
            return None
        
    def check_columns(self, data):
        if set(data.columns) != set(self.original_columns):
            print(f'Input cols: {set(data.columns)}')
            print(f'Original cols: {set(self.original_columns)}')
            raise ValueError("Columns of the input data do not match the original columns")

    def predict(self, data):
        logger.info("Making Predictions")
        return self.model.predict(data)
    
def standardize_data(data, id_col="Id"):
    """
    To get to STD 1 and Mean 0
    Polars does not have a built in standardization function, so we have to do it manually
    This is done separately for each dataset to avoid data leakage
    """
    cols_to_grab = data.select(cs.integer().exclude(id_col), cs.float()).columns
    for col_name in cols_to_grab:
        if col_name == id_col:
            None
        else:
            col_mean = data[col_name].mean()
            col_std = data[col_name].std()
            data = data.with_columns(
                ((data[col_name] - col_mean) / col_std).alias(f"{col_name}_std")
            )
    return data.select(cs.ends_with("_std"))


# if __name__=='__main__':
#     model_service = ModelService("BestWineDatasetModel")
#     model = model_service.get_model_version()
#     columns = model.get_columns()
#     model = model.load_model().model