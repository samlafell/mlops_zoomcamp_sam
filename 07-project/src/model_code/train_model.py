# pylint disable=unused-import
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import click
import mlflow
import polars as pl
import polars.selectors as cs
import json

# More Models
import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from numpy import ndarray
from sklearn.metrics import mean_squared_error
from xgboost import DMatrix

# Set MLFlow Tracking URI
# From EC2
mlflow.set_tracking_uri("http://localhost:5001")

# From Local
# ec2_public_ipv4 = 'ec2-3-14-150-154.us-east-2.compute.amazonaws.com'
# mlflow.set_tracking_uri(f'http://{ec2_public_ipv4}:5000')
logging.info(f"tracking uri: {mlflow.get_tracking_uri()}")
# Set Experiment
mlflow.set_experiment("wine_dataset")

SEARCH_SPACE = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "multi:softmax",
    "num_class": 9,
    "seed": 42,
    "eval_metric": hp.choice("eval_metric", ['mlogloss'])  # or any other metrics you want to try
}


class ImportData:
    def __init__(self, data_filepath):
        self.data_path = Path(data_filepath)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def import_data(self):
        X_train_path = self.data_path / "X_train.csv"
        y_train_path = self.data_path / "y_train.csv"
        X_val_path = self.data_path / "X_val.csv"
        y_val_path = self.data_path / "y_val.csv"

        self.X_train = pl.read_csv(X_train_path)
        self.y_train = pl.read_csv(y_train_path)
        self.X_val = pl.read_csv(X_val_path)
        self.y_val = pl.read_csv(y_val_path)
        return self

    def convert_to_numpy(self):
        X_train = self.X_train.select(cs.ends_with("_std"))
        X_val = self.X_val.select(cs.ends_with("_std"))
        datasets = [X_train, self.y_train, X_val, self.y_val]
        assert all(
            dataset is not None for dataset in datasets
        ), "One of the train/validation datasets does not exist"
        assert len(datasets) == 4, "Datasets List does not have all 4 elements"
        result = []
        for dataset in datasets:
            if dataset.shape[1] == 1:
                result.append(dataset.to_numpy().flatten())
            else:
                result.append(dataset.to_numpy())
        return result


def objective(
    params: Dict[str, Any],
    train: xgb.DMatrix,
    valid: xgb.DMatrix,
    y_val: pl.DataFrame,
    metrics: str,
) -> Dict[str, Any]:
    """
    Train an XGBoost model with given parameters and datasets, log the training
    process with MLFlow, make predictions on the validation set, calculate RMSE
    and log it with MLFlow. Return the RMSE and the status.

    Parameters:
    params (dict): A dictionary of parameters to use for the XGBoost model.

    Returns:
    dict: A dictionary with 'loss' key indicating the root mean squared error (RMSE)
    on validation set and 'status' key indicating the status of the function.
    """
    logging.info("inside objective function")
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        mlflow.xgboost.autolog()
            
        booster = xgb.train(
            params={**params, 'eval_metric': metrics},
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )
    
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        # Log the columns file as an artifact
        mlflow.log_artifact("columns.json")

    return {"loss": rmse, "status": STATUS_OK}


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("max_evals_var", type=click.INT)
def main(data_filepath, max_evals_var):
    # pylint: disable=unused-variable,unbalanced-tuple-unpacking
    """
    Training Model
    """
    logger = logging.getLogger(__name__)
    logger.info("training model")

    # Initialize preprocess class
    importing = ImportData(data_filepath)
    # Import Data
    importing.import_data()
    
    # Select original columns in dataframe
    columns = importing.X_train.select(~cs.ends_with("_std")).columns
    with open("columns.json", "w") as f:
        json.dump(columns, f)
    
    # Prepare for Training
    X_train_np, y_train_np, X_val_np, y_val_np = importing.convert_to_numpy()

    logging.info("passing to DMatrix")
    train = xgb.DMatrix(data=X_train_np, label=y_train_np)
    valid = xgb.DMatrix(data=X_val_np, label=y_val_np)

    logging.info("starting hyperparameter search")
    best_result = fmin(
        fn=lambda params: objective(
            params, train, valid, importing.y_val, params['eval_metric']
        ),
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=max_evals_var,
        trials=Trials(),
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # for name, value in os.environ.items():
    #    print("{0}: {1}".format(name, value))

    main()
