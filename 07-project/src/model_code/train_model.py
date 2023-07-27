from pathlib import Path
import logging
import click
from dotenv import find_dotenv, load_dotenv
import polars as pl
# More Models
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from typing import Dict, Any
from numpy import ndarray
from xgboost import DMatrix
import mlflow
from sklearn.metrics import mean_squared_error
import os

import configparser

# Set MLFlow Tracking URI

# From EC2
mlflow.set_tracking_uri('http://localhost:5000')

# From Local
#ec2_public_ipv4 = 'ec2-3-14-150-154.us-east-2.compute.amazonaws.com'
#mlflow.set_tracking_uri(f'http://{ec2_public_ipv4}:5000')

logging.info(f'tracking uri: {mlflow.get_tracking_uri()}')
# Set Experiment
mlflow.set_experiment('wine_dataset')

class ImportData():
    def __init__(self, data_filepath):
        self.data_path = Path(data_filepath)
        
    def set_paths(self):
        self.X_train_path = self.data_path / 'X_train.csv'
        self.y_train_path = self.data_path / 'y_train.csv'
        
        self.X_val_path = self.data_path / 'X_val.csv'
        self.y_val_path = self.data_path / 'y_val.csv'
        
        return self
    
    def import_data(self):
        X_train = pl.read_csv(self.X_train_path)
        y_train = pl.read_csv(self.y_train_path)
        X_val = pl.read_csv(self.X_val_path)
        y_val = pl.read_csv(self.y_val_path)
        return X_train, y_train, X_val, y_val


def convert_to_numpy(**args):
    result = {}
    for arg_name, arg in args.items():
        if isinstance(arg, pl.DataFrame):
            result[arg_name] = arg.to_numpy().flatten() if arg.shape[1] == 1 else arg.to_numpy()
        else:
            result[arg_name] = arg
    return result.values()


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('early_stopping_rounds_var', type=click.INT)
def main(data_filepath, early_stopping_rounds_var):
    """
    Training Model
    """
    logger = logging.getLogger(__name__)
    logger.info('training model')
    
    # Initialize preprocess class
    importing = ImportData(data_filepath)
    importing.set_paths()
    # Load in X and y's
    X_train, y_train, X_val, y_val = importing.import_data()
    X_train_np, y_train_np, X_val_np, y_val_np = convert_to_numpy(X_train=X_train, 
                                                      y_train=y_train,
                                                      X_val=X_val,
                                                      y_val=y_val)
    
    # Train Model
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'multi:softmax',
        'num_class': 9,
        'seed': 42
    }

    logging.info('passing to DMatrix')
    train = xgb.DMatrix(data=X_train_np, label = y_train_np)
    valid = xgb.DMatrix(data=X_val_np, label = y_val_np)

    def objective(params: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Train an XGBoost model with given parameters and datasets, log the training 
        process with MLFlow, make predictions on the validation set, calculate RMSE 
        and log it with MLFlow. Return the RMSE and the status.

        Parameters:
        params (dict): A dictionary of parameters to use for the XGBoost model.

        Returns:
        dict: A dictionary with 'loss' key indicating the root mean squared error (RMSE) 
        on validation set and 'status' key indicating the status of the function.
        '''
        logging.info('inside objective function')
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            mlflow.xgboost.autolog()
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=early_stopping_rounds_var,
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}
    
    logging.info('starting hyperparameter search')
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    #for name, value in os.environ.items():
    #    print("{0}: {1}".format(name, value))
    
    main()