#!/usr/bin/env python
# coding: utf-8

from logging import Logger
import os

import sys

import pickle

import pandas as pd

import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import pathlib

import uuid

import datetime
from dateutil.relativedelta import relativedelta

import prefect
from prefect import flow, task

# Have to create directory
# get_ipython().system('mkdir output/green')


import logging
logger = logging.getLogger('log_score.log')

def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    return df[categorical + numerical].to_dict(orient='records')

def load_model(exp_id, run_id):
    logged_model = logged_model = f's3://mlops-week4/{exp_id}/{run_id}/artifacts/model'
    return mlflow.pyfunc.load_model(logged_model)

@task
def apply_model(input_file, run_id, output_file):
    logger.info(f'reading the data from {input_file}')
    df = read_dataframe(input_file)
    dict_features = prepare_dictionaries(df)

    logger.info(f'loading the model {run_id}')
    model = load_model('1', run_id)

    logger.info('applying the model...')
    y_pred = model.predict(dict_features)

    logger.info(f'saving the results to {output_file}')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['lpep_dropoff_datetime'] = df['lpep_dropoff_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)


def get_path_vars(run_date, taxi_type):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    
    return input_file, output_file

@flow
def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        run_date: datetime.datetime = None):
    if run_date is None:
        ctx = prefect.get_run_context()
        run_date = ctx.flow_run.expected_start_time
        
    input_file, output_file = get_path_vars(run_date, taxi_type)
    
    path = f'output/{taxi_type}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    apply_model(input_file, 
                run_id, 
                output_file)


def run():
    taxi_type = sys.argv[1] # green
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    run_id = sys.argv[4] # '602e2fa2a0df4f5a87eef98f93b79090'

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date = datetime.datetime(year=year, month=month, day=1)
    )
    

if __name__ == '__main__':
    run()