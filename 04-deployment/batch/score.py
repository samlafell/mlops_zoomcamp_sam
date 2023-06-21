#!/usr/bin/env python
# coding: utf-8

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

# Have to create directory
# get_ipython().system('mkdir output/green')

def generate_uuids(n):
    ride_ids = []
    # Create Ride IDs with UUID
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

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
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# In[9]:


def load_model(exp_id, run_id):
    logged_model = logged_model = f's3://mlops-week4/{exp_id}/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    print(f'reading the data from {input_file}')
    df = read_dataframe(input_file)
    dict_features = prepare_dictionaries(df)
    
    print(f'loading the model {run_id}')
    model = load_model('1', run_id)
    
    print(f'applying the model...')
    y_pred = model.predict(dict_features)
    
    print(f'saving the results to {output_file}')
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


def run():
    taxi_type = sys.argv[1] # green
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    RUN_ID = sys.argv[4] # '602e2fa2a0df4f5a87eef98f93b79090'

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    apply_model(input_file, 
                RUN_ID, 
                output_file)


if __name__ == '__main__':
    run()