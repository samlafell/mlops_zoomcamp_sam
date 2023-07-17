#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import os
import pandas as pd
import boto3
import s3fs

AWS_ENDPOINT_URL = 'http://localhost:4566'  # or the URL where LocalStack is running

s3 = boto3.client('s3', endpoint_url=AWS_ENDPOINT_URL)
s3_resource = boto3.resource('s3', endpoint_url=AWS_ENDPOINT_URL)
s3_fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': AWS_ENDPOINT_URL})

options = {
    'client_kwargs': {
        'endpoint_url': AWS_ENDPOINT_URL
    }
}

def read_data(filename):
    if AWS_ENDPOINT_URL:
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
    return df
    
def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# def get_input_path(year, month):
#     default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
#     input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
#     return input_pattern.format(year=year, month=month)

def get_input_path_test(file):
    default_input_pattern = f's3://nyc-duration/{file}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/yellow/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def save_data(df, output_file):
    with s3_fs.open(output_file, 'wb') as f:
        df.to_parquet(f)
    
def main(year, month):
    input_file = get_input_path_test('test')
    output_file = get_output_path(year, month)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())
    print('predicted sum duration', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print(output_file)
    save_data(df_result, output_file)
    

if __name__=="__main__":
    YEAR = int(sys.argv[1])
    MONTH = int(sys.argv[2])
    main(YEAR, MONTH)
