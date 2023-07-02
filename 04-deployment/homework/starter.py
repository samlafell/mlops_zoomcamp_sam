import pickle
import numpy as np
import pandas as pd
import pyarrow
import os
import boto3
import sys
from io import BytesIO
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Create boto3
s3 = boto3.client('s3',
                  aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'), 
                  aws_secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY_ID')
                )

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df[categorical].to_dict(orient='records')


def make_preds(taxi_type, year, month):
    dicts = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')
    X_val = dv.transform(dicts)
    return model.predict(X_val)

def create_output_df(year, month, y_pred):
     df_result = pd.DataFrame()
     df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df_result.index.astype('str')
     df_result['prediction'] = y_pred
     return df_result


def apply_flow(taxi_type, year, month):
    y_pred = make_preds(taxi_type, year, month)
    return create_output_df(year, month, y_pred)    

def run():
    taxi_type = sys.argv[1] # green
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    #RUN_ID = sys.argv[4] # '602e2fa2a0df4f5a87eef98f93b79090'
    
    output = apply_flow(taxi_type, year, month)
    s3_bucket_name = 'mlops-week4'

    # Convert dataframe to Parquet and save in buffer
    parquet_buffer = BytesIO()
    output.to_parquet(parquet_buffer, 
                      engine='pyarrow')

    # Write buffer to S3 object
    ## s3.Object is used with boto3.resource
    # s3.Object(s3_bucket_name, 'docker_april2022_preds.parquet').put(Body=parquet_buffer.getvalue())
    
    ## s3.put_object is used with boto3.client
    parquet_buffer.seek(0)
    s3.put_object(Body = parquet_buffer,
                  Bucket = s3_bucket_name,
                  Key = 'docker_april2022_preds.parquet')
    
if __name__ == '__main__':
    
    run()



# y_pred = model.predict(X_val)

# df_result = pd.DataFrame()
# df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# df_result['prediction'] = y_pred

# output_path = f"output/{taxi_type}/{year:04d}"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# output_file = f'{output_path}/{month:02d}.parquet'
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
