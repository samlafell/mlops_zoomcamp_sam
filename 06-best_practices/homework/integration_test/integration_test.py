import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
]


columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
input_df = pd.DataFrame(data, columns=columns)

AWS_ENDPOINT_URL = 'http://localhost:4566'  # or the URL where LocalStack is running

options = {
    'client_kwargs': {
        'endpoint_url': AWS_ENDPOINT_URL
    }
}

input_df.to_parquet(
    's3://nyc-duration/test.parquet',
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)