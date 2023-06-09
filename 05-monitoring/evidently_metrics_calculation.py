import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	fare_amt_25pct float,
	fare_amt_50pct float,
	fare_amt_75pct float
)
"""

# Import Reference data
reference_data = pd.read_parquet('data/reference.parquet')
# Load the Model
with open('models/lin_reg.bin', 'rb') as f_in:
	model = joblib.load(f_in)

# Simulate production usage, we are going to read our data day by day
raw_data = pd.read_parquet('data/green_tripdata_2023-03.parquet')
# Know where to start
begin = datetime.datetime(2023, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']

# In order to use Evidently we have to generate a report
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

# Then the 2nd part of generating a report
report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ColumnQuantileMetric(column_name='fare_amount', quantile=.25),
    ColumnQuantileMetric(column_name='fare_amount', quantile=.5),
    ColumnQuantileMetric(column_name='fare_amount', quantile=.75)
])

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task(retries=2, retry_delay_seconds=5, name="calculate metrics")
def calculate_metrics_postgresql(curr, i):
	"""_summary_

	Args:
		curr (_type_): connection cursor
		i (int): number of days in the month, calculate in the values for 1 month day by day
	"""
    # Pick up the date of interest
	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
                         (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

	#current_data.fillna(0, inplace=True)
	current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))
 	
	report.run(reference_data = reference_data, current_data = current_data, column_mapping=column_mapping)
 
	result = report.as_dict()
 	
	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
	fare_amt_25pct = result['metrics'][3]['result']['current']['value']
	fare_amt_50pct = result['metrics'][4]['result']['current']['value']
	fare_amt_75pct = result['metrics'][5]['result']['current']['value']
  

	curr.execute(
		"INSERT INTO dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, fare_amt_25pct, fare_amt_50pct, fare_amt_75pct) values (%s, %s, %s, %s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values, fare_amt_25pct, fare_amt_50pct, fare_amt_75pct)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=5)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(30):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()