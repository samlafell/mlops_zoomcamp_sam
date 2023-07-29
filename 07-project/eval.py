import logging
from datetime import datetime
from io import StringIO

import polars as pl
import psycopg2 as p
import s3fs
import boto3
from utils.load_model import ModelService
import mlflow
from sklearn.metrics import log_loss

mlflow.set_tracking_uri("http://0.0.0.0:5001")
model_service = ModelService("BestWineDatasetModel")

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

s3_file_system = s3fs.S3FileSystem()
s3 = boto3.resource('s3')
client = boto3.client('s3')

create_model_performance_table = """
drop table if exists model_performance;
create table model_performance(
	date timestamp,
	misclassification_rate float
)
"""

create_predictions_table = """
drop table if exists predictions;
create table predictions(
	date timestamp,
	Id varchar,
    predictions int,
    quality int
)
"""

def prep_db():
    conn = p.connect("host=localhost port=5432 user=postgres password=example")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("create database test;")
        logger.debug("Created database test")
    conn.close()
    conn = p.connect("host=localhost port=5432 dbname=test user=postgres password=example")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(create_model_performance_table)
    cur.execute(create_predictions_table)
    logger.debug('tables created')
    conn.close()
prep_db()

def append_performance(curr, date, misclassification_rate):
    curr.execute(
        "insert into model_performance(date, misclassification_rate) values (%s, %s)",
        (date, misclassification_rate),
    )
    logger.debug(f"Inserted performance for {date}")
    
def append_predictions(curr, date, Id, predictions, quality):
    curr.execute(
        "insert into predictions(date, Id, predictions, quality) values (%s, %s, %s, %s)",
        (date, Id, predictions, quality),
    )
    logger.info("Inserted predictions")
    
def list_files(bucket: str, prefix: str):
    return [obj.key for obj in s3.Bucket(bucket).objects.filter(Prefix=prefix)]

def append_files_to_database(bucket: str = "sal-wine-quality", 
                             s3_path: str = f"models/{model_service.model_name}/preds/"):
    
        base_df = pl.DataFrame()
        # Traverse each directory under the provided s3_path
        for obj in list_files(bucket, s3_path):
            # For each directory, find the CSV files
            for file in list_files(bucket, obj):
                if file.endswith('.csv'):
                    # Get the file object
                    s3_object = client.get_object(Bucket=bucket, Key=file)

                    # Read file content
                    file_content = s3_object['Body'].read().decode('utf-8')

                    # Create a StringIO object for pandas read_csv
                    csv_content = StringIO(file_content)

                    # Read CSV using Polars
                    df_collected = pl.read_csv(csv_content)
                    base_df = base_df.vstack(df_collected)
                    
                    # Predictions
                    preds = base_df.with_columns((pl.col('predictions').cast(pl.Int64) == pl.col('quality')).cast(pl.Int8).alias('correct'))
                    
                    # Calculate the misclassification rate for each day (incorrect predictions / total predictions)
                    preds = preds.with_columns(pl.col("correct").rolling_sum(11).alias("rolling_correct")) \
                            .with_columns(pl.lit(11).alias("rolling_total")) \
                            .with_columns((1 - (pl.col("rolling_correct") / pl.col("rolling_total"))).alias("misclassification_rate"))
                            
                    if preds[-1, 'misclassification_rate'] is not None:
                        with p.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
                            conn.autocommit = True
                            curr = conn.cursor()
                            # Append last two rows to database
                            for row in [-1, -2]:
                                date = preds[row, 'date']
                                misclassification_rate = preds[row, 'misclassification_rate']
                                append_performance(curr, date, misclassification_rate)
                            
        # Insert predictions into database
        with p.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.autocommit = True
            # Establish Cursor
            curr = conn.cursor()
            
            # Iterate through each row in the DataFrame
            for row in range(base_df.shape[0]):
                # Get the date, Id, predictions, and quality
                date = base_df[row, 'date']
                Id = base_df[row, 'Id']
                predictions = base_df[row, 'predictions']
                quality = base_df[row, 'quality']
                
                # Insert the row into the database
                append_predictions(curr, date, Id, predictions, quality)
                    
if __name__=='__main__':
    append_files_to_database()