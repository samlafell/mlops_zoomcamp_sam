import logging
from datetime import datetime
from io import StringIO
import s3fs
import os
import tempfile
import boto3
import polars as pl
from flask import Flask, jsonify, request
import psycopg
from utils.load_model import ModelService, standardize_data

app = Flask(__name__)

model_service = ModelService("BestWineDatasetModel")
model = model_service.get_model_version()
columns = model.get_columns()
model = model.load_model().model

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

create_table_statement = """
drop table if exists model_performance;
create table model_performance(
	date timestamp,
	val_mlogloss float,
	avg_prediction float
)
"""


def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
            logger.info("Created database test")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)


def append_performance(curr, date, val_mlogloss, avg_prediction):
    curr.execute(
        "insert into model_performance(date, val_mlogloss, avg_prediction) values (%s, %s, %s)",
        (date, val_mlogloss, avg_prediction),
    )
    logger.info(f"Inserted performance for {date.strftime('%Y%m%d')}")


@app.route("/predict", methods=["POST"])
def predict():
    csv_file = request.files.get("file")
    csv_file2 = request.files.get("file2")
    if not csv_file:
        return "No file uploaded.", 400

    # Load DF
    df = pl.read_csv(csv_file)
    labels = pl.read_csv(csv_file2)
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%dT%H:%M:%S.%f"))

    # Check columns
    df_model_vars = df.select(columns.original_columns)
    model_service.check_columns(df_model_vars)

    # Define the S3 bucket and path where you will save the results
    s3_bucket = "sal-wine-quality"
    s3_path = f"models/{model_service.model_name}/preds/"

    # Define S3 file system
    s3_file_system = s3fs.S3FileSystem()

    # Group by 'date' and process each group
    for date in df.select("date").unique().to_series().to_list():
        group = df.filter(pl.col("date") == date)
        model_df = group.select(columns.original_columns)

        # Grab necessary columns, apply z-score scaling
        df_std = standardize_data(model_df)
        predictions = model.predict(df_std)

        # Add predictions to DataFrame
        group = group.with_columns(pl.Series(predictions).alias("predictions"))
        group_write = group.select("Id", "date", "predictions", "quality")

        # Save to a local temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            group_write.write_csv(temp.name)

        # Then upload the temporary file to S3
        s3_file_system.put(
            temp.name, f'{s3_bucket}/{s3_path}{date.strftime("%Y%m%d")}/predictions.csv'
        )
        logger.info(f"Uploaded predictions for {date.strftime('%Y%m%d')} to S3")

        # Delete the local temporary file
        os.remove(temp.name)

        # calculate val-mlogloss
        val_mlogloss = group.select("quality", "predictions").to_pandas()
        val_mlogloss = val_mlogloss["quality"] - val_mlogloss["predictions"]
        val_mlogloss = val_mlogloss.abs().mean()
        logger.info(f"val-mlogloss for {date.strftime('%Y%m%d')} is {val_mlogloss}")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example",
            autocommit=True,
        ) as conn:
            with conn.cursor() as curr:
                append_performance(curr, date, val_mlogloss, predictions.mean())

    return jsonify({"status": "success"}), 200


if __name__ == "__main__":
    prep_db()
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    app.run(host="0.0.0.0", port=8000, debug=True)
