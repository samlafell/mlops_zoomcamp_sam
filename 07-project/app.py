import logging
import os
import tempfile
from datetime import datetime
from io import StringIO

import polars as pl
import s3fs
from flask import Flask, jsonify, request

from utils.load_model import ModelService, standardize_data

app = Flask(__name__)

import mlflow

# Define S3 file system
s3_file_system = s3fs.S3FileSystem()

# Define MLFlow Tracking Server
mlflow.set_tracking_uri("http://mlflow-server:5001")

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
    
model_service = ModelService("BestWineDatasetModel")
model = model_service.get_model_version()
columns = model.get_columns()
model = model.load_model().model

def temp_file_save(s3_file_system, group_write, date, s3_bucket, s3_path):
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


@app.route("/predict", methods=["POST"])
def predict():
    csv_file = request.files.get("file")
    if not csv_file:
        return "No file uploaded.", 400

    # Load DF
    df = pl.read_csv(csv_file)
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%dT%H:%M:%S.%f"))

    # Check columns
    df_model_vars = df.select(columns.original_columns)
    model_service.check_columns(df_model_vars)

    # Define the S3 bucket and path where you will save the results
    s3_bucket = "sal-wine-quality"
    s3_path = f"models/{model_service.model_name}/preds/"

    # Group by 'date' and process each group
    for date in df.select("date").unique().to_series().to_list():
        group = df.filter(pl.col("date") == date)
        model_df = group.select(columns.original_columns)

        # Grab necessary columns, apply z-score scaling
        df_std = standardize_data(model_df)
        predictions = model.predict(df_std)
        logging.debug("made predictions for date: {date}")

        # Add predictions to DataFrame
        group = group.with_columns(pl.Series(predictions).alias("predictions"))
        group_write = group.select("Id", "date", "predictions", "quality")

        temp_file_save(s3_file_system, group_write, date, s3_bucket, s3_path)

    return jsonify({"status": "success"}), 200


if __name__ == "__main__":
    # Run application
    app.run(host="0.0.0.0", port=8000)
