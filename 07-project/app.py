from flask import Flask, request, jsonify
from utils.load_model import ModelService, standardize_data
import logging
import polars as pl

app = Flask(__name__)

model_service = ModelService("BestWineDatasetModel")
model = model_service.get_model_version()
columns = model.get_columns()
model = model.load_model().model

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

@app.route("/predict", methods=["POST"])
def predict():
    csv_file = request.files.get("file")
    if not csv_file:
        return "No file uploaded.", 400

    # Load DF
    df = pl.read_csv(csv_file)
    df = df.select(columns.original_columns)
    # Check columns
    model_service.check_columns(df)

    # Grab necessary columns, apply z-score scaling
    df_std = standardize_data(df, id_col="Id")
    predictions = model.predict(df_std)

    predictions = [float(value) for value in predictions]
    return jsonify({"ID": list(df["Id"]), "predictions": predictions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)