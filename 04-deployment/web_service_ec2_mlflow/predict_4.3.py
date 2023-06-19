import pickle
import pathlib
from flask import Flask, request, jsonify
import mlflow

# Initialize MLFlow stuff
from mlflow.tracking import MlflowClient
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5001'
RUN_ID = 'e4cb04823f7447ea93e9bf0b848778e5'

# Set MLFlow Tracking Info
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("green-taxi-duration")

# Logged Model
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

# Have to initialize the Client to be able to download artifacts from MLFlow
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Get the location of the artifiact
path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
print(f'downloading the dict vectorizer to {path}')

# Download the dict vectorizer from MLFlow artifact Store
with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

    
def prepare_features(ride):
    return {
        'PU_DO': f"""{ride['PULocationID']}_{ride["DOLocationID"]}""",
        'trip_distance': ride['trip_distance'],
    }

    
def predict(features):
    X = dv.transform(features)
    return model.predict(X)[0]


app = Flask('duration-prediction')
# Decorator to create an endpoint
@app.route('/predict', methods=["POST"])
def predict_endpoint():
    ride = request.get_json()    
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration':pred
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)