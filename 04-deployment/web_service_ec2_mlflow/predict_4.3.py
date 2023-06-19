import os
from flask import Flask, request, jsonify
import mlflow

# Initialize MLFlow stuff
# Need to EXPORT RUN_ID in the terminal
# Services running are: EC2, S3, MLFlow, Flask
RUN_ID = os.getenv('RUN_ID')
EXP_ID = '1'
logged_model = f's3://mlops-week4/{EXP_ID}/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    return {
        'PU_DO': f"""{ride['PULocationID']}_{ride["DOLocationID"]}""",
        'trip_distance': ride['trip_distance'],
    }
    
def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')
# Decorator to create an endpoint
@app.route('/predict', methods=["POST"])
def predict_endpoint():
    ride = request.get_json()    
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration':pred,
        'model_version':RUN_ID
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)