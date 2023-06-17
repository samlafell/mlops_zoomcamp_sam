import pickle
import pathlib
from flask import Flask, request, jsonify

# Need to load in the model we've created

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


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


# Shutdown Server
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)