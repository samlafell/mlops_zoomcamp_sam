import requests 
from deepdiff import DeepDiff

event = {
  "Records": [
    {
      "kinesis": {
        "kinesisSchemaVersion": "1.0",
        "partitionKey": "1",
        "sequenceNumber": "49641953411415033205925160724140909591599882595807002626",
        "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
        "approximateArrivalTimestamp": 1687434951.832
      },
      "eventSource": "aws:kinesis",
      "eventVersion": "1.0",
      "eventID": "shardId-000000000000:49641953411415033205925160724140909591599882595807002626",
      "eventName": "aws:kinesis:record",
      "invokeIdentityArn": "arn:aws:iam::640841668094:role/lambda-kinesis-role",
      "awsRegion": "us-east-2",
      "eventSourceARN": "arn:aws:kinesis:us-east-2:640841668094:stream/ride_events"
    }
  ]
}

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=event).json()

import json
print('actual response:')
print(json.dumps(actual_response, indent=2))

expected_response = {
    'predictions': [{
            'model': 'ride_duration_prediction_model',
            'version': 'Test124',
            'prediction': {
                'ride_duration': 18.16894572640533,
                'ride_id': 256   
            }
    }]
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')
assert 'type_changes' not in diff
assert 'values_changed' not in diff