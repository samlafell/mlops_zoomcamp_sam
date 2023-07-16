# pylint: disable=line-too-long
import json

import requests
from deepdiff import DeepDiff

print('imported packages')

print('setting event')
with open('event.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)

print('making POST request')
URL = 'http://localhost:8080/2015-03-31/functions/function/invocations'
response = requests.post(URL, json=event, timeout=30)
response.raise_for_status()  # This will raise an exception if the request failed
actual_response = response.json()

print('actual response:')
print(json.dumps(actual_response, indent=2))

expected_response = {
    'predictions': [
        {
            'model': 'ride_duration_prediction_model',
            'version': 'Test123',
            'prediction': {'ride_duration': 18.16894572640533, 'ride_id': 256},
        }
    ]
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')
assert 'type_changes' not in diff
assert 'values_changed' not in diff
