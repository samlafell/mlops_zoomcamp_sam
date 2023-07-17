# pylint: disable=line-too-long
import json
import pathlib

import requests
from deepdiff import DeepDiff

print('imported packages')

print('setting event')
working_dir = pathlib.Path(__file__).parent
with open(working_dir / 'event.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)

print('making POST request')
URL = 'http://localhost:8080/2015-03-31/functions/function/invocations'
response = requests.post(URL, json=event, timeout=30)

expected_response = {
    'predictions': [
        {
            'model': 'ride_duration_prediction_model',
            'version': 'Test123',
            'prediction': {'ride_duration': 18.16894572640533, 'ride_id': 256},
        }
    ]
}

# print status code and response text
print("Status code:", response.status_code)
print("Response text:", response.text)

# Only try to parse the response if the status code is 200
if response.status_code == 200:
    try:
        actual_response = response.json()
        print(json.dumps(actual_response, indent=2))
        print(actual_response)

        diff = DeepDiff(actual_response, expected_response, significant_digits=1)
        print(f'diff={diff}')
        assert 'type_changes' not in diff
        assert 'values_changed' not in diff

    except json.decoder.JSONDecodeError:
        print(response)
        print("Could not parse response as JSON")
else:
    print("Error making request, status code:", response.status_code)
