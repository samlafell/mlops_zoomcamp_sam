import requests 

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
response = requests.post(url, json=event)
if response.text:
    print(response.json())
else:
    print("Empty response")