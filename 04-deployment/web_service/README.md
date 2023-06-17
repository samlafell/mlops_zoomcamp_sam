#  Deploying a model as a web-service

- Creating a virtual environment with Pipenv
- Creating a script for predictiong
- Putting the script into a Flask app
- Packaging the app to Docker

```docker build -t ride-duration-prediction-service:v1 .```

```docker run -it --rm -p 9696:9696  ride-duration-prediction-service:v1```