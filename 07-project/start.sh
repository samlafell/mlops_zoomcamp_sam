#!/bin/bash

# The name of the image we're looking for
IMAGE_NAME="wine_quality"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Prompt the user to enter the path to the prediction file
echo "Please enter the path to the prediction file:"
read prediction_file_path

# Check if the file exists
if [ ! -f "$prediction_file_path" ]; then
    echo "File not found! Please make sure you've entered the correct path."
    exit 1
fi

# Export the prediction file path as an environment variable
export PREDICTION_FILE=$prediction_file_path

# Run Docker Compose
docker-compose up --build -d

# Give Docker Compose some time to start the services
sleep 20

# Run the curl command
curl -X POST -F file=@$prediction_file_path http://localhost:8000/predict

# Sleep
sleep 60
echo "Sleeping before moving on to eval.py to move S3 files to postgres db."

# To DB
pipenv run python eval.py

# Shut down Docker Compose
#docker-compose down