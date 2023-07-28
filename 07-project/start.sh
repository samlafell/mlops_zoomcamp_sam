#!/bin/bash

# The name of the image we're looking for
IMAGE_NAME="wine_quality"

# # Check if the image already exists
# if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" != "" ]]; then
#     echo "Image already exists."
    
#     # Check if Dockerfile has changed
#     if command -v md5sum &> /dev/null; then
#         DOCKERFILE_HASH=$(md5sum Dockerfile | awk '{ print $1 }')
#     else
#         DOCKERFILE_HASH=$(md5 -q Dockerfile)
#     fi
#     if [ -f ".dockerfilehash" ]; then
#         OLD_DOCKERFILE_HASH=$(cat .dockerfilehash)
#         if [ "$DOCKERFILE_HASH" = "$OLD_DOCKERFILE_HASH" ]; then
#             echo "Dockerfile has not changed, skipping build."
#             exit 0
#         else
#             echo "Dockerfile has changed, building image."
#         fi
#     else
#         echo "No record of previous Dockerfile, building image."
#     fi
# else
#     echo "Image does not exist, building image."
# fi

# Build the Docker image
docker build -t $IMAGE_NAME .

# Save the Dockerfile hash for next time
echo $DOCKERFILE_HASH > .dockerfilehash

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
docker-compose up -d

# Give Docker Compose some time to start the services
sleep 20

# Run the curl command
curl -X POST -F file=@$prediction_file_path http://localhost:8000/predict

# Shut down Docker Compose
#docker-compose down