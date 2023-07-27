# Use the official lightweight Python image with Python 3.8.
# For the Linux base image, Alpine is used
FROM python:3.10-bullseye

# Set the working directory in the Docker image
WORKDIR /app

# Add current directory contents to the container at /app
ADD . /app

# Install gcc and python3-dev
#RUN apt-get update && apt-get install -y gcc python3-dev

# Make sure pip is up to date
RUN pip install -U pip
RUN pip install -U pipenv

# Installing a necessary build step
#RUN apt-get install build-essential -y

# Copy over the Pipfile and Pipfile.lock files
COPY Pipfile Pipfile.lock ./

# Install the dependencies
RUN pipenv install --system --deploy

# Set
ENV MLFLOW_TRACKING_URI=s3://sal-wine-quality/models/

# # Copy src/, data/, notebooks/, Makefile from your local system into the Docker image
# COPY src/ src/
# COPY data/ data/
# COPY notebooks/ notebooks/
# COPY Makefile .

# Run the command to prepare data
# CMD ["make", "data"]