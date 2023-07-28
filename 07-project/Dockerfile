# Use the official lightweight Python image with Python 3.8.
# For the Linux base image, Alpine is used
FROM python:3.10-bullseye

# Set the working directory in the Docker image
WORKDIR /app

# Add current directory contents to the container at /app
COPY src/ /app/src
COPY tests/ /app/tests
COPY config/ /app/config
COPY data/ /app/data
COPY .vscode/ /app/.vscode
COPY utils/ /app/utils
COPY *.py /app/
COPY Makefile /app/
COPY columns.json /app/columns.json

# Make sure pip is up to date
RUN pip install -U pip
RUN pip install -U pipenv

# Copy over the Pipfile and Pipfile.lock files
COPY Pipfile Pipfile.lock ./

# Install the dependencies
RUN pipenv install --system --deploy