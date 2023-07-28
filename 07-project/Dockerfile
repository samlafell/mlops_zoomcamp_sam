# Use the official lightweight Python image with Python 3.8.
# For the Linux base image, Alpine is used
FROM python:3.10-bullseye

# Set the working directory in the Docker image
WORKDIR /app

# Add current directory contents to the container at /app
ADD . /app

# Make sure pip is up to date
RUN pip install -U pip
RUN pip install -U pipenv

# Copy over the Pipfile and Pipfile.lock files
COPY Pipfile Pipfile.lock ./

# Install the dependencies
RUN pipenv install --system --deploy

# Make port 8080 available for the app
EXPOSE 8000
EXPOSE 5001

# Train
CMD ["./start.sh"]