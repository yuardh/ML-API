# Use the official Python image as a base image
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONBUFFERED True

# Copy requirements.txt to the docker image and install packages
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app