# Description: Dockerfile for the Python 3.10-slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt update && apt install -y gcc && apt install awscli -y

# Install the required packages
RUN apt-get update && pip install --upgrade pip && pip install -r requirements.txt

# Make port 80 available to the world outside this container
CMD ['python3', 'app.py']
