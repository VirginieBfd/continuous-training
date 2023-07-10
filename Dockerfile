# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script to the working directory
COPY . .

# Set the entrypoint command to run the Python script
CMD ["python", "github_action_hello_world/hello_world.py"]
