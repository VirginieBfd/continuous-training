# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script to the working directory
COPY hello_world.py .

# Set the entrypoint command to run the Python script
CMD ["python", "hello_world.py"]
