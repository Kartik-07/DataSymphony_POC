# Dockerfile
# Use an official Python 3.12 runtime as the parent image
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=True

# Set the working directory inside the container
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# Using --no-cache-dir makes the image slightly smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container
COPY main.py .

# Command to run the Uvicorn server when the container starts
# Listens on all interfaces (0.0.0.0) on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]