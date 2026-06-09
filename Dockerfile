# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for builds
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the deployment requirements file first to leverage Docker caching
COPY requirements_deploy.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (including reels_vector.db) into the container
COPY . .

# Expose port 7860 (Hugging Face Spaces default port for Docker containers)
EXPOSE 7860

# Command to run the Chainlit application on port 7860
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "7860"]
