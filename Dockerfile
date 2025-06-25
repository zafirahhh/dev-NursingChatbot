# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables to prevent .pyc files and enable clean logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files into the container
COPY . .

# Install system dependencies for nltk and sentence-transformers
RUN apt-get update && \
    apt-get install -y gcc git libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Pre-download NLTK punkt data
RUN python -m nltk.downloader punkt

EXPOSE 8080

# Run the FastAPI server using Uvicorn when the container starts
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
