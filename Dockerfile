# Use the official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Run the FastAPI app using backend.py (dynamic port support)
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
