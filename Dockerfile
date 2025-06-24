# Use the official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install system-level dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Download NLTK data
RUN python -m nltk.downloader punkt

# Start the FastAPI app on dynamic port (Fly sets PORT=8080)
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]

