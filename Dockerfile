FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY backend.py .
COPY index.html .
COPY js/ js/
COPY css/ css/
COPY data/nursing_guide_cleaned.txt data/

RUN apt-get update && \
    apt-get install -y gcc libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Pre-download NLTK punkt and punkt_tab data to the correct location for production
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab

EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]