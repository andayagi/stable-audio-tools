FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    PIP_PREFER_BINARY=1 \
    PIP_NO_BUILD_ISOLATION=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first to reduce resolver churn and memory spikes
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.1.0 \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY service ./service
COPY stable_audio_tools ./stable_audio_tools

# Create non-root user and adjust ownership
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]


