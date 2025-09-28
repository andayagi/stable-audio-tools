FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PYTHONPATH=/app \
    PIP_PREFER_BINARY=1 \
    PIP_NO_BUILD_ISOLATION=1

WORKDIR /app

# Install system dependencies including build tools for flash-attn
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first to reduce resolver churn and memory spikes
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout=1800 torch>=2.2.0

# Install other dependencies, with flash-attn as optional
RUN pip install --no-cache-dir --timeout=1800 \
    fastapi==0.114.2 \
    uvicorn[standard]==0.31.1 \
    numpy>=1.23.0,<1.27.0 \
    safetensors>=0.4.0 \
    huggingface_hub>=0.25.0 \
    k-diffusion>=0.1.0 \
    tqdm>=4.66.0 \
    alias-free-torch \
    auraloss \
    descript-audio-codec \
    einops \
    einops-exts \
    ema-pytorch \
    encodec \
    local-attention \
    prefigure

# Try to install flash-attn, but don't fail if it doesn't work
RUN pip install --no-cache-dir --timeout=1800 flash-attn>=2.5.0 || echo "flash-attn installation failed, continuing without it"

COPY app ./app
COPY stable_audio_tools ./stable_audio_tools

# Create non-root user and adjust ownership
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]


