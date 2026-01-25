# Kokoro TTS Engine Dockerfile
# CUDA 12.4 base with Python 3.11

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including build tools for pyopenjtalk
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    espeak-ng \
    libespeak-ng1 \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy shared requirements first (for layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy engine-specific requirements
COPY engines/kokoro/requirements.txt /app/engines/kokoro/requirements.txt
RUN pip install --no-cache-dir -r engines/kokoro/requirements.txt

# Download UniDic dictionary for Japanese language support (MeCab/fugashi)
# This downloads the full UniDic dictionary (~500MB) required for Japanese TTS
RUN python -m unidic download

# Copy application code
COPY config/ /app/config/
COPY engines/__init__.py /app/engines/__init__.py
COPY engines/base.py /app/engines/base.py
COPY engines/registry.py /app/engines/registry.py
COPY engines/kokoro/ /app/engines/kokoro/
COPY api/ /app/api/
COPY deploy/runpod/ /app/deploy/runpod/

# Pre-download the Kokoro model during build (optional but reduces cold start)
# This downloads the model weights from HuggingFace
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')" || true

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TTS_ENGINE=kokoro
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the RunPod serverless handler
# For local API testing, override with: docker run ... python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
CMD ["python", "/app/deploy/runpod/handler.py"]

