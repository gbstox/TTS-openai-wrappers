# CosyVoice TTS Engine Dockerfile
# CUDA 12.4 base with Python 3.10 (CosyVoice requirement)

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /opt

# Clone CosyVoice repository
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /opt/CosyVoice

# Install CosyVoice dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    pydub>=0.25.0 \
    runpod>=1.6.0

# Download CosyVoice3 model from HuggingFace
RUN mkdir -p pretrained_models && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"

# Set working directory for the app
WORKDIR /app

# Copy shared requirements first (for layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ /app/config/
COPY engines/__init__.py /app/engines/__init__.py
COPY engines/base.py /app/engines/base.py
COPY engines/registry.py /app/engines/registry.py
COPY engines/cosyvoice/ /app/engines/cosyvoice/
COPY api/ /app/api/
COPY deploy/runpod/ /app/deploy/runpod/

# Environment variables
ENV PYTHONPATH=/app:/opt/CosyVoice
ENV PYTHONUNBUFFERED=1
ENV TTS_ENGINE=cosyvoice
ENV COSYVOICE_PATH=/opt/CosyVoice
ENV COSYVOICE_MODEL_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the RunPod serverless handler
# For local API testing, override with: docker run ... python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
CMD ["python", "/app/deploy/runpod/handler.py"]

