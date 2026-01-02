# CosyVoice TTS Engine Dockerfile
# Multi-stage build to reduce image size
# CUDA 12.4 runtime with Python 3.10 (CosyVoice requirement)

# Stage 1: Build and download dependencies
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Initialize git-lfs
RUN git lfs install

WORKDIR /opt

# Clone CosyVoice repository (shallow clone to save space)
RUN git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    rm -rf .git third_party/*/.git

WORKDIR /opt/CosyVoice

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pydub>=0.25.0 runpod>=1.6.0

# Download CosyVoice3 model from HuggingFace
RUN mkdir -p pretrained_models && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"

# Stage 2: Runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages

# Copy CosyVoice code and model
COPY --from=builder /opt/CosyVoice /opt/CosyVoice

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
CMD ["python", "/app/deploy/runpod/handler.py"]
