# CosyVoice TTS Engine Dockerfile
# Downloads model at runtime instead of baking into image
# CUDA 12.4 devel with Python 3.10

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Initialize git-lfs
RUN git lfs install

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch first (required by CosyVoice)
RUN pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Clone CosyVoice code only (no model download during build)
WORKDIR /opt
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git sparse-checkout set cosyvoice third_party requirements.txt && \
    git submodule update --init --depth 1 third_party/Matcha-TTS && \
    rm -rf .git && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/CosyVoice

# Install additional dependencies
RUN pip install --no-cache-dir pydub>=0.25.0 runpod>=1.6.0 huggingface_hub

# Force reinstall pyyaml for HyperPyYAML compatibility
RUN pip install --no-cache-dir --force-reinstall "pyyaml>=6.0.2"

# Create model directory (will be populated at runtime)
RUN mkdir -p pretrained_models

# Set working directory for the app
WORKDIR /app

# Copy shared requirements
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

# Clean up
RUN pip cache purge && rm -rf /root/.cache /tmp/*

# Environment variables
ENV PYTHONPATH=/app:/opt/CosyVoice
ENV PYTHONUNBUFFERED=1
ENV TTS_ENGINE=cosyvoice
ENV COSYVOICE_PATH=/opt/CosyVoice
ENV COSYVOICE_MODEL_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
# HuggingFace cache for model downloads (can be mounted to volume)
ENV HF_HOME=/opt/CosyVoice/pretrained_models/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "/app/deploy/runpod/handler.py"]
