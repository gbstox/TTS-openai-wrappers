# CosyVoice TTS Engine Dockerfile
# Optimized build to fit within GitHub Actions disk limits
# CUDA 12.4 runtime with Python 3.10

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
# Use PyTorch 2.4.1 which is compatible with CUDA 12.4
RUN pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install CosyVoice from pip (if available) or minimal clone
WORKDIR /opt

# Clone CosyVoice with sparse checkout (only needed directories)
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git sparse-checkout set cosyvoice third_party requirements.txt && \
    git submodule update --init --depth 1 third_party/Matcha-TTS && \
    rm -rf .git && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/CosyVoice

# Install additional runtime dependencies
# Pin pyyaml to fix Loader.max_depth compatibility issue
RUN pip install --no-cache-dir pydub>=0.25.0 runpod>=1.6.0 pyyaml==6.0.1

# Download CosyVoice3 model from HuggingFace (smallest variant)
RUN mkdir -p pretrained_models && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B', ignore_patterns=['*.md', '*.txt', 'examples/*'])"

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

# Clean up pip cache and temp files
RUN pip cache purge && rm -rf /root/.cache /tmp/*

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
