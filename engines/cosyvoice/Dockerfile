# CosyVoice TTS Engine Dockerfile
# Uses PyTorch base image (has PyTorch + CUDA pre-installed = faster build)

# Use PyTorch 2.5.1 for transformers compatibility (register_fake API)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including build tools for pyworld)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Clone CosyVoice (code only, no model)
WORKDIR /opt
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    rm -rf .git examples asset docker runtime tools

WORKDIR /opt/CosyVoice

# Install CosyVoice requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir pydub>=0.25.0 runpod>=1.6.0 huggingface_hub

# Create model directory
RUN mkdir -p pretrained_models

# Set working directory for the app
WORKDIR /app

# Copy application code
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ /app/config/
COPY engines/__init__.py /app/engines/__init__.py
COPY engines/base.py /app/engines/base.py
COPY engines/registry.py /app/engines/registry.py
COPY engines/cosyvoice/ /app/engines/cosyvoice/
COPY api/ /app/api/
COPY deploy/runpod/ /app/deploy/runpod/

# CRITICAL: Force upgrade pyyaml LAST to fix HyperPyYAML max_depth compatibility
# Some CosyVoice deps install old pyyaml, this ensures we have 6.0.2+
RUN pip install --no-cache-dir --force-reinstall --upgrade "pyyaml>=6.0.2" && \
    python -c "import yaml; print('PyYAML version:', yaml.__version__)"

# Clean up
RUN pip cache purge && rm -rf /root/.cache /tmp/*

# Environment variables
ENV PYTHONPATH=/app:/opt/CosyVoice
ENV PYTHONUNBUFFERED=1
ENV TTS_ENGINE=cosyvoice
ENV COSYVOICE_PATH=/opt/CosyVoice
ENV COSYVOICE_MODEL_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
ENV HF_HOME=/opt/CosyVoice/pretrained_models/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "/app/deploy/runpod/handler.py"]
