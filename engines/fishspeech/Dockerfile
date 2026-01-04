# Fish Speech (OpenAudio S1) TTS Engine Dockerfile
# Based on official Fish Speech Docker approach with uv package manager

# Use NVIDIA CUDA base with Python
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies + Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Initialize git-lfs
RUN git lfs install

# Clone Fish Speech
WORKDIR /opt
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git

WORKDIR /opt/fish-speech

# Create virtual environment and install Fish Speech with uv
RUN uv venv .venv --python 3.12 && \
    . .venv/bin/activate && \
    uv pip install -e ".[stable]" && \
    uv pip install httpx pydub runpod

# Download OpenAudio S1-mini model from HuggingFace
RUN mkdir -p checkpoints && \
    . .venv/bin/activate && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini', ignore_patterns=['*.md'])"

# Set up application directory
WORKDIR /app

# Copy our wrapper code
COPY requirements.txt /app/requirements.txt
RUN . /opt/fish-speech/.venv/bin/activate && uv pip install -r requirements.txt

COPY config/ /app/config/
COPY engines/__init__.py /app/engines/__init__.py
COPY engines/base.py /app/engines/base.py
COPY engines/registry.py /app/engines/registry.py
COPY engines/fishspeech/ /app/engines/fishspeech/
COPY api/ /app/api/
COPY deploy/runpod/ /app/deploy/runpod/

# Create startup script that runs both Fish Speech API and RunPod handler
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    '' \
    '# Activate virtual environment' \
    'source /opt/fish-speech/.venv/bin/activate' \
    '' \
    '# Set paths' \
    'export PYTHONPATH="/app:/opt/fish-speech:$PYTHONPATH"' \
    'export LLAMA_CHECKPOINT_PATH="/opt/fish-speech/checkpoints/openaudio-s1-mini"' \
    'export DECODER_CHECKPOINT_PATH="/opt/fish-speech/checkpoints/openaudio-s1-mini/codec.pth"' \
    'export DECODER_CONFIG_NAME="modded_dac_vq"' \
    '' \
    '# Start Fish Speech API server in background' \
    'echo "Starting Fish Speech API server..."' \
    'cd /opt/fish-speech' \
    'python tools/api_server.py \' \
    '  --listen "127.0.0.1:8080" \' \
    '  --llama-checkpoint-path "$LLAMA_CHECKPOINT_PATH" \' \
    '  --decoder-checkpoint-path "$DECODER_CHECKPOINT_PATH" \' \
    '  --decoder-config-name "$DECODER_CONFIG_NAME" \' \
    '  --device cuda &' \
    '' \
    '# Wait for API server to be ready' \
    'echo "Waiting for Fish Speech API to be ready..."' \
    'for i in {1..60}; do' \
    '  if curl -s http://127.0.0.1:8080/v1/health > /dev/null 2>&1; then' \
    '    echo "Fish Speech API is ready!"' \
    '    break' \
    '  fi' \
    '  sleep 2' \
    'done' \
    '' \
    '# Start RunPod handler' \
    'echo "Starting RunPod handler..."' \
    'cd /app' \
    'exec python /app/deploy/runpod/handler.py' \
    > /app/start.sh && chmod +x /app/start.sh

# Environment variables
ENV PYTHONPATH=/app:/opt/fish-speech
ENV TTS_ENGINE=fishspeech
ENV FISH_API_URL=http://127.0.0.1:8080

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

CMD ["/app/start.sh"]

