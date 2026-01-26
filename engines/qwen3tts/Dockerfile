# Qwen3-TTS Engine Dockerfile
# Optimized for RunPod and Modal serverless deployments
#
# Build: docker build -t qwen3tts -f engines/qwen3tts/Dockerfile .
# Run:   docker run --gpus all -p 8000:8000 qwen3tts

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-c"]

# System dependencies
RUN apt-get update -y --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ffmpeg \
        sox \
        libsox-fmt-all \
        git \
        git-lfs \
        curl \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create model cache directories
ENV HF_HOME=/models
ENV QWEN3TTS_MODEL_DIR=/models/qwen3tts
RUN mkdir -p ${QWEN3TTS_MODEL_DIR}

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    "torch>=2.1.0" \
    "torchaudio>=2.1.0" \
    --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention for faster inference (requires CUDA)
# Using pre-built wheels to avoid long compilation times
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Copy and install engine requirements
WORKDIR /app
COPY engines/qwen3tts/requirements.txt /app/engines/qwen3tts/requirements.txt
RUN pip install --no-cache-dir -r /app/engines/qwen3tts/requirements.txt

# Copy application code
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY config/ /app/config/
COPY engines/__init__.py /app/engines/__init__.py
COPY engines/base.py /app/engines/base.py
COPY engines/registry.py /app/engines/registry.py
COPY engines/qwen3tts/ /app/engines/qwen3tts/
COPY api/ /app/api/
COPY deploy/runpod/ /app/deploy/runpod/

# Pre-download models at build time (optional but recommended for faster cold starts)
# Uncomment the models you want to pre-download:
#
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
#
# # Download tokenizer (required for all models)
# RUN python3 -c "\
# from huggingface_hub import snapshot_download; \
# snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz', \
#     local_dir='/models/qwen3tts/tokenizer', \
#     ignore_patterns=['*.md', '*.txt'])"
#
# # Download 1.7B CustomVoice model (recommended)
# RUN python3 -c "\
# from huggingface_hub import snapshot_download; \
# snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', \
#     local_dir='/models/qwen3tts/1.7b-customvoice', \
#     ignore_patterns=['*.md', '*.txt'])"
#
# # Download 0.6B CustomVoice model (lightweight alternative)
# RUN python3 -c "\
# from huggingface_hub import snapshot_download; \
# snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', \
#     local_dir='/models/qwen3tts/0.6b-customvoice', \
#     ignore_patterns=['*.md', '*.txt'])"
#
# # Download 1.7B VoiceDesign model (for voice design feature)
# RUN python3 -c "\
# from huggingface_hub import snapshot_download; \
# snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', \
#     local_dir='/models/qwen3tts/1.7b-voicedesign', \
#     ignore_patterns=['*.md', '*.txt'])"

# Environment variables
ENV PYTHONPATH=/app
ENV TTS_ENGINE=qwen3tts
ENV QWEN3TTS_MODEL=1.7b-customvoice

# Expose port for API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run the RunPod handler
CMD ["python", "/app/deploy/runpod/handler.py"]
