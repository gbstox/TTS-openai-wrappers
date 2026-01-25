# CosyVoice TTS Engine Dockerfile
# Based on official CosyVoice Dockerfile for tested dependencies

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# Install system dependencies (matching official Dockerfile)
RUN apt-get update -y --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev ca-certificates && \
    apt-get clean && \
    git lfs install

# Install miniforge (conda) - same as official
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Create conda environment
RUN conda create -y -n cosyvoice python=3.10
ENV CONDA_DEFAULT_ENV=cosyvoice
ENV PATH=/opt/conda/bin:/opt/conda/envs/cosyvoice/bin:$PATH

WORKDIR /opt

# Clone CosyVoice
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# Install pynini via conda (required, can't pip install easily)
RUN conda activate cosyvoice && conda install -y -c conda-forge pynini==2.1.5

# Install CosyVoice requirements
RUN conda activate cosyvoice && cd CosyVoice && \
    pip install -r requirements.txt --no-cache-dir

# CRITICAL: Force upgrade pyyaml to fix HyperPyYAML max_depth compatibility
# CosyVoice requirements install old pyyaml that causes "'Loader' object has no attribute 'max_depth'"
RUN conda activate cosyvoice && \
    pip install --no-cache-dir --force-reinstall --upgrade "pyyaml>=6.0.2" && \
    python -c "import yaml; print('PyYAML version:', yaml.__version__)"

# Install additional dependencies for our wrapper
RUN conda activate cosyvoice && \
    pip install --no-cache-dir pydub runpod huggingface_hub

# Create model directory (will download at runtime)
RUN mkdir -p /opt/CosyVoice/pretrained_models

# Set up app directory
WORKDIR /app

# Copy our wrapper code
COPY requirements.txt /app/requirements.txt
RUN conda activate cosyvoice && pip install --no-cache-dir -r requirements.txt

COPY config/ /app/config/
COPY engines/__init__.py /app/engines/__init__.py
COPY engines/base.py /app/engines/base.py
COPY engines/registry.py /app/engines/registry.py
COPY engines/cosyvoice/ /app/engines/cosyvoice/
COPY api/ /app/api/
COPY deploy/runpod/ /app/deploy/runpod/

# Environment variables
ENV PYTHONPATH=/app:/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS
ENV TTS_ENGINE=cosyvoice
ENV COSYVOICE_PATH=/opt/CosyVoice
ENV COSYVOICE_MODEL_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use conda environment for runtime
CMD ["conda", "run", "-n", "cosyvoice", "python", "/app/deploy/runpod/handler.py"]
