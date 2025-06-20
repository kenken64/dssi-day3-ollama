# Text-to-SQL Model Training Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/huggingface

# Install runtime dependencies and Ollama
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create app user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p cache logs outputs models processed_data

# Make scripts executable
RUN chmod +x *.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU Available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "text_to_sql_train.py", "--help"]

# Development stage
FROM production AS development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    pytest \
    black \
    flake8 \
    memory-profiler \
    line-profiler

# Setup Jupyter
RUN mkdir -p /app/notebooks && \
    chown -R appuser:appuser /app/notebooks

# Switch back to app user
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Training stage - optimized for training workloads
FROM production AS training

# Switch to root for optimization
USER root

# Install training optimizations
RUN pip install --no-cache-dir \
    flash-attn \
    deepspeed \
    accelerate

# Switch back to app user
USER appuser

# Training command
CMD ["python", "text_to_sql_train.py", "--mode", "full"]

# Inference stage - minimal for deployment
FROM production AS inference

# Remove training dependencies to reduce size
RUN pip uninstall -y \
    transformers[training] \
    datasets \
    evaluate \
    peft

# Install minimal inference dependencies
RUN pip install --no-cache-dir \
    transformers \
    torch

# Inference command
CMD ["python", "deploy.py", "--demo"]
