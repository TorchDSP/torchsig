FROM python:3.10-slim

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is not buffered
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Create and activate virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Copy torchsig
COPY ../. .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libffi-dev libssl-dev \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install build setuptools

# Install torchsig
RUN pip install --no-cache-dir .

CMD ["bash"]
