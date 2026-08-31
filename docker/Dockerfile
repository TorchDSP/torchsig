FROM python:3.10-slim

# Prevents Python from writing .pyc files
# Ensures Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-is-python3 \
    build-essential \
    cmake \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python*/EXTERNALLY-MANAGED

WORKDIR /workspace

COPY . .


# Replace opencv-python with opencv-python-headless
RUN sed -i 's/opencv-python/opencv-python-headless/g' pyproject.toml

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -e .

CMD ["bash"]
