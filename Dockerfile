FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    zsh \
    ssh \
    rsync \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    build-essential \
    gcc \
    ca-certificates \
    curl \
    && apt-get clean

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -fsSL https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit || true

RUN pip3 install --upgrade pip
RUN if lspci | grep -i nvidia; then \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY torchsig/ /build/torchsig
COPY pyproject.toml /build/pyproject.toml

RUN pip3 install -e /build

RUN pip3 install notebook jupyterlab==4.2.3
RUN pip3 install jupyterlab_theme_solarized_dark ipywidgets

WORKDIR /workspace/code

ADD examples/ /workspace/code/examples
