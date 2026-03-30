# +----------------------------------------------------------------------------+
# |                        Stage 0: uv binaries                               |
# +----------------------------------------------------------------------------+
FROM ghcr.io/astral-sh/uv:0.11.1 AS uv

# +----------------------------------------------------------------------------+
# |                       Stage 1: Builder (with CUDA Toolkit)                |
# +----------------------------------------------------------------------------+
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

# +----------------------------------------------------------------------------+
# | Install system build tools, Git, and native dependencies                  |
# +----------------------------------------------------------------------------+
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      libffi-dev \
      libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# +----------------------------------------------------------------------------+
# | Install rustup and set up latest stable Rust (>=1.81)                     |
# +----------------------------------------------------------------------------+
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable && \
    rustc --version && \
    uv python install 3.12

# +----------------------------------------------------------------------------+
# | Copy dependency metadata first for better layer reuse                      |
# +----------------------------------------------------------------------------+
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src ./src
COPY rust ./rust

# +----------------------------------------------------------------------------+
# | Sync runtime environment with uv                                           |
# +----------------------------------------------------------------------------+
RUN uv sync --locked --no-dev --python 3.12

# +----------------------------------------------------------------------------+
# | Copy the full project into the builder image                               |
# +----------------------------------------------------------------------------+
COPY . .

# +----------------------------------------------------------------------------+
# | Sync the project with source present                                       |
# +----------------------------------------------------------------------------+
RUN uv sync --locked --no-dev --python 3.12

# +============================================================================+
# |                Stage 2: Runtime (CUDA Runtime Only)                       |
# +============================================================================+
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=uv /uv /uvx /bin/
WORKDIR /workspace

# +----------------------------------------------------------------------------+
# | Install minimal system libraries required at runtime (e.g., OpenCV deps)   |
# +----------------------------------------------------------------------------+
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libgl1 \
      libsm6 \
      libxrender1 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    PATH=/workspace/.venv/bin:/bin:$PATH

# +----------------------------------------------------------------------------+
# | Copy managed Python and project environment from builder                   |
# +----------------------------------------------------------------------------+
COPY --from=builder /python /python
COPY --from=builder /workspace/.venv /workspace/.venv

# +----------------------------------------------------------------------------+
# | (Optional) Copy source/tests/scripts if you need them in the runtime       |
# +----------------------------------------------------------------------------+
COPY --from=builder /workspace /workspace

# +----------------------------------------------------------------------------+
# | Default to bash for interactive GPU testing                                |
# +----------------------------------------------------------------------------+
CMD ["bash"]
