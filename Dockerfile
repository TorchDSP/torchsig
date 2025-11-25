# +----------------------------------------------------------------------------+
# |                                 Stage 1: Builder                           |
# +----------------------------------------------------------------------------+
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder
WORKDIR /workspace

# +----------------------------------------------------------------------------+
# | Install packages and python                                                |
# +----------------------------------------------------------------------------+
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential curl \
      libffi-dev libssl-dev \
      libgl1 libsm6 \
      libxrender1 \
      libxext6 git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV UV_MANAGED_PYTHON=1
ENV UV_COMPILE_BYTECODE=1
ARG CACHE_DIR=/opt/cache
ENV UV_CACHE_DIR=${CACHE_DIR}/uv/cache/
ENV UV_PYTHON_CACHE_DIR=${CACHE_DIR}/uv/python

# +----------------------------------------------------------------------------+
# | Install rustup (uv will bootstrap)                                         |
# +----------------------------------------------------------------------------+
# ENV RUSTUP_HOME=/usr/local/rustup \
#     CARGO_HOME=/usr/local/cargo \
#     PATH=/usr/local/cargo/bin:$PATH
#
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable \
#     && rustc --version

# +----------------------------------------------------------------------------+
# | Install Python package dependencies                                        |
# +----------------------------------------------------------------------------+
RUN --mount=type=cache,target=${CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project --no-editable


# +============================================================================+
# |                Stage 2: Runtime                                           |
# +============================================================================+
FROM builder AS app
WORKDIR /workspace

COPY . .

RUN --mount=type=cache,target=${CACHE_DIR} \
    uv sync --locked --no-editable

CMD ["bash"]
