FROM python:3.12-slim

# Prevents Python from writing .pyc files
# Ensures Python output is not buffered
# UV_COMPILE_BYTECODE=1 -> generate .pyc at install time for faster startup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match

# System deps (opencv runs headless: no GUI/GL libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-is-python3 \
    build-essential \
    cmake \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python*/EXTERNALLY-MANAGED \
    && pip install --no-cache-dir "uv==0.13.6"

WORKDIR /workspace

# Phase 1: heavy deps, cached on pyproject.toml only (torch is not
# re-downloaded when only source changes).
COPY pyproject.toml README.md LICENSE.md ./
RUN uv pip install --system --no-cache \
    $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

# Phase 2: overlay source, cheap editable reinstall (no dep re-download).
COPY . .
RUN uv pip install --system --no-cache -e . --no-deps

CMD ["bash"]
