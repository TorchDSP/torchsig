# +----------------------------------------------------------------------------+
# |                       Stage 1: Builder (with CUDA Toolkit)                |
# +----------------------------------------------------------------------------+
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
WORKDIR /workspace

# +----------------------------------------------------------------------------+
# | Install system build tools, Python headers, pip and Git                   |
# +----------------------------------------------------------------------------+
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \          
      curl \                    
      libffi-dev libssl-dev \      
      python3.10 python3-pip \      
      git && \                       
    rm -rf /var/lib/apt/lists/*     

# +----------------------------------------------------------------------------+
# | Install rustup and set up latest stable Rust (>=1.81)                     |
# +----------------------------------------------------------------------------+
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable \
    && rustc --version           

# +----------------------------------------------------------------------------+
# | Upgrade pip and install Python packaging tools (setuptools-rust, wheel)    |
# +----------------------------------------------------------------------------+
RUN python3 -m pip install --upgrade pip setuptools setuptools-rust wheel

# +----------------------------------------------------------------------------+
# | Copy the entire project into the builder image                             |
# +----------------------------------------------------------------------------+
COPY . .

# +----------------------------------------------------------------------------+
# | Install TorchSig (builds the Rust extension in-place)                     |
# +----------------------------------------------------------------------------+
RUN pip install . --no-cache-dir

# +============================================================================+
# |                Stage 2: Runtime (CUDA Runtime Only)                       |
# +============================================================================+
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
WORKDIR /workspace

# +----------------------------------------------------------------------------+
# | Install minimal system libraries required at runtime (e.g., OpenCV deps)   |
# +----------------------------------------------------------------------------+
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \   
      libsm6 \   
      libxrender1 \
      libxext6 && \ 
    rm -rf /var/lib/apt/lists/*

# +----------------------------------------------------------------------------+
# | Copy installed Python packages from builder into runtime image             |
# +----------------------------------------------------------------------------+
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ \
                     /usr/local/lib/python3.10/dist-packages/

# +----------------------------------------------------------------------------+
# | (Optional) Copy source/tests/scripts if you need them in the runtime       |
# +----------------------------------------------------------------------------+
COPY --from=builder /workspace /workspace

# +----------------------------------------------------------------------------+
# | Default to bash for interactive GPU testing                                |
# +----------------------------------------------------------------------------+
CMD ["bash"]
