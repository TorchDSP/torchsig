FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \ 
    zsh \ 
    ssh \
    rsync \
    libgl1-mesa-glx

ADD torchsig/ /build/torchsig

ADD pyproject.toml /build/pyproject.toml

RUN pip3 install -e /build

RUN pip3 install notebook jupyterlab==4.2.3
RUN pip3 install jupyterlab_theme_solarized_dark
RUN pip3 install ipywidgets

WORKDIR /workspace/code

ADD examples/ /workspace/code/examples
