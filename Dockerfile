FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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

RUN pip3 install notebook jupyterlab==4.2.1
RUN pip3 install jupyterlab_theme_solarized_dark
RUN pip3 install ipywidgets

WORKDIR /workspace/code
