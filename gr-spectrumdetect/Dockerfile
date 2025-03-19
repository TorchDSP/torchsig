FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    less \
    apt-utils \
    git \
    cmake \
    python3 \
    python3-dev \
    python3-numpy \
    python3-apt \
    pip \
    gcc \
    g++ \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    ffmpeg \
    libharfbuzz-dev \
    libfreetype-dev \
    libonnx-dev \
    protobuf-compiler \
    pybind11-dev \
    vim \
    mlocate \
    cmake-curses-gui \
    libfftw3-dev \ 
    bazel-bootstrap \
    libyaml-cpp-dev \
    libnats-dev \
    rapidjson-dev \
    libboost-all-dev \
    liblog4cpp5-dev \
    sudo \
    net-tools \ 
    iptables

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN pip install pip -U

RUN pip install scikit-learn -U

RUN pip install onnx -U

RUN pip install pybind11 -U

RUN pip install ultralytics -U

RUN pip install torchaudio -U

RUN pip install pybombs

RUN pybombs config makewidth 4

RUN pybombs recipes add gr-recipes https://github.com/gnuradio/gr-recipes.git

RUN pybombs recipes add gr-etcetera https://github.com/gnuradio/gr-etcetera.git

RUN mkdir -p /opt/gnuradio/v3.10

RUN pybombs prefix init /opt/gnuradio/v3.10 -R gnuradio-default

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN source /opt/gnuradio/v3.10/setup_env.sh && \
    uhd_images_downloader 

ADD . /build/gr-spectrumdetect

RUN cd /build/gr-spectrumdetect/ && \
    mkdir -p build && \
    cd build && \
    source /opt/gnuradio/v3.10/setup_env.sh && \
    cmake -D CMAKE_INSTALL_PREFIX=/opt/gnuradio/v3.10/ ../ && \
    make && \
    make install && \ 
    ldconfig

RUN cd /build/ && \
    git clone https://github.com/TorchDSP/torchsig.git && \
    cd torchsig/ && \
    pip install -e .

RUN pip uninstall opencv-python -y

RUN pip install opencv-python-headless

WORKDIR /workspace
