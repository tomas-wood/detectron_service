FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    cmake \
    git \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    libprotobuf-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-pip

RUN pip install future numpy protobuf

RUN apt-get update && apt-get install -y --no-install-recommends libgflags-dev

COPY ./pytorch /app/pytorch

ENV FULL_CAFFE2=1

RUN cd /app/pytorch && \
    python setup.py install

WORKDIR "/app"


CMD ["/bin/bash"]
