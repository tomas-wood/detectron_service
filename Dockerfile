FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

COPY ./pytorch /app/pytorch

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
    libgflags-dev \
    protobuf-compiler \
    python-dev \
    python-pip \
    nano

# Have to install setuptools and wheel first or it craps out.
RUN pip install setuptools wheel && \
    pip install future numpy protobuf pyyaml typing

# Naming cudnn.h weird shit does not help folks.
RUN cp /usr/include/x86_64-linux-gnu/cudnn_v7.h /usr/include/x86_64-linux-gnu/cudnn.h

# Okay that didnt work we are just going to have to do it ourself with cmake.
RUN mkdir /app/pytorch/build && \
    cd /app/pytorch/build && \
    cmake \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.2  \
    -DCUDA_CUDA_LIB=/usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/libcuda.so \
    -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include \
    -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.7 \
    -DCUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -DUSE_OPENCV=ON \
    .. && \
    make -j$(grep -c ^processor /proc/cpuinfo) install


COPY ./detectron /app/detectron

ENV PYTHONPATH=/app/pytorch/build

RUN pip install hypothesis


WORKDIR "/app"


CMD ["/bin/bash"]
