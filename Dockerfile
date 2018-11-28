FROM ubuntu:16.04

COPY . /app

ENV DETECTRON_HOME=/app/detectron
ENV PYTHONPATH=/app/pytorch/build
ENV CAFFE2_HOME=/app/pytorch/build



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
    nano  && \
    pip install setuptools wheel && \
    pip install future numpy protobuf pyyaml typing && \
    mkdir /app/pytorch/build && \
    cd /app/pytorch/build && \
    cmake -DUSE_CUDA=0 .. && \
    make -j$(grep -c ^processor /proc/cpuinfo) install && \
    pip install hypothesis && \
    pip install cython && \
    pip install urllib3 \
    opencv-python \
    pycocotools \
    scipy \
    matplotlib==2.2.3 \
    pillow \
    flask \
    flask_restful && \
    cd $DETECTRON_HOME && \
    make && \
    ln -s /usr/local/lib/libcaffe2.so /usr/lib/libcaffe2.so

#WORKDIR "/app"

#ENTRYPOINT ["python"]

#CMD ["server.py"]
