FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    cmake \
    g++ \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*
