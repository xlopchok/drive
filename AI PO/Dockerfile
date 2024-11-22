FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

WORKDIR /AI PO
COPY ./requirements.txt .

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils git wget curl unzip && \ 
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM builder AS stage

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libopencv-dev \
    libeigen3-dev \
    libassimp-dev \
    cmake \
    && apt-get clean

COPY mesh_projection_mt.cpp .

RUN g++ -std=c++17 -O2 -o mesh_projection_mt mesh_projection_mt.cpp \
    -I/usr/include/opencv4 \
    -I/usr/include/eigen3 \
    -lopencv_core \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lassimp \
    -pthread

WORKDIR /AI PO

RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "front_inference.py"]
