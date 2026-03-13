# Use NVIDIA CUDA 11.8 base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install OS dependencies, including Python 3.10, venv, and GCC-11 for CUDA 11.8 compatibility
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-glx \
    gcc-11 \
    g++-11 \
    cmake \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 11 as the default compiler and Python 3.10 as default python
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 100 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 100

# Set up the working directory
WORKDIR /app

# Create a Python virtual environment and set it in PATH
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the image
COPY docker/requirements.txt /tmp/requirements.txt

# Install dependencies into the venv
# RUN pip install -r /tmp/requirements.txt

# Create an entrypoint or default command for interactive use
CMD ["tail", "-f", "/dev/null"]
