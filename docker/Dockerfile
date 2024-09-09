# Use the official Ubuntu image as the base
FROM ubuntu:20.04

# Set environment variables to prevent interactive installation prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-tk\
    libblas-dev \
    liblapack-dev \
    gfortran \
    wget \
    cmake \
    git \
    libeigen3-dev \
    libboost-all-dev \
    libomp-dev \
    libopencv-dev \
    time \
    texlive-full \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /home/lpvds

## Install python requirements 
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

# Copy files from the host machine's adjacent folder (../your_folder) to the container's /app directory
# Adjust the source path as needed
COPY ./dataset ./dataset
COPY ./src ./src
COPY main.py .
COPY main_increm.py .

# Copy mosek license
COPY ./mosek /root/mosek

# Default command to run when the container starts
CMD ["bash"]

