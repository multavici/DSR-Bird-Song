# Base image
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
LABEL Name=dsr-bird-song Version=0.0.4

# Setup system
RUN apt-get clean && apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils tzdata software-properties-common build-essential gcc libsm6 libxrender1 libxext6 libxtst6 libxi6 libfreetype6 git wget nano ffmpeg

# Install Python and pip
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y python3.6

# System upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && apt-get dist-upgrade -y && apt-get autoremove -y

# Install command line applications
RUN apt-get -y update && apt-get install -y wget nano git vim build-essential yasm pkg-config

# Compile and install ffmpeg from source
RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
    cd /root/ffmpeg && \
    ./configure --enable-nonfree --disable-shared --extra-cflags=-I/usr/local/include && \
    make -j8 && make install -j8
ARG CACHEBREAK=1

# Install dependencies
RUN pip3 install matplotlib==3.0.2
RUN pip3 install urllib3==1.24.1
RUN pip3 install scikit-learn==0.20.2
RUN pip3 install pandas==0.24.0
RUN pip3 install librosa==0.6.2
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision
RUN pip3 install lxml
RUN pip3 install tensorboardX
