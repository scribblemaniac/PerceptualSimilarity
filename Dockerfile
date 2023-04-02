FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL maintainer="Seyoung Park <seyoung.arts.park@protonmail.com>"

# This Dockerfile is forked from Tensorflow Dockerfile

# Pick up some PyTorch gpu dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y \
      cpio \
      file \
      flex \
      g++ \
      make \
      patch \
      rpm2cpio \
      unar \
      wget \
      xz-utils \
      zlib1g-dev \
      libjpeg8-dev

RUN apt-get install -y python3-pip python3-opencv
RUN pip3 install -U pip
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install numpy scipy jupyter matplotlib scikit-image tqdm

#RUN apt-get install -y --no-install-recommends ffmpeg

WORKDIR /build

RUN apt-get install -y git #libavutil-dev libavformat-dev libswresample-dev libavcodec-dev libswscale-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget && \
    MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH $PATH:/miniconda/bin
RUN conda install -c conda-forge ffmpeg=4.2
RUN wget "https://raw.githubusercontent.com/FFmpeg/FFmpeg/release/4.2/libavcodec/bsf.h" -O "/miniconda/include/libavcodec/bsf.h"

RUN git clone https://github.com/pytorch/vision.git /build/vision && \
  cd /build/vision && \
  git checkout release/0.14

RUN apt-get install -y kmod
ARG nvidia_binary_version="525.89.02"
ARG nvidia_binary="NVIDIA-Linux-x86_64-${nvidia_binary_version}.run"
RUN wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/${nvidia_binary_version}/${nvidia_binary} && \
  chmod +x ${nvidia_binary} && \
  ./${nvidia_binary} --accept-license --ui=none --no-kernel-module --no-questions && \
  rm -rf ${nvidia_binary}

COPY Video_Codec_SDK_12.0.16 Video_Codec
ARG TORCHVISION_INCLUDE="/build/Video_Codec/Interface"
ARG TORCHVISION_LIBRARY="/build/Video_Codec/Lib/linux/stubs/x86_64"
ARG CUDA_HOME=/usr/local/cuda
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ARG CACHE_BREAKER=2

RUN cd /build/vision && \
  python3 setup.py install

WORKDIR /

RUN rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI. Maybe PyTorch needs this too.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# IPython
EXPOSE 8888

WORKDIR "/notebooks"
