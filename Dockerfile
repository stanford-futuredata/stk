FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Install tools and dependencies.
RUN apt-get -y update --fix-missing
RUN apt-get update -y
RUN apt-get install -y \
  emacs \
  git \
  wget \
  libgoogle-glog-dev \
  python3 \
  python3-pip \
  nsight-compute-2020.3.1

ENV PATH="/opt/nvidia/nsight-compute/2020.3.1/:${PATH}"

# Install CMake.
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.sh
RUN mkdir /usr/local/cmake
RUN sh cmake-3.21.3-linux-x86_64.sh --skip-license --prefix=/usr/local/cmake
ENV PATH="/usr/local/cmake/bin:${PATH}"

# Install Sputnik.
RUN mkdir /mount
WORKDIR /mount
RUN git clone --recursive https://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@github.com/tgale96/sputnik.git && \
	mkdir sputnik/build

# HACK: Send a PR to fix this include.
RUN sed -i '1i#include "cutlass/layout/matrix.h"' sputnik/third_party/cutlass/include/cutlass/gemm/gemm.h
WORKDIR /mount/sputnik/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF \
	-DCUDA_ARCHS="80" -DCMAKE_INSTALL_PREFIX=/usr/local/sputnik && \
	make -j8 install

# Install PyTorch.
#
# TODO(tgale): Update this to 11.3 as soon as possible.
RUN pip3 install torch==1.9.0+cu111 \
    	  torchvision==0.10.0+cu111 \
     	  torchaudio==0.9.0 \
     	  -f https://download.pytorch.org/whl/torch_stable.html

# Install STK deps and set path.
RUN pip3 install absl-py
ENV PYTHONPATH="/mount/stk:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/local/sputnik/lib:${LD_LIBRARY_PATH}"

# Aliases for py3.
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN echo 'alias pip="pip3"' >> ~/.bashrc

# Set the working directory.
WORKDIR /mount/stk
