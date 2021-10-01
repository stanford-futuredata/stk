# Sparse Toolkit

A PyTorch library for sparse matrices.

## Installation

Use the docker container with nvidia-docker. Run `sudo docker build . -t stk-dev` to build the image. Run `sh docker.sh` to launch the container and mount this repository at /mount/stk. Once in the container, run `python setup.py install` to build and install the kernel bindings.
