FROM nvcr.io/nvidia/pytorch:23.01-py3

# Install Triton v2.1
RUN pip uninstall -y triton
WORKDIR /tmp/install
RUN git clone https://github.com/openai/triton.git
RUN pip install cmake
WORKDIR /tmp/install/triton/python
RUN pip install -e .

WORKDIR /mount/stk
