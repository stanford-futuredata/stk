# Sparse Toolkit

A light-weight PyTorch library for block-sparse matrices and block-sparse matrix multiplication.

## Installation

We recommend using NGC's [`nvcr.io/nvidia/pytorch:21.12-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) PyTorch container and using the pre-built `.whl` under [dist/](https://github.com/tgale96/stk/tree/main/dist):

```bash
export REPO=https://github.com/tgale96/stk/blob/main
pip install ${REPO}/dist/21.12-py3/stk-0.0.1-cp38-cp38-linux_x86_64.whl?raw=true
```

For other environments, please follow the steps outlined in the [Dockerfile](https://github.com/tgale96/stk/blob/main/Dockerfile).

## Citation

```
@article{megablocks-arxiv,
  author    = {Trevor Gale and
               Deepak Narayanan and
               Cliff Young and
               Matei Zaharia},
  title     = {MegaBlocks: Efficient Sparse Training with Mixture-of-Experts},
  journal   = {CoRR},
  volume    = {abs/2211.15841},
  year      = {2022},
}
```
