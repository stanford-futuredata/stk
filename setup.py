import os
from pathlib import Path
from setuptools import setup, find_packages
import subprocess
import torch

if not torch.cuda.is_available():
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

setup(
    name="stanford-stk",
    version="0.0.4",
    author="Trevor Gale",
    author_email="tgale@stanford.edu",
    description="Sparse Toolkit",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/stanford-futuredata/stk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    install_requires=["absl-py", "numpy", "torch"],
)
