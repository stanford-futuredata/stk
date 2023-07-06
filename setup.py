from setuptools import setup, find_packages
import torch

setup(
    name="stanford-stk",
    version="0.0.5",
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
    install_requires=[
        "absl-py",
        "numpy",
        "torch",
        # TODO(tgale): Update this to a stable Triton release once the software pipelining feature is supported.
        "triton @ git+https://github.com/openai/triton.git@787cdff#egg=triton&subdirectory=python"
    ],
)
