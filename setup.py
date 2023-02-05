import os
from pathlib import Path
from setuptools import setup, find_packages
import subprocess
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# NOTE: The 'glob' package didn't get the root_dir argument until 3.10.
def glob(ext, root_dir):
    file_list = []
    for dirname, _, files in os.walk(root_dir):
        file_list += [
            f"{dirname}/{f}" for f in files if f.endswith(ext)
        ]
    return file_list

# HACK: Upgrade our CUTLASS dependency so we do not need to monkey-patch
# the source to build.
subprocess.call([
    "sed", "-i", "1i#include \"cutlass/layout/matrix.h\"",
    "third_party/sputnik/third_party/cutlass/include/cutlass/gemm/gemm.h"
])

def generate_sputnik_sources():
    sources = glob(".cu", "third_party/sputnik/sputnik/block/")
    sources += glob(".cc", "third_party/sputnik/sputnik/")

    sputnik_exclude_suffixes = [
        "_test.cu",
        "_benchmark.cu",
        "matrix_utils.cu",
    ]
    def filter_sources(f):
        return not any([
            f.endswith(suffix) for suffix in sputnik_exclude_suffixes
        ])

    # Filter test and benchmark files.
    sources = list(filter(filter_sources, sources))
    return sources

if not torch.cuda.is_available():
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

# NOTE: Initialize all submodules recursively prior to build.
subprocess.call(["git", "submodule", "update", "--init", "--recursive"])

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
print(f"{cwd}/third_party/sputnik/")
ext_modules = [
    CUDAExtension(
        "sputnik_backend",
        ["csrc/backend.cc"] + generate_sputnik_sources(),
        include_dirs = [
            f"{cwd}/third_party/sputnik/",
            f"{cwd}/third_party/sputnik/third_party/cutlass/include/"
        ],
        extra_compile_args={
            "cxx": ["-fopenmp", "-fPIC", "-Wno-strict-aliasing"],
            "nvcc": ["-gencode", "arch=compute_80,code=sm_80"]
        }
    )
]

setup(
    name="stanford-stk",
    version="0.0.3",
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
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["absl-py", "numpy", "torch"],
)
