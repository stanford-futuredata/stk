from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# TODO(tgale): Add the Sputnik build to this script so we don't
# need to already have it installed.
ext_modules = [
    CUDAExtension(
        "sputnik_backend",
        ["csrc/backend.cc"],
        include_dirs = ["/usr/local/sputnik/include"],
        libraries = ["sputnik"],
        library_dirs = ["/usr/local/sputnik/lib"],
        extra_compile_args={"cxx": ["-fopenmp", "-fPIC"]})
]

setup(
    name="stk",
    version="0.0.1",
    author="Trevor Gale",
    description="A PyTorch library for block-sparse matrices and block-sparse matrix multiplication.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[l.strip() for l in open('requirements.txt', 'r').readlines()],
)
