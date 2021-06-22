from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# TODO(tgale): Add the Sputnik build to this script so we don't
# need to already have it installed.
ext_modules = [
    CUDAExtension(
        "backend",
        ["csrc/backend.cc"],
        include_dirs = ["/usr/local/sputnik/include"],
        libraries = ["sputnik"],
        library_dirs = ["/usr/local/sputnik/lib"],
        extra_compile_ars={"cxx": ["-fopenmp"]})
]

setup(
    name="stk_backend",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
