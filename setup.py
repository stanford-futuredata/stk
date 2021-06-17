from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# TODO(tgale): Add the Sputnik build to this script so we don't
# need to already have it installed.
ext_modules = [
    CppExtension(
        'backend',
        ['backend.cc'],
        include_dirs = ["/usr/local/sputnik/include"],
        libraries = ["sputnik"],
        library_dirs = ["/usr/local/sputnik/lib"])
]

setup(
    name='stk_backend',
    ext_modules=ext_modules
    cmdclass={'build_ext': BuildExtension}
)
