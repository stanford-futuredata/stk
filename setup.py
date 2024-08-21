from setuptools import find_packages, setup

install_requires = [
    'torch>=2.3.0,<3.0',
    'triton>=2.1.0',
]

extra_deps = {}

extra_deps['dev'] = [
    'absl-py',
]

extra_deps['all'] = list(set(dep for deps in extra_deps.values() for dep in deps))

setup(
    name="stanford-stk",
    version="0.7.1",
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
    install_requires=install_requires,
    extras_require=extra_deps,
)
