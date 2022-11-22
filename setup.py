import os
import setuptools
from distutils.core import setup

with open("README.md") as f:
    long_description = f.read()

exec(open('torchsig/version.py').read())

setup(
    name='torchsig',
    version=__version__,
    description='Signal Processing Machine Learning Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='TorchSig Team',
    url='https://github.com/torchdsp/torchsig',
    packages=setuptools.find_packages(),
)
