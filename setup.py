from distutils.core import setup
import setuptools

with open("README.md") as f:
    long_description = f.read()

setup(
    name="torchsig",
    version="0.1.0",
    description="Signal Processing Machine Learning Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TorchSig Team",
    url="https://github.com/torchdsp/torchsig",
    packages=setuptools.find_packages(),
)
