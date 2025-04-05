import os
import subprocess
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Custom post-installation step to install Trainers package
def install_trainers():
    subprocess.check_call(["python", "-m", "pip", "install", "./Trainers/"])

# Run the custom installation step
install_trainers()

# Setup configuration
setup(
    name="FaceNet_PyTorch",
    version="0.1.0",
    description="A PyTorch implementation of the FaceNet model for face recognition and clustering.",
    author="Your Name",
    packages=find_packages(),
    install_requires=requirements,
)
