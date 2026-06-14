# setup.py
from setuptools import setup, find_packages

setup(
    name="adaptivebelief",
    version="1.0.0",
    author="AdaptiveBelief Team",
    description="Memory Framework for Long-Horizon AI Agents",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
)