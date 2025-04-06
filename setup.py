"""
setup.py

Setup script for the Growing Neural Networks project.

This script defines the package metadata, dependencies, and installation
requirements to ensure the project can be installed and run easily.

Usage:
    pip install -e .

Author: Sam Collin
"""

from setuptools import setup, find_packages

setup(
    name="growing_networks_project",
    version="0.1",
    description="Growing neural networks project",
    author="Sam Collin",
    packages=find_packages(where="scripts"),
    package_dir={"": "scripts"},
    include_package_data=True,
    install_requires=["torch", "torchvision", "matplotlib", "numpy"],
    python_requires=">=3.8",
)
