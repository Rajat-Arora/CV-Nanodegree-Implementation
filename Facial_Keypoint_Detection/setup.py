"""
Setup script for Facial Keypoint Detection package.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="facial-keypoint-detection",
    version="1.0.0",
    author="Rajat",
    author_email="arora24rajat@gmail.com",
    description="CNN-based facial keypoint detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rajat-Arora/facial-keypoint-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)
