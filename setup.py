"""
TTS Voice AI System - Setup Configuration
A production-grade Text-to-Speech system supporting Hindi + English with multiple accents.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the requirements file
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="learn-voice",
    version="0.1.0",
    author="Aman Akhtar",
    author_email="",
    description="A production-grade TTS system supporting Hindi + English with voice cloning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amanakhtar/learn-voice",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tts-synthesize=02_tts_core.inference.synthesizer:main",
            "tts-train=03_training.trainers.train:main",
            "tts-api=05_api.app.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
