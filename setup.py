"""
Setup script for DeepSafaitic
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "docs" / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Neural Epigraphy for Ancient Desert Inscriptions"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="deepsafaitic",
    version="0.1.0",
    description="Neural Epigraphy for Ancient Desert Inscriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepSafaitic Contributors",
    author_email="deepsafaitic@example.com",
    url="https://github.com/YOUR_USERNAME/DeepSafaitic",
    project_urls={
        "Bug Reports": "https://github.com/YOUR_USERNAME/DeepSafaitic/issues",
        "Source": "https://github.com/YOUR_USERNAME/DeepSafaitic",
        "Documentation": "https://github.com/YOUR_USERNAME/DeepSafaitic/tree/main/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="epigraphy, ancient-scripts, safaitic, computer-vision, deep-learning, siamese-networks, archaeology, neural-networks, pytorch, ocr",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: Jupyter",
    ],
)

