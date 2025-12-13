"""
Combat Racing Championship - Installation Script
=================================================

Professional installation script for the Combat Racing RL project.

Usage:
    pip install -e .                    # Development install
    pip install .                       # Standard install
    python setup.py develop             # Alternative dev install
    python setup.py sdist bdist_wheel   # Build distribution
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#") and not line.startswith("--")
        ]

setup(
    # Project Metadata
    name="combat-racing-rl",
    version="1.0.0",
    description="AI-Powered Combat Racing Game with Advanced Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@ensam.ma",
    url="https://github.com/yourusername/combat-racing-rl",
    license="MIT",
    
    # Package Configuration
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "docs"]),
    package_dir={"": "."},
    include_package_data=True,
    
    # Python Version
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional Dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=2.0.0",
        ],
        "viz": [
            "streamlit>=1.24.0",
            "plotly>=5.14.0",
            "seaborn>=0.12.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",  # GPU acceleration
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "sphinx>=7.0.0",
            "streamlit>=1.24.0",
            "plotly>=5.14.0",
        ],
    },
    
    # Entry Points (CLI commands)
    entry_points={
        "console_scripts": [
            "combat-racing-train=scripts.train:main",
            "combat-racing-demo=scripts.demo:main",
            "combat-racing-eval=scripts.evaluate:main",
            "combat-racing-dashboard=scripts.visualize:main",
        ],
    },
    
    # Package Data
    package_data={
        "": [
            "config/*.yaml",
            "assets/**/*",
            "docs/**/*",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords=[
        "reinforcement-learning",
        "deep-learning",
        "machine-learning",
        "ai",
        "game",
        "racing",
        "dqn",
        "ppo",
        "q-learning",
        "multi-agent",
        "pytorch",
        "pygame",
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/combat-racing-rl/issues",
        "Source": "https://github.com/yourusername/combat-racing-rl",
        "Documentation": "https://combat-racing-rl.readthedocs.io/",
    },
    
    # Zip Safe
    zip_safe=False,
)
