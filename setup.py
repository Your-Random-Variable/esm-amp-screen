from setuptools import setup, find_packages

setup(
    name="esm_amp_screen",
    version="1.3.0",
    description="High-Throughput ESM-Powered Protein-Peptide Virtual Screening, Physics Engine, & In Silico Lead Optimizer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Antigravity Computational Science Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "gemmi>=0.6.5",
        "rdkit>=2023.09.1",
        "scikit-learn>=1.3.0",
        "biopython>=1.81",
        "PeptideBuilder>=0.1.1",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0"
    ],
    entry_points={
        "console_scripts": [
            "esm-amp-screen = esm_amp_screen.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
)
