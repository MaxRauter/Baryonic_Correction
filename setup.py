from setuptools import setup, find_packages

setup(
    name="baryonic-correction",  # Using hyphen is more common for PyPI
    version="0.1.0",
    description="Baryonic Correction Model for cosmological simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Max Rauter",  
    author_email="maxi.rauter@gmx.net",  
    url="https://github.com/MaxRauter/Baryonic_Correction",  
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "hdf5plugin",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    extras_require={
        "docs": [
            "sphinx>=4.0.0",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
        ],
    },
)