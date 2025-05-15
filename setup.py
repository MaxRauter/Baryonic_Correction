from setuptools import setup, find_packages

setup(
    name="BCM",
    version="0.1.0",
    description="Baryonic Correction Model",
    author="Your Name",
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
    extras_require={
        "docs": [
            "sphinx>=4.0.0",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
        ],
    },
)