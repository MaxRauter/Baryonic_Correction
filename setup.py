from setuptools import setup, find_packages

setup(
    name="bcm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "hdf5plugin"
    ],
)