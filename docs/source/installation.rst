Installation
===========

Prerequisites
------------

Before installing the Baryonic Correction Model, ensure you have the following dependencies:

* Python 3.8 or higher
* NumPy (>=1.20.0)
* SciPy (>=1.7.0)
* Matplotlib (>=3.4.0)
* h5py (for working with HDF5 simulation files)
* tqdm (for progress indicators)

Installing from PyPI
-------------------

The simplest way to install the package is using pip:

.. code-block:: bash

    pip install baryonic-correction

Installing from Source
---------------------

To install the latest development version from the repository:

.. code-block:: bash

    git clone https://github.com/yourusername/Baryonic_Correction.git
    cd Baryonic_Correction
    pip install -e .

This will install the package in development mode, allowing you to modify the code and have the changes immediately available.

Verifying Installation
---------------------

To verify that the installation was successful, you can run a simple test:

.. code-block:: python

    from BCM import utils
    utils.verify_schneider()

This should run the verification tests comparing the implementation against reference cases from Schneider et al. (2015).

Using with CAMELS Simulations
----------------------------

If you plan to use the BCM with CAMELS simulations, make sure you have the CAMELS data files downloaded and organized in the standard CAMELS directory structure.

The BCM package will automatically detect and work with the following CAMELS file formats:
- Group catalogs (.hdf5)
- Snapshot files (.hdf5)
