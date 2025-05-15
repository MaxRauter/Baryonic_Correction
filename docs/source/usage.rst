Usage
=====

Basic Usage
----------

The Baryonic Correction Model provides tools to correct dark matter-only simulations by accounting for baryonic effects. Here's how to use the basic functionality:

Loading Simulation Data
~~~~~~~~~~~~~~~~~~~~~~

First, load your simulation data:

.. code-block:: python

    from BCM import simulations
    
    # Initialize with paths to your simulation files
    reader = simulations.CAMELSReader(
        path_group="path/to/groups/file.hdf5",
        path_snapshot="path/to/snapshot/file.hdf5"
    )

Initializing the BCM Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Next, initialize the BCM model and start the calculations:

.. code-block:: python

    # Set up BCM calculations
    reader.init_calculations()

View resulting displacements:
~~~~~~~~~~~~~~~~~~~

Get the displaced vs original power spectrum:

.. code-block:: python

    # Calculate and plot power spectrum
    reader.calc_displ_and_compare_powerspectrum()

Running Verification Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

To verify your implementation against reference cases:

.. code-block:: python

    from BCM import utils
    
    # Run verification against Schneider et al. 2015
    utils.verify_schneider()


Example Workflow
---------------

Here's a complete example workflow:

.. code-block:: python

    import matplotlib.pyplot as plt
    from BCM import simulations, utils
    
    # Initialize reader
    reader = simulations.CAMELSReader(
        path_group="simulation/groups/fof_subhalo_tab_033.hdf5",
        path_snapshot="simulation/snapshots/snap_033.hdf5"
    )
    
    # Setup BCM
    reader.init_calculations()

    # Apply displacement and calculate power spectrum
    reader.calc_displ_and_compare_powerspectrum(output_file='power_spectrum.png')
    
    print("Analysis complete. Check power_spectrum.png")