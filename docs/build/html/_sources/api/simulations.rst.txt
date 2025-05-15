Simulations
===========

The ``simulations`` module provides the core functionality for applying the Baryonic Correction Model to cosmological simulations. It contains classes for reading simulation data, calculating density profiles, and applying displacement fields.

CAMELSReader
-----------

.. autoclass:: BCM.simulations.CAMELSReader
   :special-members: __init__
   :noindex:

Main Methods
~~~~~~~~~~~

**Initialization and Setup**

.. automethod:: BCM.simulations.CAMELSReader.init_calculations

**Profile Calculation**

.. automethod:: BCM.simulations.CAMELSReader.calculate
.. automethod:: BCM.simulations.CAMELSReader._compute_density_profiles
.. automethod:: BCM.simulations.CAMELSReader._compute_mass_profiles

**Particle Manipulation**

.. automethod:: BCM.simulations.CAMELSReader.apply_displacement
.. automethod:: BCM.simulations.CAMELSReader.get_halo_particles
.. automethod:: BCM.simulations.CAMELSReader.get_particles_relative_position

**Visualization and Analysis**

.. automethod:: BCM.simulations.CAMELSReader.plot_density_profiles
.. automethod:: BCM.simulations.CAMELSReader.plot_displacement
.. automethod:: BCM.simulations.CAMELSReader.calc_displ_and_compare_powerspectrum
.. automethod:: BCM.simulations.CAMELSReader.print_components

Example Usage
------------

.. code-block:: python

    # Initialize a CAMELS simulation reader
    reader = simulations.CAMELSReader(
        path_group="path/to/groups/file.hdf5",
        path_snapshot="path/to/snapshot/file.hdf5"
    )
    
    # Initialize the BCM calculation with halo parameters
    reader.init_calculations(
        M200=1e14,      # Halo mass (M☉)
        r200=0.77,      # Virial radius (Mpc/h)
        c=3.2,          # Concentration parameter
        h=0.6777,       # Hubble parameter
        z=0,            # Redshift
        Omega_m=0.3071, # Matter density parameter
        verbose=True    # Show detailed output
    )
    
    # Calculate BCM profiles
    reader.calculate()
    
    # Apply displacement to particles and compare power spectra
    reader.calc_displ_and_compare_powerspectrum(output_file='powerspectrum.png')