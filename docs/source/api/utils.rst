Utility Functions
================

.. module:: BCM.utils

The ``utils`` module provides utility functions that support the Baryonic Correction Model. These utilities include methods for halo structure calculations, mass profiles, normalization, and visualization.

Functions
---------

Halo Structure Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calc_concentration

.. autofunction:: calc_r_ej

.. autofunction:: calc_r_ej2

.. autofunction:: calc_R_h

Mass Calculations
~~~~~~~~~~~~~~~

.. autofunction:: cumul_mass

.. autofunction:: cumul_mass_single

.. autofunction:: bracket_rho0

Normalization
~~~~~~~~~~~

.. autofunction:: normalize_component_total

.. autofunction:: normalize_component

Visualization
~~~~~~~~~~~

.. autofunction:: plot_bcm_profiles

Examples
--------

Calculating Halo Concentration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from BCM import utils
    
    # Calculate concentration for a Milky Way-sized halo at z=0
    M200 = 1e12  # Msun/h
    z = 0.0
    
    c = utils.calc_concentration(M200, z)
    print(f"Concentration for a {M200:.2e} Msun/h halo at z={z}: c = {c:.2f}")

Cumulative Mass Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from BCM import utils
    import numpy as np
    
    # Create sample data
    r_array = np.logspace(-3, 1, 1000)  # from 0.001 to 10 Mpc/h
    rho_array = np.ones_like(r_array) * 1e7  # constant density
    
    # Calculate cumulative mass at a specific radius
    radius = 0.5  # Mpc/h
    mass = utils.cumul_mass_single(radius, rho_array, r_array)
    print(f"Mass within {radius} Mpc/h: {mass:.2e} Msun/h")
    
    # Calculate cumulative mass profile
    masses = utils.cumul_mass(r_array, rho_array)
    print(f"Total mass: {masses[-1]:.2e} Msun/h")

Component Normalization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from BCM import utils
    from BCM import density_profiles as dp
    
    # Normalize a density component to contain a specified mass
    r_s = 0.2  # Scale radius in Mpc/h
    r200 = 1.0  # Virial radius in Mpc/h
    M200 = 1e14  # Msun/h
    
    # Get normalization factor for an NFW profile
    norm = utils.normalize_component(
        dp.rho_nfw,         # Density function
        (r_s, 1.0, 5*r200), # Arguments: r_s, rho0=1.0, r_tr=5*r200
        0.8 * M200,         # Target mass (80% of M200)
        r200                # Integration radius
    )
    
    print(f"Normalization factor: {norm:.2e}")

