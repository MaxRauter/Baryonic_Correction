Density Profiles
===============

.. module:: BCM.density_profiles

The ``density_profiles`` module provides functions for calculating various density and mass profiles used in the Baryonic Correction Model (BCM). These profiles describe the distribution of different matter components in and around halos.

Functions
--------

NFW Profile
~~~~~~~~~~~

.. autofunction:: rho_nfw

.. autofunction:: mass_nfw_analytical

Background Density
~~~~~~~~~~~~~~~~~

.. autofunction:: rho_background

Baryonic Components
~~~~~~~~~~~~~~~~~~

.. autofunction:: y_bgas

.. autofunction:: y_egas

.. autofunction:: y_cgal

Dark Matter Components
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: y_rdm_fixed_xi

.. autofunction:: y_rdm_ac

.. autofunction:: y_rdm_ac2

Utility Functions
~~~~~~~~~~~~~~~~

.. autofunction:: mass_profile

Examples
--------

NFW Profile
~~~~~~~~~~~

.. code-block:: python

    from BCM import density_profiles as dp
    import numpy as np
    
    # Define parameters
    r_s = 0.1  # Scale radius in Mpc/h
    rho0 = 1e7  # Characteristic density in Msun/h/(Mpc/h)^3
    r_tr = 1.0  # Truncation radius in Mpc/h
    
    # Create radius array
    r_vals = np.logspace(-3, 1, 1000)  # from 0.001 to 10 Mpc/h
    
    # Calculate NFW density profile
    rho_nfw_vals = np.array([dp.rho_nfw(r, r_s, rho0, r_tr) for r in r_vals])
    
    # Calculate analytical mass
    r = 0.5  # Mpc/h
    mass = dp.mass_nfw_analytical(r, r_s, rho0)
    print(f"Mass within {r} Mpc/h: {mass:.2e} Msun/h")

Component Profiles
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from BCM import density_profiles as dp
    import numpy as np
    
    # Define parameters
    r_s = 0.1  # Scale radius in Mpc/h
    r200 = 0.5  # Virial radius in Mpc/h
    r_ej = 2.0  # Ejection radius in Mpc/h
    R_h = 0.01  # Hernquist scale radius in Mpc/h
    rho0 = 1e7  # Characteristic density in Msun/h/(Mpc/h)^3
    r_tr = 4.0  # Truncation radius in Mpc/h
    c = 5.0  # Concentration parameter
    
    # Create radius array
    r_vals = np.logspace(-3, 1, 1000)  # from 0.001 to 10 Mpc/h
    
    # Normalizations (would normally be calculated in a real application)
    norm_bgas = 0.5
    norm_egas = 1.5e13
    norm_cgal = 5e11
    
    # Calculate component profiles
    bgas_vals = np.array([dp.y_bgas(r, r_s, r200, norm_bgas, c, rho0, r_tr) for r in r_vals])
    egas_vals = np.array([dp.y_egas(r, norm_egas, r_ej) for r in r_vals])
    cgal_vals = np.array([dp.y_cgal(r, norm_cgal, R_h) for r in r_vals])
