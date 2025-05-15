Abundance Fractions
==================

.. module:: BCM.abundance_fractions

The ``abundance_fractions`` module provides functions for calculating mass fractions of different components in dark matter halos according to the Baryonic Correction Model. These fractions describe how the cosmic baryons are distributed between bound gas, ejected gas, and the central galaxy.

Functions
---------

Baryon Component Fractions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: f_bgas

.. autofunction:: f_cgal

.. autofunction:: f_egas

.. autofunction:: f_rdm

Helper Functions
~~~~~~~~~~~~~~

.. autofunction:: g_func

Examples
--------

Calculating Component Fractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from BCM import abundance_fractions as af
    
    # Define parameters
    halo_mass = 1e14  # Msun/h
    cosmic_baryon_fraction = 0.157  # Omega_b/Omega_m
    
    # Calculate individual component fractions
    f_bgas_val = af.f_bgas(halo_mass, cosmic_baryon_fraction)
    f_cgal_val = af.f_cgal(halo_mass)
    f_egas_val = af.f_egas(f_bgas_val, f_cgal_val, cosmic_baryon_fraction)
    f_rdm_val = af.f_rdm(cosmic_baryon_fraction)
    
    # Print results
    print(f"For a halo of mass {halo_mass:.2e} Msun/h:")
    print(f"  Bound gas fraction: {f_bgas_val:.3f}")
    print(f"  Central galaxy fraction: {f_cgal_val:.3f}")
    print(f"  Ejected gas fraction: {f_egas_val:.3f}")
    print(f"  Relaxed dark matter fraction: {f_rdm_val:.3f}")
    print(f"  Sum of all fractions: {f_bgas_val + f_cgal_val + f_egas_val + f_rdm_val:.3f}")

Mass Dependence of Fractions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from BCM import abundance_fractions as af
    
    # Create a range of halo masses
    masses = np.logspace(11, 15, 100)  # 10^11 to 10^15 Msun/h
    cosmic_baryon_fraction = 0.157
    
    # Calculate fractions for each mass
    f_bgas_vals = [af.f_bgas(m, cosmic_baryon_fraction) for m in masses]
    f_cgal_vals = [af.f_cgal(m) for m in masses]
    f_egas_vals = [af.f_egas(b, c, cosmic_baryon_fraction) 
                  for b, c in zip(f_bgas_vals, f_cgal_vals)]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.semilogx(masses, f_bgas_vals, label='Bound Gas')
    plt.semilogx(masses, f_cgal_vals, label='Central Galaxy')
    plt.semilogx(masses, f_egas_vals, label='Ejected Gas')
    plt.axhline(cosmic_baryon_fraction, ls='--', color='black', 
               label='Cosmic Baryon Fraction')
    
    plt.xlabel('Halo Mass (Msun/h)')
    plt.ylabel('Mass Fraction')
    plt.title('Mass Dependence of Baryonic Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
