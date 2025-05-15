Introduction
============

What is the Baryonic Correction Model?
--------------------------------------

The Baryonic Correction Model (BCM) is a computational technique that accounts for the effects of baryonic physics (gas, stars, black holes) on dark matter-only (DMO) cosmological simulations. It applies a physically motivated displacement field to dark matter particles based on an analytic model of galaxy formation physics.

This implementation provides an efficient way to mimic the effects of complex hydrodynamical processes without the enormous computational cost of full hydrodynamical simulations.

Scientific Motivation
--------------------

In the standard ΛCDM cosmological model, most of the matter in the universe is in the form of dark matter. While DMO simulations provide an excellent approximation of the large-scale structure of the universe, they neglect the effects of baryonic physics that become important on smaller scales (k > 0.1 h/Mpc):

* **Adiabatic Contraction**: Gas cooling causes dark matter to contract in the central regions of halos
* **Feedback Processes**: Supernova and AGN feedback can expel gas from halos, modifying the matter distribution
* **Star Formation**: Converting gas into stars changes the spatial distribution of matter

These processes significantly alter the matter power spectrum at small scales, which is critical for accurate predictions in several cosmological probes including weak lensing.

The BCM Approach
---------------

Based on the methodology developed by Schneider et al. (2015, 2019), our BCM implementation:

1. Identifies dark matter halos in a simulation
2. Models the redistribution of matter using physically motivated density profiles:
   * Contracted dark matter
   * Bound gas
   * Ejected gas
   * Central galaxy
3. Calculates a displacement field that shifts dark matter particles to match this redistribution
4. Applies the displacement field to create a corrected simulation

Key Features
-----------

* Support for CAMELS simulation format
* Fully configurable baryonic parameters
* Efficient, vectorized implementation
* Built-in validation against Schneider et al. (2015) reference cases
* Tools for analyzing and visualizing the power spectrum effects

When to Use BCM
--------------

The Baryonic Correction Model is ideal when you need to:

* Account for baryonic effects in large cosmological simulations
* Conduct parameter space explorations of baryonic physics
* Generate realistic matter distributions for mock observations
* Test the sensitivity of cosmological probes to baryonic physics