# Baryonic_Correction

## Description
A Python package for applying baryonic corrections to cosmological simulations following Schneider & Teyssier 2015. This tool helps account for the effects of baryonic physics on dark matter distributions, improving the accuracy of cosmological models without the need of expensive hydrodynamic simulations.

With always increasing compute power, cosmological simulations have gotten more accurate and span over larger redshift and space. But increasing accuracy still comes with a price because hydrodynamic simulations computationally very expensive compared to only N-Body ones. The baryonic correction method proposed by Schneider & Teyssier in 2015 


## Table of Contents
- [Usage](#usage)
- [Requirements](#requirements)
- [User Stories](#user-stories)
- [Examples](#examples)
- [License](#license)
- [Authors](#authors)


## Usage

```python
import baryonic_correction as bc

# Load simulation data from Dm Snapshot
data = bc.load_simulation("path/to/simulation")

# Apply baryonic correction
corrected_data = bc.apply_correction(data)

# Save or analyze results
bc.save_results(corrected_data, "path/to/output")
```

## Requirements
- Baryonic correction methods depend on a lot of formulas and profile which can be quickly overwelming. This package aims to make it easy to correct N-body simulations to come close to the computationally more expensive hydrodynamic ones.
- This package should be able to read in local camels dark matter snapshots and go through halo by halo and calculate a displacement function. This displacement is then added for each halo respectively and return as a snapshot. Comparison between two snapshots for accuracy testing.
- Correct read in of snapshot in two files (group, and snapshot) is necessary for further calculations. Accurate moddeling from Schneider & Teyssier 2015 will provide accurate results without much computation. The comparison is necessary to verify the results with actual dark matter and hydrodynamic simulations.
- Dependencies:
    - Python 3.8+
    - NumPy
    - SciPy
    - Matplotlib
    - Astropy
    - h5py (for HDF5 file support)

## User stories
- Max is working with Camels Simulations and wants to add a mass threshold to halo masses because small halos have very little impact but take reasonable amount of computational time.
- Max is working with subparts of Camels and wants to simulate with custom parameters because different halos maybe need different model parameters.

- Max is trying different Simulations and needs to be able to customly input necessary parameters if datastructure is different to camles. To allow tool to work with a wider variety of simulations.
- Max was working with various camels snapshots and wants to add input checks because the use of gas+dm simulations for the correction is allowed to calculated but makes no sense.


## Examples
### Example: Basic Correction and Verification
```python
import baryonic_correction as bc

# Apply default correction to a simulation
data = bc.load_simulation("examples/dm_only_sim.hdf5")
corrected = bc.apply_correction(data)
bc.verify_sim(corrected,"examples/gas_sim.hdf5")
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Maximilian Rauter - [GitHub](https://github.com/MaxRauter)