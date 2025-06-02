import numpy as np
import os
import h5py
import glob
import hdf5plugin
import matplotlib.pyplot as plt
from tqdm import tqdm
from Baryonic_Correction import density_profiles as dp
from Baryonic_Correction import utils as ut
from Baryonic_Correction import abundance_fractions as af
from Baryonic_Correction import parameters as par


class CAMELSReader:
    """
    A class for reading and handling CAMELS simulation data.
    Stores important parameters from the simulations.
    """
    
    def __init__(self, path_group=None, path_snapshot=None, index = None, verbose = False):
        """
        Initialize the CAMELSReader with optional paths to group and snapshot data.
        
        Parameters
        ----------
        path_group : str, optional
            Path to the directory containing the group (halo) data.
        path_snapshot : str, optional
            Path to the directory containing the snapshot (simulation) data.
        index : int, optional
            Index of the simulation or snapshot to load.
        verbose : bool, default False
            If True, enables verbose output during data loading.
            
        Notes
        -----
        If `path_group` is provided, halo data will be loaded automatically.
        If `path_snapshot` is provided, simulation and particle data will be loaded automatically.
        """
        self.path_group = path_group
        self.path_snapshot = path_snapshot
        self.index = index
        self.verbose = verbose
        # Only load data if paths are provided
        if path_group and os.path.exists(path_group):
            self._load_halodata()
        if path_snapshot and os.path.exists(path_snapshot):
            print(f"Loading snapshot data...")
            self._load_simdata()
            self._load_particles()
        
    def _load_halodata(self):
        """
        Load halo data from the CAMELS simulation.
        
        This method reads and processes halo-related information from the specified
        group file path, storing properties like mass, radius, and positions.
        
        Returns
        -------
        bool or None
            Returns False if the path does not exist, None otherwise.
        
        Notes
        -----
        The loaded data is stored in the `self.halo` attribute as a dictionary
        with keys 'm200', 'r200', 'lentype_h', 'pos', and 'id'.
        """
        path = self.path_group
        if path is None:
            print("No path provided.")
            return
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return False
        
        try:
            with h5py.File(path, 'r') as f:
                    m200 = f['Group/Group_M_Crit200'][:] * 1e10     # Msun/h
                    r200 = f['Group/Group_R_Crit200'][:] / 1e3      # Mpc/h
                    lentype_h = f['Group/GroupLenType'][:]
                    halo_pos = f['Group/GroupPos'][:]
                    halo_id = range(len(m200))
                    self.halo = {
                        'm200': m200,
                        'r200': r200,
                        'lentype_h': lentype_h,
                        'pos': halo_pos,
                        'id': halo_id
                    }
        except Exception as e:
            print(f"Error loading halo data: {e}")
            return 
       
    def _load_simdata(self):
        """
        Load simulation data from the CAMELS snapshot.
        
        This method reads and processes general simulation information from the
        snapshot file, including cosmological parameters and box properties.
        
        Returns
        -------
        bool or None
            Returns False if the path does not exist, None otherwise.
            
        Notes
        -----
        The loaded data is stored as attributes of the object, including
        boxsize, redshift, Hubble parameter, and cosmological densities.
        """
        path = self.path_snapshot
        if path is None:
            print("No path provided.")
            return
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return False
        
        try:
            with h5py.File(path, 'r') as f:
                # Read the header
                if 'Header' in f:
                    header = f['Header']
                    
                    # Store common simulation properties
                    if 'BoxSize' in header.attrs:
                        self.boxsize = header.attrs['BoxSize']/ 1e3  # Convert to Mpc/h
                    if 'Redshift' in header.attrs:
                        self.z = header.attrs['Redshift']
                    if 'Time' in header.attrs:
                        self.time = header.attrs['Time']
                    if 'HubbleParam' in header.attrs:
                        self.h = header.attrs['HubbleParam']
                    if 'Omega0' in header.attrs:
                        self.Om = header.attrs['Omega0']
                    if 'OmegaLambda' in header.attrs:
                        self.Ol = header.attrs['OmegaLambda']
                    # Calculate baryon fraction from cosmological parameters
                    if 'OmegaBaryon' in header.attrs:
                        self.Ob = header.attrs['OmegaBaryon']
                    else:
                        # Use a standard value if not available
                        self.Ob = 0.0483
                    self.fbar = self.Ob / self.Om
        except Exception as e:
            print(f"Error loading halo data: {e}")
            return
                
    def _calc_offset(self, index = None):
        """
        Calculate the offset for the specified halo index.

        This method computes the sum of the lengths of all Friends-of-Friends (FoF) halos
        preceding the specified index, and returns the second element of the resulting sum.
        If no index is provided, it uses the current object's `self.index` attribute.

        Parameters
        ----------
        index : int, optional
            The index of the halo for which to calculate the offset. If None, uses `self.index`.

        Returns
        -------
        offset : int
            The offset value corresponding to the sum of the lengths of all previous FoF halos.
        """
        if index is None:
            index = self.index
        offset = np.sum(self.halo['lentype_h'][:index], axis=0)[1] #this is the sum of the lengths of all FoF halos previous to the one we consider
        return offset
    
    def _load_particles(self):
        """
        Load particle data for halos from the simulation snapshot.
        
        This method reads particle positions, velocities, and IDs for each halo
        specified either by self.index or for all halos if self.index is None.
        
        Notes
        -----
        The loaded data is stored in the `self.particles` attribute as a list of dictionaries,
        where each dictionary contains 'pos', 'vel', and 'id' for a halo.
        Positions and velocities are converted to units of Mpc/h.
        """
        path = self.path_snapshot        
        if self.index == None:
            halos = self.halo['id'][:]
        else:
            halos = [self.index]
        particles = []
        with h5py.File(path, 'r') as f:
            # Read the particle data
            if 'PartType1' in f:
                for halo in halos:
                    start = self._calc_offset(halo)
                    stop = start + self.halo['lentype_h'][halo][1]
                    parttype1 = f['PartType1']
                    particles.append({
                        'pos': parttype1['Coordinates'][start:stop]/1e3,
                        'vel': parttype1['Velocities'][start:stop]/1e3,
                        'id': parttype1['ParticleIDs'][start:stop],
                    })
        self.particles = particles
    
    def get_halo_particles(self, index = None):
        """
        Retrieve particle data for a specific halo.
        
        This method loads position, velocity, and ID information for particles
        belonging to the specified halo. If no index is provided, all particles
        are retrieved.
        
        Parameters
        ----------
        index : int, optional
            The index of the halo for which to retrieve particles. If None, 
            all particles from the snapshot are returned.
        
        Returns
        -------
        dict or None
            A dictionary containing particle data with the following keys:
            - 'pos': positions of particles in Mpc/h
            - 'vel': velocities of particles in appropriate units
            - 'id': particle IDs
            Returns None if the file does not exist or if there's an error loading the data.
        
        Notes
        -----
        The particle data is also stored in the `self.particles` attribute.
        """
        path = self.path_snapshot 
        #print(f"Trying to open snapshot file: {path}")  # Add this line
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return None       
        if index == None:
            start = 0
            stop = None
        else:
            start = self._calc_offset(index)
            stop = start + self.halo['lentype_h'][index][1]
        with h5py.File(path, 'r') as f:
            # Read the particle data
            if 'PartType1' in f:
                parttype1 = f['PartType1']
                particles ={
                    'pos': parttype1['Coordinates'][start:stop]/1e3,
                    'vel': parttype1['Velocities'][start:stop]/1e3,
                    'id': parttype1['ParticleIDs'][start:stop],
                }
        self.particles = particles
        return particles
    
    def get_halo_center(self, index=None):
        """
        Get the center position of a halo.
        
        Parameters
        ----------
        index : int, optional
            The index of the halo for which to retrieve the center. If None,
            and `self.index` is set, uses that. If both are None, returns
            centers for all halos.
        
        Returns
        -------
        numpy.ndarray
            The 3D coordinates of the halo center in Mpc/h.
        """
        if index is None:
            if self.index is None:
                return self.halo['pos']
            else: 
                index = self.index
        return self.halo['pos'][index]/1e3
    
    def get_particles_relative_position(self, index=None):
        """
        Get particle positions relative to their halo center.
        
        This method calculates the position of particles relative to the center 
        of the halo they belong to.
        
        Parameters
        ----------
        index : int, optional
            The index of the halo for which to calculate relative positions.
            If None, relative positions are calculated for all halos.
        
        Returns
        -------
        numpy.ndarray or dict
            If `index` is provided, returns a numpy array of shape (N, 3) containing
            the positions of N particles relative to the halo center.
            If `index` is None, returns a dictionary mapping halo indices to their
            relative particle positions.
            Returns None if there's an error processing the data.
        
        Notes
        -----
        For multiple halos, this function produces verbose output about the
        processing progress and results.
        """
        if index is not None:
            particles = self.get_halo_particles(index)
            
            if particles is None or 'pos' not in particles:
                print(f"No valid particles found for halo {index}")
                return None
            
            center = self.get_halo_center(index)
            if center is None:
                print(f"No center found for halo {index}")
                return None
            
            rel_pos = particles['pos'] - center
            #print(f"Relative positions shape of particles in halo {index}: {np.shape(rel_pos)}")
            return rel_pos
        else:
            # Use all halos
            indices = self.halo['id']
            
            result = {}
            for idx in indices:
                print(f"Processing halo {idx}...")
                particles = self.get_halo_particles(idx)
                center = self.get_halo_center(idx)
                rel_pos = particles['pos'] - center
                if rel_pos is not None:
                    result[idx] = rel_pos
                else:
                    print(f"Skipping halo {idx} due to errors")
            
            print(f"Processed {len(result)} halos successfully out of {len(indices)} requested")
            print(f"Shape of result: {np.shape(result)}")
            print(f"Keys of result: {result.keys()}")
            print(f"Result: {result}")
            # Convert dict of arrays to a single ndarray if possible
            # Each value in result is an array of shape (N_i, 3)
            # We'll concatenate along axis=0 to match the return type when index is not None
            if result:
                return np.vstack(list(result.values()))
            else:
                return None
    
    def plot_halo_masses_histogram(self, masslimit=None):
        """
        Plot a histogram of halo masses.
        
        This method creates a histogram of the masses of all halos in the simulation,
        optionally highlighting a specific mass limit.
        
        Parameters
        ----------
        masslimit : float, optional
            If provided, a vertical line will be drawn at this mass value, and
            statistics about halos above/below this limit will be reported.
            
        Returns
        -------
        str
            A string containing information about the mass range and, if masslimit
            is provided, the number of halos below the limit.
            
        Notes
        -----
        The histogram is plotted on a logarithmic scale for better visualization
        of the mass distribution, which often spans several orders of magnitude.
        """
        import matplotlib.pyplot as plt
        
        # Filter out any zero or negative masses before taking log10
        valid_masses = self.halo['m200'][self.halo['m200'] > 0]
        str = ''
        
        if len(valid_masses) == 0:
            print("No valid (positive) masses found in data")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(np.log10(valid_masses), bins=50, color='blue', alpha=0.7)
        if masslimit is not None:
            plt.axvline(np.log10(masslimit), color='red', linestyle='dashed', linewidth=1, label='Mean')
            masses_below_limit = len(valid_masses[valid_masses > masslimit])
            str = f" with {masses_below_limit}/{len(self.halo['m200'])} halos below {masslimit:.2e} M_sun/h"
        plt.xlabel('Halo Mass (log10 M_sun/h)')
        plt.ylabel('Count')
        plt.title('Halo Mass Distribution')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Return minimum and maximum mass for informational purposes
        return f"Mass range: {valid_masses.min():.2e} - {valid_masses.max():.2e} M_sun/h" + str
    
    def init_calculations(self, M200=None, r200=None, c=None, h=None, z=None, Omega_m=None, f=None, verbose=False):
        """
        Initialize the Baryonic Correction Model for a given halo.
        
        This method sets up all the required parameters for the BCM calculation,
        either using values provided as arguments or taken from the loaded halo data.
        After initialization, the calculation is automatically performed.
        
        Parameters
        ----------
        M200 : float, optional
            Halo mass in Msun/h. If None and self.index is set, uses the value from loaded halo data.
        r200 : float, optional
            Halo radius in Mpc/h. If None and self.index is set, uses the value from loaded halo data.
        c : float, optional
            Concentration parameter of the halo. If None, calculated based on M200 and redshift.
        h : float, optional
            Hubble parameter. If None, uses the value from loaded simulation data.
        z : float, optional
            Redshift. If None, uses the value from loaded simulation data.
        Omega_m : float, optional
            Matter density parameter. If None, uses the value from loaded simulation data.
        f : list, dict, or None, optional
            Abundance fractions specification. Can be either:
            - A list of fractions [f_rdm, f_bgas, f_cgal, f_egas]
            - A dictionary with keys 'f_rdm', 'f_bgas', 'f_cgal', 'f_egas'
            - None to calculate fractions based on mass and redshift
        verbose : bool, default False
            Whether to print detailed information during the calculation process.
            
        Returns
        -------
        self : CAMELSReader
            Returns self for method chaining.
            
        Notes
        -----
        This method automatically calls the calculate() method after initialization.
        The abundance fractions must sum to 1.0 with a tolerance of 1e-6.
        """
        
        # FIXED: Set parameters in correct order - M200 and z BEFORE calculating concentration
        
        # First, set M200 and r200 from halo data or input parameters
        if self.index is not None and hasattr(self, 'halo') and self.halo is not None:
            self.M200 = self.halo['m200'][self.index] if M200 is None else M200
            self.r200 = self.halo['r200'][self.index] if r200 is None else r200
        else:
            if M200 is None or r200 is None:
                raise ValueError("M200 and r200 must be provided when no halo data is available")
            self.M200 = M200
            self.r200 = r200
        
        # Set cosmological parameters
        self.h = h if h is not None else getattr(self, 'h', 0.6777)
        self.z = z if z is not None else getattr(self, 'z', 0.0)
        self.Om = Omega_m if Omega_m is not None else getattr(self, 'Om', 0.3071)
        self.Ol = getattr(self, 'Ol', 1 - self.Om)
        self.Ob = getattr(self, 'Ob', par.DEFAULTS['Omega_b'])
        
        # Validate that we have all required parameters
        if self.M200 is None:
            raise ValueError("M200 cannot be None. Please provide M200 or ensure halo data is loaded.")
        if self.r200 is None:
            raise ValueError("r200 cannot be None. Please provide r200 or ensure halo data is loaded.")
        if self.z is None:
            raise ValueError("Redshift (z) cannot be None. Please provide z or ensure simulation data is loaded.")
        
        # NOW calculate concentration (after M200 and z are set)
        if c is None:
            c = ut.calc_concentration(self.M200, self.z)
        self.c = c
        
        # Calculate baryon fraction
        self.fbar = self.Ob / self.Om if hasattr(self, 'Ob') else 0.0483
        
        # Set verbose mode
        self.verbose = verbose
        
        # Derived parameters
        self.r_s = self.r200 / self.c  # Scale radius for NFW profile
        self.r_tr = 8 * self.r200  # Truncation radius
        
        # Set abundance fractions
        self._set_abundance_fractions(f)
        
        #self.write_af()
        
        # Calculate other parameters
        self.r_ej = par.DEFAULTS['r_ej_factor'] * self.r200
        self.R_h = par.DEFAULTS['R_h_factor'] * self.r200
        
        # Initialize component storage
        self.components = {}
        self.r_vals = None
        
        if self.verbose:
            self._print_parameters()
            
        self.calculate()
        
        #return self
    
    def write_af(self, path=None):
        if path is None:
            path = 'halo_data/abundance_fractions.txt'
            
        # Check if the file exists before writing
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            f.write(f"{self.f_rdm:.6f} {self.f_bgas:.6f} {self.f_cgal:.6f} {self.f_egas:.6f}\n")
        
    def _set_abundance_fractions(self, f):
        """
        Set abundance fractions based on input.
        
        Parameters
        ----------
        f : list, dict, or None
            Abundance fractions specification. Can be either:
            - A list of fractions [f_rdm, f_bgas, f_cgal, f_egas]
            - A dictionary with keys 'f_rdm', 'f_bgas', 'f_cgal', 'f_egas'
            - None to calculate fractions based on mass and redshift
            
        Raises
        ------
        ValueError
            If the sum of fractions does not equal 1.0 within tolerance.
            
        Notes
        -----
        Sets class attributes f_rdm, f_bgas, f_cgal, and f_egas.
        """
        # Check if f is list/dict or has to be calculated
        if (isinstance(f, list) and len(f) == 4) or (isinstance(f, dict) and len(f) == 4):
            # Custom abundance fractions
            if self.verbose:
                print("Using fixed abundance fractions.")
            if isinstance(f, dict):
                self.f_rdm = f['f_rdm']
                self.f_bgas = f['f_bgas']
                self.f_cgal = f['f_cgal']
                self.f_egas = f['f_egas']
            else:
                self.f_rdm = f[0]
                self.f_bgas = f[1]
                self.f_cgal = f[2]
                self.f_egas = f[3]
        else:
            # Custom abundance fractions
            if self.verbose:
                print("Using custom abundance fractions.")
            self.f_rdm = af.f_rdm(self.fbar)
            self.f_bgas = af.f_bgas(self.M200, self.fbar)
            self.f_cgal = af.f_cgal(self.M200)
            self.f_egas = af.f_egas(self.f_bgas,self.f_cgal,self.fbar)
        
        # Validate fractions sum to 1.0
        total = self.f_rdm + self.f_bgas + self.f_cgal + self.f_egas
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Abundance fractions do not sum to 1.0 but to {total:.6f}")
    
    def _print_parameters(self):
        """
        Print the model parameters.
        
        This method displays all the relevant BCM parameters including
        halo properties, cosmological parameters, and abundance fractions.
        """
        print(f"BCM with M200 = {self.M200:.2e} Msun/h, r200 = {self.r200:.3f} Mpc/h, "
              f"c = {self.c:.2f}, h = {self.h:.3f}, z = {self.z:.2f}, Omega_m = {self.Om:.3f}, Omega_b = {self.Ob:.3f}, fbar = {self.fbar:.3f}")
        print("Abundance fractions:")
        print(f"  f_rdm  = {self.f_rdm:.3f}")
        print(f"  f_bgas = {self.f_bgas:.3f}")
        print(f"  f_cgal = {self.f_cgal:.3f}")
        print(f"  f_egas = {self.f_egas:.3f}")
        
    def _print_components_at(self,r):
        """
        Print the calculated components at a given radius.
        
        Parameters
        ----------
        r : float
            Radius in Mpc/h at which to display component values.
        """
        print(f"Components at r = {r:.3f} Mpc/h:")
        print(f"  rho_dmo = {dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr):.3e}")
        print(f"  rho_bcm = {self.components['rho_bcm'][r]:.3e}")
        print(f"  rho_bkg = {self.components['rho_bkg'][r]:.3e}")
        print(f"  rho_rdm = {self.components['rdm'][r]:.3e}")
        print(f"  rho_bgas = {self.components['bgas'][r]:.3e}")
        print(f"  rho_egas = {self.components['egas'][r]:.3e}")
        print(f"  rho_cgal = {self.components['cgal'][r]:.3e}")
        
    def _create_radius_array(self, r_min, r_max, n_points):
        """
        Create a radius array for calculations.
        
        This method generates a non-uniform array of radii that has higher
        resolution in the inner regions of the halo.
        
        Parameters
        ----------
        r_min : float
            Minimum radius in Mpc/h.
        r_max : float
            Maximum radius in Mpc/h.
        n_points : int
            Total number of points in the array.
            
        Notes
        -----
        The array uses logarithmic spacing for the inner 70% of points and
        linear spacing for the outer 30%, providing better resolution where
        profiles change more rapidly.
        Sets the class attribute r_vals.
        """
        # Use a combination of log and linear spacing to get more points in the center
        n_log = int(n_points * 0.7)
        n_lin = n_points - n_log

        # Log-spaced points for the inner region
        r_log = np.logspace(np.log10(r_min), np.log10(r_max * 0.1), n_log, endpoint=False)
        # Linearly spaced points for the outer region
        r_lin = np.linspace(r_max * 0.1, r_max, n_lin)

        # Concatenate and ensure uniqueness and sorting
        self.r_vals = np.unique(np.concatenate([r_log, r_lin]))

    def _calc_NFW_target_mass(self):
        """
        Calculate the target mass for the NFW profile.
        
        This method integrates the NFW profile over the entire radius range
        to determine the total mass of the halo.
        
        Returns
        -------
        numpy.ndarray
            Array of cumulative NFW masses at each radius.
            
        Notes
        -----
        Also sets the class attributes rho0 and fixed_M_tot.
        """
        # Integrate NFW profile over a large range to approximate total mass
        self.rho0 = ut.bracket_rho0(self.M200, self.r_s, self.r_tr, self.r200)
        rho_nfw = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) for r in self.r_vals])
        M_nfw = ut.cumul_mass(self.r_vals, rho_nfw)
        M_tot = M_nfw[-1]
        #M_tot2 = dp.mass_nfw_analytical(self.r_vals[-1], self.r_s, self.rho0)
        #print(f"Fixed M_tot: {M_tot:.3e}, M_tot2: {M_tot2:.3e}")
        self.fixed_M_tot = M_tot
        return M_nfw

    def _calculate_normalizations(self):
        """
        Calculate normalization factors for density components.
        
        This method computes the normalization constants needed for each
        component of the BCM to contain the correct fraction of the total mass.
        
        Returns
        -------
        tuple of float
            Normalization constants for (bgas, egas, cgal, rdm).
        """
        norm_bgas = ut.normalize_component_total(
            lambda r, r_s, r200, y0, c, rho0, r_tr: dp.y_bgas(r, r_s, r200, y0, c, rho0, r_tr), 
            (self.r_s, self.r200, 1.0, self.c, self.rho0, self.r_tr), self.f_bgas * self.fixed_M_tot, self.r200
        )
        norm_egas = ut.normalize_component_total(
            lambda r, M_tot, r_ej: dp.y_egas(r, M_tot, r_ej), 
            (1.0, self.r_ej), self.f_egas * self.fixed_M_tot, self.r200
        )
        norm_cgal = ut.normalize_component_total(
            lambda r, M_tot, R_h: dp.y_cgal(r, M_tot, R_h), 
            (1.0, self.R_h), self.f_cgal * self.fixed_M_tot, self.r200
        )
        norm_yrdm_fixed_xi = ut.normalize_component(
            lambda r, r_s, rho0, r_tr, xi: dp.y_rdm_fixed_xi(r, r_s, rho0, r_tr, xi), 
            (self.r_s, self.rho0, self.r_tr, 0.85), self.f_rdm * self.fixed_M_tot, self.r200
        )
        
        return norm_bgas, norm_egas, norm_cgal, norm_yrdm_fixed_xi

    def _calculate_normalizations_old(self):
        """
        Calculate normalizations using old method.
        
        This is a legacy method kept for comparison purposes.
        
        Returns
        -------
        tuple of float
            Normalization constants for (bgas, egas, cgal, rdm).
        """
        norm_bgas = ut.normalize_component_total(
            lambda r, r_s, r200, y0, c, rho0, r_tr: dp.y_bgas(r, r_s, r200, y0, c, rho0, r_tr), 
            (self.r_s, self.r200, 1.0, self.c, self.rho0, self.r_tr), self.f_bgas * self.M200, self.r200
        )
        norm_egas = ut.normalize_component_total(
            lambda r, M_tot, r_ej: dp.y_egas(r, M_tot, r_ej), 
            (1.0, self.r_ej), self.f_egas * self.M200, self.r200
        )
        norm_cgal = ut.normalize_component_total(
            lambda r, M_tot, R_h: dp.y_cgal(r, M_tot, R_h), 
            (1.0, self.R_h), self.f_cgal * self.M200, self.r200
        )
        norm_rdm = ut.normalize_component_total(
            lambda r, r_s, rho0, r_tr, xi: dp.y_rdm(r, r_s, rho0, r_tr, xi), 
            (self.r_s, self.rho0, self.r_tr, 0.85), self.M200, self.r200
        )
        return norm_bgas, norm_egas, norm_cgal, norm_rdm

    def normalize_cgal_special(density_func, args, M_target, R_h):
        """
        Special normalization for central galaxy component.
        
        This method uses a much larger integration radius adapted to the
        Hernquist profile to ensure proper mass normalization.
        
        Parameters
        ----------
        density_func : callable
            The density profile function to normalize.
        args : tuple
            Arguments to pass to the density function.
        M_target : float
            Target mass for the component in Msun/h.
        R_h : float
            Characteristic radius of the Hernquist profile in Mpc/h.
            
        Returns
        -------
        float
            Normalization factor to apply to the density profile.
        """
        def unnorm_func(r):
            return density_func(r, 1.0, *args)
        
        # Use a much larger radius that adapts to the concentration
        r_max = 1000 * R_h  # This captures essentially all mass for a Hernquist profile
        unnorm_mass = dp.mass_profile(r_max, unnorm_func)
        
        return M_target / unnorm_mass
    
    def _compute_density_profiles(self, norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi, M_nfw):
        """
        Compute density profiles for all components.
        
        This method calculates the density profiles for each BCM component
        using the provided normalization constants.
        
        Parameters
        ----------
        norm_bgas : float
            Normalization constant for baryonic gas.
        norm_egas : float
            Normalization constant for ejected gas.
        norm_cgal : float
            Normalization constant for central galaxy.
        norm_rdm_fixed_xi : float
            Normalization constant for remaining dark matter.
        M_nfw : numpy.ndarray
            Cumulative mass profile of the NFW halo.
            
        Returns
        -------
        tuple
            All density profiles: (rho_dmo, rho_nfw, rho_bkg, y_bgas, y_egas, y_cgal, y_rdm, rho_bcm).
            
        Notes
        -----
        Also sets the class attribute profiles with all calculated profiles.
        """
        rho_dmo_vals = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) + 
                               dp.rho_background(r, 1) for r in self.r_vals])
        rho_nfw_vals = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) 
                               for r in self.r_vals])
        rho_bkg_vals = np.array([dp.rho_background(r, 1) for r in self.r_vals])
        
        y_bgas_vals = np.array([dp.y_bgas(r, self.r_s, self.r200, norm_bgas, self.c, self.rho0, self.r_tr) 
                              for r in self.r_vals])
        y_egas_vals = np.array([dp.y_egas(r, norm_egas, self.r_ej) 
                              for r in self.r_vals])
        y_cgal_vals = np.array([dp.y_cgal(r, norm_cgal, self.R_h) 
                              for r in self.r_vals])
        y_rdm_vals_fixed_xi = np.array([dp.y_rdm_fixed_xi(r, self.r_s, self.rho0, self.r_tr, norm_rdm_fixed_xi)
                                for r in self.r_vals])
        
        y_bgas_vals, y_egas_vals, y_cgal_vals,y_rdm_vals_fixed_xi = self.correction_factors_baryons(
            [self.f_bgas, self.f_egas, self.f_cgal,self.f_rdm], 
            [y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals_fixed_xi]
        )
        
        # baryon components for xi
        # Note: y_bgas, y_egas, and y_cgal are already normalized
        baryons = [(self.r_vals, y_bgas_vals), 
           (self.r_vals, y_egas_vals), 
           (self.r_vals, y_cgal_vals)]
        
        # Calculate unnormalized profile
        rho_dm_contracted = dp.y_rdm_ac(self.r_vals, self.r_s, self.rho0, self.r_tr, 
                                    norm=1.0, a=0.68, f_cdm=0.839, 
                                    baryon_components=baryons, verbose=self.verbose)

        # Calculate total mass and correction factor
        M_contracted = ut.cumul_mass(self.r_vals, rho_dm_contracted)[-1]
        target_mass = self.f_rdm * self.fixed_M_tot
        correction_factor = target_mass / M_contracted
        
        if self.verbose:
            print(f"RDM mass correction factor: {correction_factor:.4f}")

        # Apply correction
        rho_dm_contracted *= correction_factor
        y_rdm_vals = rho_dm_contracted
        #y_rdm_vals = y_rdm_vals_fixed_xi

        rho_bcm = y_rdm_vals + y_bgas_vals + y_egas_vals + y_cgal_vals + rho_bkg_vals
        self.profiles = {
            'rho_dmo': rho_dmo_vals,
            'rho_nfw': rho_nfw_vals,
            'rho_bkg': rho_bkg_vals,
            'y_bgas': y_bgas_vals,
            'y_egas': y_egas_vals,
            'y_cgal': y_cgal_vals,
            'y_rdm': y_rdm_vals,
            'rho_bcm': rho_bcm
        }
        return rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm

    def correction_factors_baryons(self, fractions, profiles):
        """
        Apply mass correction factors to baryonic components.
        
        This method adjusts the density profiles to ensure each component
        contains the exact fraction of the total mass.
        
        Parameters
        ----------
        fractions : list of float
            List of mass fractions [f_rdm, f_bgas, f_cgal, f_egas].
        profiles : list of numpy.ndarray
            List of density profiles [rdm, bgas, egas, cgal].
            
        Returns
        -------
        list of numpy.ndarray
            Corrected density profiles.
        """
        cor_profiles = []
        for i in range(len(fractions)):
            mass = ut.cumul_mass(self.r_vals, profiles[i])[-1]
            correction = (fractions[i] * self.fixed_M_tot) / mass
            cor_profiles.append(correction*profiles[i])
        return cor_profiles

    def _compute_mass_profiles(self, rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, M_nfw):
        """
        Compute cumulative mass profiles for all components.
        
        This method integrates the density profiles to obtain the enclosed
        mass as a function of radius for each component.
        
        Parameters
        ----------
        rho_dmo_vals : numpy.ndarray
            DMO density profile.
        rho_bkg_vals : numpy.ndarray
            Background density profile.
        y_rdm_vals : numpy.ndarray
            Remaining dark matter density profile.
        y_bgas_vals : numpy.ndarray
            Baryonic gas density profile.
        y_egas_vals : numpy.ndarray
            Ejected gas density profile.
        y_cgal_vals : numpy.ndarray
            Central galaxy density profile.
        rho_bcm : numpy.ndarray
            Total BCM density profile.
        M_nfw : numpy.ndarray
            Cumulative NFW mass profile.
            
        Returns
        -------
        tuple
            All mass profiles: (M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm).
            
        Notes
        -----
        Also sets the class attribute masses with all calculated mass profiles.
        Calls _check_masses to validate the results and _print_masses_at_infinity
        if verbose is True.
        """
        M_dmo = ut.cumul_mass(self.r_vals, rho_dmo_vals)
        M_bkg = ut.cumul_mass(self.r_vals, rho_bkg_vals)
        M_rdm = ut.cumul_mass(self.r_vals, y_rdm_vals)
        M_bgas = ut.cumul_mass(self.r_vals, y_bgas_vals)
        M_egas = ut.cumul_mass(self.r_vals, y_egas_vals)
        M_cgal = ut.cumul_mass(self.r_vals, y_cgal_vals)
        M_bcm = ut.cumul_mass(self.r_vals, rho_bcm)
        self.masses = {
            'M_dmo': M_dmo,
            'M_bkg': M_bkg,
            'M_rdm': M_rdm,
            'M_bgas': M_bgas,
            'M_egas': M_egas,
            'M_cgal': M_cgal,
            'M_bcm': M_bcm,
            'M_nfw': M_nfw
        }
        
#        self._check_masses()
        if self.verbose:
            self._print_masses_at_infinity()
        return M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm

    def _check_masses(self):
        """
        Validate that mass profiles have the expected values.
        
        This method checks that the mass of each component equals the expected
        fraction of the total mass within a specified tolerance.
        
        Raises
        ------
        ValueError
            If any component's mass doesn't match its expected value.
            
        Notes
        -----
        Uses a relative tolerance of 1%.
        """
        tol = 1e-2  # Tolerance for mass comparison
        if not np.isclose(self.masses['M_bgas'][-1], self.f_bgas * self.fixed_M_tot, rtol=tol):
            raise ValueError(f"Mass of baryons does not match: {self.masses['M_bgas'][-1]:.3e} != {self.f_bgas * self.fixed_M_tot:.3e}")
        if not np.isclose(self.masses['M_egas'][-1], self.f_egas * self.fixed_M_tot, rtol=tol):
            raise ValueError(f"Mass of baryons does not match: {self.masses['M_egas'][-1]:.3e} != {self.f_egas * self.fixed_M_tot:.3e}")
        if not np.isclose(self.masses['M_cgal'][-1], self.f_cgal * self.fixed_M_tot, rtol=tol):
            raise ValueError(f"Mass of baryons does not match: {self.masses['M_cgal'][-1]:.3e} != {self.f_cgal * self.fixed_M_tot:.3e}")
        """if not np.isclose(self.masses['M_rdm'][-1], self.f_rdm * self.fixed_M_tot, rtol=tol):
            raise ValueError(f"Mass of baryons does not match: {self.masses['M_rdm'][-1]:.3e} != {self.f_rdm * self.fixed_M_tot:.3e}")"""
        total_bcm = self.masses['M_bgas'][-1] + self.masses['M_egas'][-1] + self.masses['M_cgal'][-1] + self.masses['M_rdm'][-1]
        if not np.isclose(self.masses['M_nfw'][-1], self.fixed_M_tot, rtol=tol):
            raise ValueError(f"Mass of baryons does not match: {self.masses['M_nfw'][-1]:.3e} != {self.fixed_M_tot:.3e}")
    
    def _invert_mass_profile(self, M_bcm):
        """
        Create an interpolator to find radius given a mass.
        
        Parameters
        ----------
        M_bcm : numpy.ndarray
            BCM cumulative mass profile.
            
        Returns
        -------
        callable
            Function that returns radius given a mass value.
        """
        from scipy.interpolate import interp1d
        return interp1d(M_bcm, self.r_vals, bounds_error=False, fill_value="extrapolate")

    def _compute_displacement(self, M_dmo, f_inv_bcm):
        """
        Compute displacement field for particles.
        
        This method calculates how much each particle needs to be moved
        to transform the DMO density profile into the BCM profile.
        
        Parameters
        ----------
        M_dmo : numpy.ndarray
            DMO cumulative mass profile.
        f_inv_bcm : callable
            Function that returns radius given a mass value.
            
        Returns
        -------
        numpy.ndarray
            Displacement values at each radius.
        """
        disp = np.zeros_like(self.r_vals)
        for i, r in enumerate(self.r_vals):
            M_target = M_dmo[i]
            r_bcm_val = f_inv_bcm(M_target)
            disp[i] = r_bcm_val - r
        return disp
    
    def _print_masses_at_infinity(self):
        """
        Print the masses of all components at the maximum calculated radius.
        
        This method displays the asymptotic mass values for each component
        and their fractions of the total mass.
        
        Raises
        ------
        ValueError
            If the radius array hasn't been created yet.
        """
        if self.r_vals is None:
            raise ValueError("Radius array not created. Call calculate() first.")
        r_infinity_index = -1
        print(f"Masses at infinity:")
        sum_of_masses = np.sum([
            self.masses['M_rdm'][r_infinity_index],
            self.masses['M_bgas'][r_infinity_index],
            self.masses['M_egas'][r_infinity_index],
            self.masses['M_cgal'][r_infinity_index],
        ])
        masses = {
            'M_rdm': self.masses['M_rdm'],
            'M_bgas': self.masses['M_bgas'],
            'M_egas': self.masses['M_egas'],
            'M_cgal': self.masses['M_cgal'],
            'M_nfw': self.masses['M_nfw'],
        }
        print(f"  M_inf is: {self.fixed_M_tot:.3e},   the sums are: {sum_of_masses:.3e}")
        for key, value in masses.items():
            print(f"  {key}: {value[r_infinity_index]:.3e}, fraction: {value[r_infinity_index]/self.fixed_M_tot:.3f}")
    
    def _calculate_rdm(self, M_i, M_b):
        """
        Calculate the RDM profile using the adiabatic contraction method.
        
        Parameters
        ----------
        M_i : numpy.ndarray
            Initial mass profile.
        M_b : numpy.ndarray
            Baryonic mass profile.
            
        Returns
        -------
        tuple
            (rho_rdm, M_rdm) - density and mass profiles for RDM.
            
        Raises
        ------
        ValueError
            If the calculated mass doesn't match the expected value.
        """
        f_cdm = M_i/(M_i + M_b)
        M_f = f_cdm * M_i + M_b
        if np.isclose(M_f[-1], self.fixed_M_tot, atol=1e-2):
            raise ValueError(f"M_f + M_b != M_tot: {M_f[-1]*self.f_rdm + M_b[-1]} != {self.fixed_M_tot}\n ratio M_f: {M_f[-1]*self.f_rdm / self.fixed_M_tot} \n ratio M_b: {M_b[-1] / self.fixed_M_tot}")
        # Calculate the RDM profile
        rho_rdm = dp.y_rdm_ac2(self.r_vals, self.r_s, self.rho0, self.r_tr, M_i, M_f, self.verbose)
        
        # Calculate the total mass and correction factor
        M_contracted = ut.cumul_mass(self.r_vals, rho_rdm)[-1]
        target_mass = self.f_rdm * self.fixed_M_tot
        correction_factor = target_mass / M_contracted
        
        # Apply correction
        rho_rdm *= correction_factor
        
        # Calculate the total mass profile
        M_rdm = ut.cumul_mass(self.r_vals, rho_rdm)
        
        return rho_rdm, M_rdm
    
    def print_components(self, r = None):
        """
        Print all component values at a specific radius.
        
        This method displays the density values of all components at the
        specified radius, along with key halo parameters.
        
        Parameters
        ----------
        r : float, optional
            Radius in Mpc/h. If None, uses the maximum calculated radius.
            
        Raises
        ------
        ValueError
            If the radius array hasn't been created yet or if the specified
            radius is out of bounds.
        """
        if self.r_vals is None:
            raise ValueError("Radius array not created. Call calculate() first.")
        
        if r is None:
            r = self.r_vals[-1]
        if r < self.r_vals[0] or r > self.r_vals[-1]:
            raise ValueError(f"Radius {r} is out of bounds. Must be between {self.r_vals[0]} and {self.r_vals[-1]}.")
        r_index = np.searchsorted(self.r_vals, r)
        print(f"Components at r = {r:.3f} Mpc/h:")
        print(f"  rho_dmo = {self.components['rho_dmo'][r_index]:.3e}")
        print(f"  rho_bcm = {self.components['rho_bcm'][r_index]:.3e}")
        print(f"  rho_bkg = {self.components['rho_bkg'][r_index]:.3e}")
        print(f"  rho_rdm = {self.components['rdm'][r_index]:.3e}")
        print(f"  rho_bgas = {self.components['bgas'][r_index]:.3e}")
        print(f"  rho_egas = {self.components['egas'][r_index]:.3e}")
        print(f"  rho_cgal = {self.components['cgal'][r_index]:.3e}")
        print(f"  M200 = {self.M200:.3e}")
        print(f"  r200 = {self.r200:.3f}")  
        print(f"  r_s = {self.r_s:.3f}")
        print(f"  rho0 = {self.rho0:.3e}")
        print(f"  r_ej = {self.r_ej:.3f}")
        print(f"  R_h = {self.R_h:.3f}")
        print(f"  f_rdm = {self.f_rdm:.3f}")
        print(f"  f_bgas = {self.f_bgas:.3f}")
        print(f"  f_cgal = {self.f_cgal:.3f}")
        print(f"  f_egas = {self.f_egas:.3f}")
        print(f"  rho0 = {self.rho0:.3e}")
    
    def calculate(self, r_min=0.001, r_max=None, n_points=1000):
        """
        Calculate all BCM profiles and properties.
        
        This method performs the complete BCM calculation, computing density
        profiles, mass profiles, and displacement fields.
        
        Parameters
        ----------
        r_min : float, default 0.001
            Minimum radius in Mpc/h for calculations.
        r_max : float, optional
            Maximum radius in Mpc/h. If None, uses 10000 times r200.
        n_points : int, default 1000
            Number of radius points for calculations.
            
        Returns
        -------
        self : CAMELSReader
            Returns self for method chaining.
            
        Notes
        -----
        Results are stored in the components dictionary attribute.
        """
        if r_max is None:
            r_max = 10000 * self.r200 
        # Create a radius array
        self._create_radius_array(r_min, r_max, n_points)
        
        M_nfw = self._calc_NFW_target_mass()
        
        # Calculate normalizations
        norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi = self._calculate_normalizations()
        if self.verbose:
            print(f"Component normalizations to contain M200:")
            print(f"  bgas: {norm_bgas:.3e}")
            print(f"  egas: {norm_egas:.3e}")
            print(f"  cgal: {norm_cgal:.3e}")
        
        # Calculate density profiles
        rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm = self._compute_density_profiles(norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi, M_nfw)
        
        # Calculate mass profiles
        M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm = self._compute_mass_profiles(rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, M_nfw)
        
        #M_b = M_bgas + M_egas + M_cgal
        #M_i = M_nfw
        # Calculate RDM profile
        #y_rdm_vals2, M_rdm2 = self._calculate_rdm(M_i, M_b)
        
        # Calculate displacement
        f_inv_bcm = self._invert_mass_profile(M_bcm)
        disp = self._compute_displacement(M_dmo, f_inv_bcm)
        
        # Store results in the components dictionary
        self.components = {
            'M200': self.M200,
            'r200': self.r200,
            'r_s': self.r_s,
            'rho_dmo': rho_dmo_vals,
            'rho_bcm': rho_bcm,
            'rho_bkg': rho_bkg_vals,
            'rdm': y_rdm_vals,
            'bgas': y_bgas_vals,
            'egas': y_egas_vals,
            'cgal': y_cgal_vals,
            'M_dmo': M_dmo,
            'M_rdm': M_rdm,
            'M_bgas': M_bgas,
            'M_egas': M_egas,
            'M_cgal': M_cgal,
            'M_bcm': M_bcm,
            'M_bkg': M_bkg,
            'M_nfw': M_nfw,
            'disp': disp
        }
        
        return self
    
    def apply_displacement(self, particles=None):
        """
        Apply calculated displacements to particles.
        
        This method moves particles according to the displacement field
        calculated by the BCM to transform the DMO density into the BCM density.
        
        Parameters
        ----------
        particles : list or dict, optional
            Particle data to use. If None, uses self.particles.
            
        Returns
        -------
        numpy.ndarray or None
            Array of new particle positions. None if there's an error.
            
        Raises
        ------
        ValueError
            If the radius array hasn't been created yet.
            
        Notes
        -----
        Particles are displaced radially, preserving their angular positions.
        """
        if particles is None:
            particles = self.particles
        if self.r_vals is None:
            raise ValueError("Radius array not created. Call calculate() first.")
        displaced_positions = []
        particles_per_halo =[]

        # Get the displacement for each particle in each halo
        for halo in tqdm(self.halo['id'], desc="Applying displacement"):
            try:
                rel_pos = self.get_particles_relative_position(halo)
            except Exception as e:
                print(f"Error calculating relative positions: {e}")
                return None

            r = np.linalg.norm(rel_pos, axis=1)
            disp = np.interp(r, self.r_vals, self.components['disp'])

            # Avoid division by zero for particles at the center
            with np.errstate(invalid='ignore', divide='ignore'):
                direction = np.zeros_like(rel_pos)
                mask = r > 0
                direction[mask] = rel_pos[mask] / r[mask, np.newaxis]

            # Apply the displacement along the radial direction
            new_rel_pos = rel_pos + direction * disp[:, np.newaxis]

            # Shift back to absolute coordinates
            center = self.get_halo_center(halo)
            displaced_positions.append(new_rel_pos + center)
            theorectical_particle_count = self.halo['lentype_h'][halo][1]
            particles_per_halo.append(theorectical_particle_count)
            if theorectical_particle_count != len(rel_pos):
                print(f"Warning: Theoretical particle count {theorectical_particle_count} does not match actual count {len(rel_pos)} for halo {halo}")
                return None
        # Concatenate all displaced positions into a single array
        if len(displaced_positions) > 0:
            new_pos = np.vstack(displaced_positions)
        else:
            new_pos = np.array([])
        print(f"Displaced positions shape: {new_pos.shape}")
        #print(f"Particles per halo: {particles_per_halo}")
        print(f"Total number of particles (per halo): {np.sum(particles_per_halo)}")
        print(f"Total number of displaced positions: {len(new_pos)}")
        return new_pos
    
    def plot_density_profiles(self):
        """
        Plot the density profiles of all components.
        
        This method creates a log-log plot showing the density profiles
        of all BCM components and the DMO profile for comparison.
        
        Raises
        ------
        ValueError
            If the radius array hasn't been created yet.
        """
        if self.r_vals is None:
            raise ValueError("Radius array not created. Call calculate() first.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.r_vals, self.components['rho_dmo'], label='DMO', color='blue')
        plt.plot(self.r_vals, self.components['rho_bcm'], label='BCM', color='orange')
        plt.plot(self.r_vals, self.components['bgas'], label='Baryonic Gas', color='green')
        plt.plot(self.r_vals, self.components['egas'], label='Ejected Gas', color='red')
        plt.plot(self.r_vals, self.components['cgal'], label='Central Galaxy', color='purple')
        plt.plot(self.r_vals, self.components['rdm'], label='RDM', color='brown')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Radius (Mpc/h)')
        plt.ylabel('Density (Msun/h/Mpc^3)')
        plt.title('Density Profiles of BCM Components')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_displacement(self):
        """
        Plot the displacement field.
        
        This method creates a semi-log plot showing how particle displacement
        varies with radius.
        
        Raises
        ------
        ValueError
            If the radius array hasn't been created yet.
        """
        if self.r_vals is None:
            raise ValueError("Radius array not created. Call calculate() first.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.r_vals, self.components['disp'], label='Displacement', color='blue')
        plt.xscale('log')
        plt.xlabel('Radius (Mpc/h)')
        plt.ylabel('Displacement (Mpc/h)')
        plt.title('Displacement of Particles')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calc_displ_and_compare_powerspectrum(self, output_file=None):
        """
        Calculate power spectra for DMO and BCM particle distributions.
        
        This method computes the displacement field, applies it to particles,
        and calculates power spectra for both the original and displaced particles.
        
        Parameters
        ----------
        output_file : str, optional
            If provided, results will be saved to this file.
            
        Returns
        -------
        tuple or None
            (k_dmo, Pk_dmo, k_bcm, Pk_bcm) - wavenumbers and power spectra.
            None if there's an error in processing.
        """
        # Get particles associated with halos only
        total_particles = 0
        all_positions = []
        
        # Process each halo to get original positions
        for halo in self.halo['id']:
            particles = self.get_halo_particles(halo)
            if particles is not None and 'pos' in particles:
                all_positions.append(particles['pos'])
                total_particles += len(particles['pos'])
        
        # Combine all positions
        if len(all_positions) > 0:
            dmo_positions = np.vstack(all_positions)
        else:
            print("No particles found in halos")
            return None
        
        # Get displaced positions
        bcm_positions = self.apply_displacement()
        if bcm_positions is None:
            print("Error: Could not apply displacement")
            return None
        
        k_dmo, Pk_dmo, k_bcm, Pk_bcm = ut.compare_power_spectra(dmo_positions, bcm_positions, self.boxsize, output_file)
        
        return k_dmo, Pk_dmo, k_bcm, Pk_bcm
    
    def calc_power_spectrum(self):
        """
        Calculate and plot the power spectrum of particles.
        
        This method computes the power spectrum of the particle distribution
        and displays it.
        
        Returns
        -------
        None
        
        Notes
        -----
        Uses the utility function calc_power_spectrum for computation and
        plot_power_spectrum for visualization.
        """
        particles_dict = self.get_halo_particles()
        if particles_dict is None:
            print("Error: Could not get halo particles")
            return None
        
        # Extract just the position arrays from the dictionaries
        dmo_positions = particles_dict['pos']
        
        # Get displaced positions

        k, Pk = ut.calc_power_spectrum(dmo_positions, self.boxsize)
        
        ut.plot_power_spectrum(k, Pk)
    
    def calculate_displacement_all_halos(self, min_mass=1e9, max_halos=None, save_individual=True, output_dir="displacement_results"):
        """
        Calculate displacement for all halos in the simulation and save to CSV files.
    
        Parameters
        ----------
        min_mass : float, default 1e12
            Minimum halo mass to process (Msun/h)
        max_halos : int, optional
            Maximum number of halos to process
        save_individual : bool, default True
            Whether to save individual halo displacement data to CSV
        output_dir : str, default "displacement_results"
            Directory to save CSV results
    
        Returns
        -------
        dict
            Dictionary containing displacement results for all processed halos
        """
        import os
        import pandas as pd
    
        if save_individual:
            os.makedirs(output_dir, exist_ok=True)
    
        # Filter halos
        valid_halos = [i for i, mass in enumerate(self.halo['m200']) if mass >= min_mass]
    
        if max_halos is not None:
            mass_indices = sorted(valid_halos, key=lambda x: self.halo['m200'][x], reverse=True)
            valid_halos = mass_indices[:max_halos]
    
        print(f"Processing {len(valid_halos)} halos for displacement...")
    
        # Store original state
        original_index = self.index
    
        # Storage for CSV data
        halo_summary_data = []
        displacement_profiles_data = []
        density_profiles_data = []
        mass_profiles_data = []
        bcm_parameters_data = []
    
        try:
            for halo_idx in tqdm(valid_halos, desc="Calculating displacements"):
                try:
                    # Set current halo
                    self.index = halo_idx
    
                    # Get halo properties
                    M200 = self.halo['m200'][halo_idx]
                    r200 = self.halo['r200'][halo_idx]
    
                    # Initialize BCM
                    self.init_calculations(
                        M200=M200, r200=r200, c=None,
                        h=self.h, z=self.z, Omega_m=self.Om,
                        f=None, verbose=self.verbose,
                    )
    
                    # Collect halo summary data
                    halo_summary = {
                        'halo_id': halo_idx,
                        'M200': M200,
                        'r200': r200,
                        'z': self.z,
                        'h': self.h,
                        'Omega_m': self.Om,
                        'Omega_b': self.Ob,
                        'boxsize': self.boxsize,
                        'time': getattr(self, 'time', 0.0),
                        'processing_timestamp': pd.Timestamp.now()
                    }
                    halo_summary_data.append(halo_summary)
    
                    # Collect BCM parameters
                    bcm_params = {
                        'halo_id': halo_idx,
                        'c': self.c,
                        'r_s': self.r_s,
                        'rho0': self.rho0,
                        'f_rdm': self.f_rdm,
                        'f_bgas': self.f_bgas,
                        'f_cgal': self.f_cgal,
                        'f_egas': self.f_egas,
                        'fbar': self.fbar,
                        'r_ej': self.r_ej,
                        'R_h': self.R_h,
                        'fixed_M_tot': self.fixed_M_tot
                    }
                    bcm_parameters_data.append(bcm_params)
    
                    # Collect profile data for each radius point
                    n_points = len(self.r_vals)
                    
                    for j in range(n_points):
                        # Displacement profile
                        disp_row = {
                            'halo_id': halo_idx,
                            'r_index': j,
                            'r_val': self.r_vals[j],
                            'displacement': self.components['disp'][j]
                        }
                        displacement_profiles_data.append(disp_row)
    
                        # Density profiles
                        density_row = {
                            'halo_id': halo_idx,
                            'r_index': j,
                            'r_val': self.r_vals[j],
                            'rho_dmo': self.components['rho_dmo'][j],
                            'rho_bcm': self.components['rho_bcm'][j],
                            'rho_bkg': self.components['rho_bkg'][j],
                            'rho_rdm': self.components['rdm'][j],
                            'rho_bgas': self.components['bgas'][j],
                            'rho_egas': self.components['egas'][j],
                            'rho_cgal': self.components['cgal'][j]
                        }
                        density_profiles_data.append(density_row)
    
                        # Mass profiles
                        mass_row = {
                            'halo_id': halo_idx,
                            'r_index': j,
                            'r_val': self.r_vals[j],
                            'M_dmo': self.components['M_dmo'][j],
                            'M_bcm': self.components['M_bcm'][j],
                            'M_rdm': self.components['M_rdm'][j],
                            'M_bgas': self.components['M_bgas'][j],
                            'M_egas': self.components['M_egas'][j],
                            'M_cgal': self.components['M_cgal'][j],
                            'M_bkg': self.components['M_bkg'][j],
                            'M_nfw': self.components['M_nfw'][j]
                        }
                        mass_profiles_data.append(mass_row)
    
                except Exception as e:
                    print(f"Warning: Error processing halo {halo_idx}: {e}")
                    continue
    
        finally:
            # Restore original state
            self.index = original_index
    
        # Convert to DataFrames and save CSV files
        if save_individual and halo_summary_data:
            
            # Save halo summary
            df_summary = pd.DataFrame(halo_summary_data)
            summary_file = os.path.join(output_dir, "halo_summary.csv")
            df_summary.to_csv(summary_file, index=False)
            print(f"Saved halo summary to: {summary_file}")
    
            # Save BCM parameters
            df_bcm_params = pd.DataFrame(bcm_parameters_data)
            bcm_params_file = os.path.join(output_dir, "bcm_parameters.csv")
            df_bcm_params.to_csv(bcm_params_file, index=False)
            print(f"Saved BCM parameters to: {bcm_params_file}")
    
            # Save displacement profiles
            df_displacement = pd.DataFrame(displacement_profiles_data)
            displacement_file = os.path.join(output_dir, "displacement_profiles.csv")
            df_displacement.to_csv(displacement_file, index=False)
            print(f"Saved displacement profiles to: {displacement_file}")
    
            # Save density profiles
            df_density = pd.DataFrame(density_profiles_data)
            density_file = os.path.join(output_dir, "density_profiles.csv")
            df_density.to_csv(density_file, index=False)
            print(f"Saved density profiles to: {density_file}")
    
            # Save mass profiles
            df_mass = pd.DataFrame(mass_profiles_data)
            mass_file = os.path.join(output_dir, "mass_profiles.csv")
            df_mass.to_csv(mass_file, index=False)
            print(f"Saved mass profiles to: {mass_file}")
    
            # Create and save analysis summary
            analysis_summary = {
                'total_halos_processed': len(df_summary),
                'mass_range_min': df_summary['M200'].min(),
                'mass_range_max': df_summary['M200'].max(),
                'redshift': df_summary['z'].iloc[0] if len(df_summary) > 0 else 0.0,
                'h_param': df_summary['h'].iloc[0] if len(df_summary) > 0 else 0.6777,
                'omega_m': df_summary['Omega_m'].iloc[0] if len(df_summary) > 0 else 0.3071,
                'omega_b': df_summary['Omega_b'].iloc[0] if len(df_summary) > 0 else 0.048,
                'boxsize': df_summary['boxsize'].iloc[0] if len(df_summary) > 0 else 25.0,
                'mean_f_rdm': df_bcm_params['f_rdm'].mean(),
                'std_f_rdm': df_bcm_params['f_rdm'].std(),
                'mean_f_bgas': df_bcm_params['f_bgas'].mean(),
                'std_f_bgas': df_bcm_params['f_bgas'].std(),
                'mean_f_cgal': df_bcm_params['f_cgal'].mean(),
                'std_f_cgal': df_bcm_params['f_cgal'].std(),
                'mean_f_egas': df_bcm_params['f_egas'].mean(),
                'std_f_egas': df_bcm_params['f_egas'].std(),
                'mean_concentration': df_bcm_params['c'].mean(),
                'std_concentration': df_bcm_params['c'].std(),
                'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'min_mass_threshold': min_mass,
                'max_halos_limit': max_halos
            }
    
            # Save analysis summary
            df_analysis = pd.DataFrame([analysis_summary])
            analysis_file = os.path.join(output_dir, "analysis_summary.csv")
            df_analysis.to_csv(analysis_file, index=False)
            print(f"Saved analysis summary to: {analysis_file}")
    
            # Save a detailed text summary
            summary_text_file = os.path.join(output_dir, "displacement_summary.txt")
            with open(summary_text_file, 'w') as f:
                f.write("Displacement Analysis Summary\n")
                f.write("============================\n\n")
                f.write(f"Processing Date: {analysis_summary['processing_date']}\n")
                f.write(f"Total Halos Processed: {analysis_summary['total_halos_processed']}\n")
                f.write(f"Mass Range: {analysis_summary['mass_range_min']:.2e} - {analysis_summary['mass_range_max']:.2e} Msun/h\n")
                f.write(f"Redshift: {analysis_summary['redshift']:.3f}\n")
                f.write(f"Cosmological Parameters:\n")
                f.write(f"  h = {analysis_summary['h_param']:.4f}\n")
                f.write(f"  Omega_m = {analysis_summary['omega_m']:.4f}\n")
                f.write(f"  Omega_b = {analysis_summary['omega_b']:.4f}\n")
                f.write(f"  Boxsize = {analysis_summary['boxsize']:.1f} Mpc/h\n\n")
                f.write(f"Mean BCM Parameters:\n")
                f.write(f"  f_rdm:  {analysis_summary['mean_f_rdm']:.4f} ± {analysis_summary['std_f_rdm']:.4f}\n")
                f.write(f"  f_bgas: {analysis_summary['mean_f_bgas']:.4f} ± {analysis_summary['std_f_bgas']:.4f}\n")
                f.write(f"  f_cgal: {analysis_summary['mean_f_cgal']:.4f} ± {analysis_summary['std_f_cgal']:.4f}\n")
                f.write(f"  f_egas: {analysis_summary['mean_f_egas']:.4f} ± {analysis_summary['std_f_egas']:.4f}\n")
                f.write(f"  c:      {analysis_summary['mean_concentration']:.4f} ± {analysis_summary['std_concentration']:.4f}\n\n")
                f.write(f"Files Generated:\n")
                f.write(f"  - halo_summary.csv: Basic halo properties\n")
                f.write(f"  - bcm_parameters.csv: BCM model parameters for each halo\n")
                f.write(f"  - displacement_profiles.csv: Displacement vs radius for each halo\n")
                f.write(f"  - density_profiles.csv: Density profiles for all components\n")
                f.write(f"  - mass_profiles.csv: Mass profiles for all components\n")
                f.write(f"  - analysis_summary.csv: Overall statistics\n")
    
            print(f"Saved detailed summary to: {summary_text_file}")
    
            # Return summary data for further analysis
            return {
                'summary': df_summary,
                'bcm_parameters': df_bcm_params,
                'displacement_profiles': df_displacement,
                'density_profiles': df_density,
                'mass_profiles': df_mass,
                'analysis_summary': analysis_summary
            }
    
        else:
            print("No valid halo data to save!")
            return {}
    
        print(f"Displacement calculation completed for {len(halo_summary_data)} halos")
        print(f"All results saved to directory: {output_dir}")
    
        # ... your existing code ...
        
        print("="*50)
        print("TESTING ABUNDANCE FRACTIONS")
        print("="*50)
        test_g_func_and_fractions()
        print("="*50)
        
        # ... rest of your code ...
        
    def calc_displ_and_compare_powerspectrum_from_csv(self, csv_dir="halo_data/displacement_all_halos", output_file=None):
        """
        Calculate power spectra using pre-computed displacement data from CSV files.
        
        This method reads displacement profiles from CSV files generated by 
        calculate_displacement_all_halos() and applies them to compute power spectra.
        
        Parameters
        ----------
        csv_dir : str, default "displacement_results"
            Directory containing CSV files with displacement data
        output_file : str, optional
            If provided, results will be saved to this file
            
        Returns
        -------
        tuple or None
            (k_dmo, Pk_dmo, k_bcm, Pk_bcm) - wavenumbers and power spectra
        """
        import pandas as pd
        import os
        
        # Check if CSV files exist
        disp_file = os.path.join(csv_dir, "displacement_profiles.csv")
        if not os.path.exists(disp_file):
            print(f"Error: Displacement file not found: {disp_file}")
            print("Run calculate_displacement_all_halos() first!")
            return None
        
        # Load displacement data
        df_disp = pd.read_csv(disp_file)
        
        print("Loading displacement data from CSV...")
        
        # Get original particle positions for all halos
        all_dmo_positions = []
        all_bcm_positions = []
        
        # Process each halo
        for halo_id in tqdm(df_disp['halo_id'].unique(), desc="Applying displacements from CSV"):
            try:
                # Get particles for this halo
                particles = self.get_halo_particles(halo_id)
                if particles is None or 'pos' not in particles:
                    continue
                    
                # Get halo center
                center = self.get_halo_center(halo_id)
                rel_pos = particles['pos'] - center
                r = np.linalg.norm(rel_pos, axis=1)
                
                # Get displacement profile for this halo
                halo_disp_data = df_disp[df_disp['halo_id'] == halo_id]
                r_vals = halo_disp_data['r_val'].values
                disp_vals = halo_disp_data['displacement'].values
                
                # Interpolate displacement for particle radii
                disp = np.interp(r, r_vals, disp_vals)
                
                # Apply displacement
                with np.errstate(invalid='ignore', divide='ignore'):
                    direction = np.zeros_like(rel_pos)
                    mask = r > 0
                    direction[mask] = rel_pos[mask] / r[mask, np.newaxis]
                
                new_rel_pos = rel_pos + direction * disp[:, np.newaxis]
                new_pos = new_rel_pos + center
                
                # Store positions
                all_dmo_positions.append(particles['pos'])
                all_bcm_positions.append(new_pos)
                
            except Exception as e:
                print(f"Warning: Error processing halo {halo_id}: {e}")
                continue
        
        if not all_dmo_positions:
            print("Error: No valid particle data found")
            return None
        
        # Combine all positions
        dmo_positions = np.vstack(all_dmo_positions)
        bcm_positions = np.vstack(all_bcm_positions)
        
        print(f"Total particles: DMO={len(dmo_positions)}, BCM={len(bcm_positions)}")
        
        # Calculate power spectra
        k_dmo, Pk_dmo, k_bcm, Pk_bcm = ut.compare_power_spectra(
            dmo_positions, bcm_positions, self.boxsize, output_file
        )
        
        return k_dmo, Pk_dmo, k_bcm, Pk_bcm
    
    
    # Batch methods for mass and radius calculations
    def _batch_load_particles(self, halo_indices):
        """
        Load particles for multiple halos in a single HDF5 operation.
        
        This reduces I/O overhead by reading the entire particle dataset once
        and extracting data for all requested halos.
        
        Parameters
        ----------
        halo_indices : list
            List of halo indices to load particles for
            
        Returns
        -------
        dict
            Mapping of halo_index -> {'pos': array, 'vel': array, 'id': array}
        """
        particles_dict = {}
        
        print(f"Batch loading particles for {len(halo_indices)} halos...")
        
        with h5py.File(self.path_snapshot, 'r') as f:
            if 'PartType1' in f:
                parttype1 = f['PartType1']
                
                # ✅ OPTIMIZATION: Single large read instead of many small reads
                coords = parttype1['Coordinates'][:]  # Read entire dataset once
                vels = parttype1['Velocities'][:]
                ids = parttype1['ParticleIDs'][:]
                
                # Extract data for each halo from the loaded arrays
                for halo_idx in halo_indices:
                    start = self._calc_offset(halo_idx)
                    stop = start + self.halo['lentype_h'][halo_idx][1]
                    
                    particles_dict[halo_idx] = {
                        'pos': coords[start:stop] / 1e3,  # Convert to Mpc/h
                        'vel': vels[start:stop] / 1e3,
                        'id': ids[start:stop]
                    }
        
        return particles_dict
    
    def _batch_calculate_bcm_parameters(self, halo_indices):
        """
        Pre-calculate BCM parameters for a batch of halos.
        
        This avoids redundant cosmological parameter setup for each halo.
        
        Parameters
        ----------
        halo_indices : list
            List of halo indices to process
            
        Returns
        -------
        dict
            Mapping of halo_index -> bcm_parameters
        """
        bcm_params = {}
        
        # Set cosmological parameters once for the batch
        original_index = self.index
        
        for halo_idx in halo_indices:
            self.index = halo_idx
            
            # Extract halo properties
            M200 = self.halo['m200'][halo_idx]
            r200 = self.halo['r200'][halo_idx]
            c = ut.calc_concentration(M200, self.z)
            # ✅ OPTIMIZATION: Calculate parameters without full BCM calculation
            bcm_params[halo_idx] = {
                'M200': M200,
                'r200': r200,
                'c': c,
                'r_s' : r200 / c,  # Scale radius for NFW profile
                'r_tr' : 8 * r200,  # Truncation radius
                'r_ej': par.DEFAULTS['r_ej_factor'] * r200,  # Ejection radius
                'R_h' : par.DEFAULTS['R_h_factor'] * r200,
                'fbar': self.fbar,
                'f_rdm': af.f_rdm(self.fbar),
                'f_bgas': af.f_bgas(M200, self.fbar),
                'f_cgal': af.f_cgal(M200),
                'f_egas': None  # Will calculate after others
            }
            
            # Calculate f_egas (depends on other fractions)
            f_bgas = bcm_params[halo_idx]['f_bgas']
            f_cgal = bcm_params[halo_idx]['f_cgal']
            bcm_params[halo_idx]['f_egas'] = af.f_egas(f_bgas, f_cgal, self.fbar)
        
        self.index = original_index
        return bcm_params
    
    def _batch_write_results(self, batch_results, output_dir):
        """
        Write results for an entire batch to CSV files efficiently.
        
        Parameters
        ----------
        batch_results : dict
            Results for all halos in the batch
        output_dir : str
            Output directory for CSV files
        """
        import pandas as pd
        
        # ✅ OPTIMIZATION: Accumulate data and write in batches
        displacement_data = []
        bcm_data = []
        summary_data = []
        
        for halo_idx, results in batch_results.items():
            # Displacement profiles
            for i, (r, disp) in enumerate(zip(results['r_vals'], results['displacement'])):
                displacement_data.append([halo_idx, i, r, disp])
            
            # BCM parameters
            bcm_data.append([
                halo_idx, results['M200'], results['r200'], results['c'],
                results['fbar'], results['f_rdm'], results['f_bgas'], 
                results['f_cgal'], results['f_egas']
            ])
            
            # Summary data
            summary_data.append([
                halo_idx, results['M200'], results['r200'], results['c'],
                results['particle_count'], self.z, self.h, self.Om, self.Ob
            ])
        
        # Write all data at once (much faster than individual writes)
        if displacement_data:
            df = pd.DataFrame(displacement_data, 
                            columns=['halo_id', 'r_index', 'r_val', 'displacement'])
            df.to_csv(os.path.join(output_dir, 'batch_displacement.csv'), 
                    mode='a', header=False, index=False)
        
        if bcm_data:
            df = pd.DataFrame(bcm_data, 
                            columns=['halo_id', 'M200', 'r200', 'c', 'fbar',
                                    'f_rdm', 'f_bgas', 'f_cgal', 'f_egas'])
            df.to_csv(os.path.join(output_dir, 'batch_bcm_parameters.csv'),
                    mode='a', header=False, index=False)
        
    def calculate_displacement_all_halos_batched(self, min_mass=1e9, max_halos=None, 
                                               batch_size=20, save_individual=True,
                                               n_points = 200, 
                                               output_dir="displacement_results"):
        """
        Calculate displacement for all halos using batch processing for efficiency.
        
        This method processes halos in batches to minimize I/O overhead and 
        improve performance significantly.
        
        Parameters
        ----------
        min_mass : float, default 1e9
            Minimum halo mass to process (Msun/h)
        max_halos : int, optional
            Maximum number of halos to process
        batch_size : int, default 20
            Number of halos to process in each batch
        save_individual : bool, default True
            Whether to save results to CSV
        output_dir : str, default "displacement_results"
            Directory to save results
            
        Returns
        -------
        dict
            Results for all processed halos
        """
        import os
        import pandas as pd
        
        if save_individual:
            os.makedirs(output_dir, exist_ok=True)
            # Initialize CSV files with headers
            self._initialize_csv_files(output_dir)
        
        # Filter and sort halos
        valid_halos = [i for i, mass in enumerate(self.halo['m200']) if mass >= min_mass]
        if max_halos is not None:
            # Sort by mass (descending) and take the most massive
            mass_indices = sorted(valid_halos, key=lambda x: self.halo['m200'][x], reverse=True)
            valid_halos = mass_indices[:max_halos]
        
        print(f"Processing {len(valid_halos)} halos in batches of {batch_size}\n")
        
        # ✅ MAIN OPTIMIZATION: Process in batches
        original_index = self.index
        all_results = {}
        
        try:
            # Split halos into batches
            for batch_start in tqdm(range(0, len(valid_halos), batch_size), 
                                   desc="Processing batches"):
                batch_end = min(batch_start + batch_size, len(valid_halos))
                batch_halos = valid_halos[batch_start:batch_end]
                
                print(f"\nProcessing batch {batch_start//batch_size + 1}: "
                      f"halos {batch_start}-{batch_end-1}")
                
                # Step 1: Batch load particles (major speedup)
                particles_batch = self._batch_load_particles(batch_halos)
                
                # Step 2: Pre-calculate BCM parameters
                bcm_params_batch = self._batch_calculate_bcm_parameters(batch_halos)
                
                # Step 3: Process each halo in the batch
                batch_results = {}
                for halo_idx in batch_halos:
                    try:
                        self.index = halo_idx
                        
                        # Use pre-loaded data and parameters
                        particles = particles_batch[halo_idx]
                        bcm_params = bcm_params_batch[halo_idx]
                        
                        # Set BCM parameters
                        self.M200 = bcm_params['M200']
                        self.r200 = bcm_params['r200']
                        self.c = bcm_params['c']
                        self.r_s = bcm_params['r_s']
                        self.r_tr = bcm_params['r_tr']
                        self.r_ej = bcm_params['r_ej']
                        self.R_h = bcm_params['R_h']
                        self.fbar = bcm_params['fbar']
                        self.f_rdm = bcm_params['f_rdm']
                        self.f_bgas = bcm_params['f_bgas']
                        self.f_cgal = bcm_params['f_cgal']
                        self.f_egas = bcm_params['f_egas']
                        
                        # Calculate BCM (this is still the slow part, but unavoidable)
                        self.calculate(n_points=n_points)  # Reduced resolution
                        
                        # Store results
                        batch_results[halo_idx] = {
                            'r_vals': self.r_vals.copy(),
                            'displacement': self.components['disp'].copy(),
                            'particle_count': len(particles['pos']),
                            **bcm_params
                        }
                        
                    except Exception as e:
                        print(f"Warning: Error processing halo {halo_idx}: {e}")
                        continue
                
                # Step 4: Batch write results
                if save_individual and batch_results:
                    self._batch_write_results(batch_results, output_dir)
                
                all_results.update(batch_results)
                
                print(f"Completed batch {batch_start//batch_size + 1}: "
                      f"{len(batch_results)}/{len(batch_halos)} successful")
        
        finally:
            self.index = original_index
        
        print(f"\nCompleted processing: {len(all_results)}/{len(valid_halos)} halos successful")
        return all_results
    
    def _initialize_csv_files(self, output_dir):
        """Initialize CSV files with proper headers."""
        import pandas as pd
        
        # Create empty DataFrames with headers and save
        pd.DataFrame(columns=['halo_id', 'r_index', 'r_val', 'displacement']).to_csv(
            os.path.join(output_dir, 'batch_displacement.csv'), index=False)
        
        pd.DataFrame(columns=['halo_id', 'M200', 'r200', 'c', 'fbar',
                             'f_rdm', 'f_bgas', 'f_cgal', 'f_egas']).to_csv(
            os.path.join(output_dir, 'batch_bcm_parameters.csv'), index=False) 
                             

