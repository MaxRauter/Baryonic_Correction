import numpy as np
import os
import h5py
import glob
import hdf5plugin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    
    def __init__(self, path_group=None, path_snapshot=None, swift_path = None,index = None, load_part = True, verbose = False):
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
            if load_part:
                self._load_particles()
        if swift_path is not None:
            self.swift_path = swift_path
            self._swift_load_halodata()
    
    def _swift_load_halodata(self):
        
        path = self.swift_path
        # Check if path exists
        if path is None:
            print("No path provided.")
            return
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return False
        
        try:
            with h5py.File(path, 'r') as f:
                        print(f.keys())  # <KeysViewHDF5 ['Cells', 'Code', 'Cosmology', 'DMParticles', 'GravityScheme', 'Header', 'ICs_parameters', 'InternalCodeUnits', 'Parameters', 'PartType1', 'PhysicalConstants', 'Policy', 'RecordingTriggers', 'SubgridScheme', 'Units', 'UnusedParameters']>
                        header = f["Header"].attrs
                        cosmology = f["Cosmology"].attrs
                        cells = f["Cells"].attrs
                        for attr in f.keys():
                            print(f"---  {attr}: {f[attr]}  ---")
                            for key, value in f[attr].items():
                                print(f"{key}: {value}")
                        print("------------------\n")
                        for key,value in f['Cells'].items():
                            for val in f['Cells'][key]:
                                print(f"{key} : {val[0]}")
                        print("------------------\n")
                        print(f"Cosmology: {cosmology}")
                        print(f"z={header['Redshift'][0]:.2f} a={header['Scale-factor'][0]:.2f}")  # z=0.00 a=1.00
                        boxsize = header['BoxSize'][0]  # keep in mind that all length-units are in Mpc, not Mpc/h
                        print(f"{boxsize=}")  # boxsize=np.float64(442.856721302682)

                        parttype1 = f["SubgridScheme"]
                        for key,value in parttype1.items():
                            print(f"{key}: {value}")
                            print(f"{key} Unique values: {np.unique(value).shape}")  # Check unique values for Coordinates, Velocities, ParticleIDs
                        # load data into memory as a numpy array
                        coordinates = np.asarray(f["PartType1/Potentials"])
                        print(f"Coordinates shape: {coordinates.shape}")  # (2097152, 3)
                        print(np.count_nonzero(np.unique(coordinates)))  # (2097152, 3)

        except Exception as e:
            print(f"Error loading halo data: {e}")
            return 
        
    
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
            fractions_new = af.new_fractions(self.M200)
            self.f_star = fractions_new[0]
            self.f_cga = fractions_new[1]
            self.f_sga = fractions_new[2]
        else:
            # Custom abundance fractions
            if self.verbose:
                print("Using custom abundance fractions.")
            self.f_rdm = af.f_rdm(self.fbar)
            self.f_bgas = af.f_bgas(self.M200, self.fbar)
            self.f_cgal = af.f_cgal(self.M200)
            self.f_egas = af.f_egas(self.f_bgas,self.f_cgal,self.fbar)
            fractions_new = af.new_fractions(self.M200)
            self.f_star = fractions_new[0]
            self.f_cga = fractions_new[1]
            self.f_sga = fractions_new[2]
        
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

    def _calc_NFW_target_mass(self, inf = True):
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
        rho_nfw2 = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) for r in self.r_vals])
        M_nfw2 = ut.cumul_mass(self.r_vals, rho_nfw2)
        M_tot2 = dp.M_tot_truncated(self.rho0,self.r_s, self.r_tr)
        #print(f"Fixed M_tot: {M_tot:.3e}, M_tot2: {M_tot2:.3e}")
        self.fixed_M_tot = M_tot
        inf = True
        if not inf:
            # Integrate NFW profile over a large range to approximate total mass
            #rho_nfw = np.array(dp.rho_nfw(self.r200, self.r_s, self.rho0, self.r_tr))
            M_nfw = ut.cumul_mass_single(self.r200, rho_nfw,self.r_vals)
            M_tot = M_nfw
            #M_tot2 = dp.mass_nfw_analytical(self.r_vals[-1], self.r_s, self.rho0)
            #print(f"Fixed M_tot: {M_tot:.3e}, M_tot2: {M_tot2:.3e}")
            #rho0_2 = ut.bracket_rho0(self.fixed_M_tot, self.r_s, self.r_tr, self.r200)
        #rho0_2 = ut.bracket_rho0_2(self.M200, self.r_s, self.r_tr, self.r200)
        #print(f"Rho0 for M_tot_new: {rho0_2:.3e}")
        #M_tot_new = dp.mass_nfw_analytical_inf(self.r_tr, self.r_s, self.rho0)
        #print(f"Rho0: {self.rho0:.3e}")
        #print(f"Fixed M_tot_new: {M_tot2:.3e}")
        #print(f"Fixed M_tot: {M_tot:.3e}")
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
        y_bgas_vals2, y_cgal_vals2,y_egas_vals2,y_rdm_vals_fixed_xi2 = self.correction_factors_baryons(
            [self.f_bgas, self.f_cgal,self.f_egas,self.f_rdm], 
            [y_bgas_vals, y_cgal_vals, y_egas_vals, y_rdm_vals_fixed_xi],inf=False
        )
        
        #print(f"Comparison between bgas normalization at r_200 and infinity:")
        #print(f"  r_200: {ut.cumul_mass_single(self.r_vals[-1],y_bgas_vals2,self.r_vals):.3e}, inf: {ut.cumul_mass_single(self.r_vals[-1],y_bgas_vals,self.r_vals):.3e}")
        #y_cgal_vals = y_cgal_vals2
        #y_egas_vals = y_egas_vals2
        #y_bgas_vals = y_bgas_vals2
        # baryon components for xi
        # Note: y_bgas, y_egas, and y_cgal are already normalized
        baryons = [(self.r_vals, y_cgal_vals), 
            (self.r_vals, y_bgas_vals),
            (self.r_vals, y_egas_vals)
            ]
        
        # Calculate unnormalized profile
        rho_dm_contracted = dp.y_rdm_ac(self.r_vals, self.r_s, self.rho0, self.r_tr, 
                                    norm=1.0, a=0.68, f_cdm=0.839, 
                                    baryon_components=baryons, verbose=self.verbose)

        # Calculate total mass and correction factor
        M_contracted_inf = ut.cumul_mass(self.r_vals, rho_dm_contracted)[-1]
        M_contracted_m200 = ut.cumul_mass_single(self.r200, rho_dm_contracted,self.r_vals)
        target_mass = self.f_rdm * self.fixed_M_tot
        correction_factor = target_mass / M_contracted_inf
        
        if self.verbose:
            print(f"RDM mass correction factor: {correction_factor:.4f}")

        # Apply correction
        rho_dm_contracted *= correction_factor
        y_rdm_vals = rho_dm_contracted
        #y_rdm_vals = y_rdm_vals_fixed_xi

        rho_bcm = y_rdm_vals + y_bgas_vals + y_egas_vals + y_cgal_vals + rho_bkg_vals
        
        # new paper profiles
        y_rho_clm = np.array([dp.rho_clm(
            r = r, 
            f_sga=self.f_sga,  # Stellar growth adiabatic factor
            O_dm=self.Om - self.Ob,  # Dark matter density parameter
            O_m=self.Om,   # Total matter density parameter
            r_s=self.r_s, 
            rho0=self.rho0,
            r_tr=self.r_tr
            ) for r in self.r_vals])
        
        y_rho_cga = np.array([dp.rho_cga(
            r=r,
            R_h=self.R_h,
            M=self.M200,
            f_cga=self.f_cga
            ) for r in self.r_vals])
        
        y_rho_gas = np.array([dp.rho_gas(
            r=r,
            M=self.M200,
            r_vir=self.r200,
            f_b=self.fbar,
            f_star=self.f_star 
            ) for r in self.r_vals])
            
        rho_dmb = y_rho_cga + y_rho_clm + y_rho_gas
        
        self.profiles = {
            'rho_dmo': rho_dmo_vals,
            'rho_nfw': rho_nfw_vals,
            'rho_bkg': rho_bkg_vals,
            'y_bgas': y_bgas_vals,
            'y_egas': y_egas_vals,
            'y_cgal': y_cgal_vals,
            'y_rdm': y_rdm_vals,
            'rho_bcm': rho_bcm,
            'y_rho_clm': y_rho_clm,
            'y_rho_cga': y_rho_cga,
            'y_rho_gas': y_rho_gas,
            'rho_dmb': rho_dmb
        }
        
        
        return rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm, y_rho_clm, y_rho_cga, y_rho_gas, rho_dmb

    def correction_factors_baryons(self, fractions, profiles,inf=True):
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
        target = self._calc_NFW_target_mass(inf=False)
        for i in range(len(fractions)):
            if inf:
                mass = ut.cumul_mass(self.r_vals, profiles[i])[-1]
                correction = (fractions[i] * self.fixed_M_tot) / mass
            else:
                mass = ut.cumul_mass_single(self.r200, profiles[i],self.r_vals)
                correction = (fractions[i] * target) / mass

            cor_profiles.append(correction*profiles[i])
        return cor_profiles

    def _compute_mass_profiles(self, rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, y_rho_clm, y_rho_cga, y_rho_gas, rho_dmb, M_nfw):
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
        M_clm = ut.cumul_mass(self.r_vals, y_rho_clm)
        M_cga = ut.cumul_mass(self.r_vals, y_rho_cga)
        M_gas = ut.cumul_mass(self.r_vals, y_rho_gas)
        M_dmb = ut.cumul_mass(self.r_vals, rho_dmb)
        self.masses = {
            'M_dmo': M_dmo,
            'M_bkg': M_bkg,
            'M_rdm': M_rdm,
            'M_bgas': M_bgas,
            'M_egas': M_egas,
            'M_cgal': M_cgal,
            'M_bcm': M_bcm,
            'M_nfw': M_nfw,
            'M_clm': M_clm,
            'M_cga': M_cga,
            'M_gas': M_gas,
            'M_dmb': M_dmb
        }
        
#        self._check_masses()
        if self.verbose:
            self._print_masses_at_infinity()
        return M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm, M_clm, M_cga, M_gas, M_dmb

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
        rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm ,y_rho_clm, y_rho_cga, y_rho_gas, rho_dmb= self._compute_density_profiles(norm_bgas, norm_egas, norm_cgal, norm_rdm_fixed_xi, M_nfw)
        
        # Calculate mass profiles
        M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm, M_clm, M_cga, M_gas, M_dmb = self._compute_mass_profiles(rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, y_rho_clm, y_rho_cga, y_rho_gas, rho_dmb, M_nfw)
        
        #M_b = M_bgas + M_egas + M_cgal
        #M_i = M_nfw
        # Calculate RDM profile
        #y_rdm_vals2, M_rdm2 = self._calculate_rdm(M_i, M_b)
        
        # Calculate displacement
        f_inv_bcm = self._invert_mass_profile(M_bcm)
        disp = self._compute_displacement(M_dmo, f_inv_bcm)
        
        f_inv_bcm_new = self._invert_mass_profile(M_dmb)
        disp_new = self._compute_displacement(M_dmo, f_inv_bcm_new)
        
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
            'M_clm': M_clm,
            'M_cga': M_cga,
            'M_gas': M_gas,
            'M_dmb': M_dmb,
            'y_rho_clm': y_rho_clm,
            'y_rho_cga': y_rho_cga,
            'y_rho_gas': y_rho_gas,
            'y_rho_dmb': rho_dmb,
            'disp_new': disp_new,
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
                        'fixed_M_tot': self.fixed_M_tot,
                        'f_star': self.f_star,
                        'f_cga': self.f_cga,
                        'f_clm': self.f_clm,
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
                            'rho_cgal': self.components['cgal'][j],
                            'rho_clm': self.components['y_rho_clm'][j],
                            'rho_cga': self.components['y_rho_cga'][j],
                            'rho_gas': self.components['y_rho_gas'][j],
                            'rho_dmb': self.components['y_rho_dmb'][j]
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
                            'M_nfw': self.components['M_nfw'][j],
                            'M_clm': self.components['M_clm'][j],
                            'M_cga': self.components['M_cga'][j],
                            'M_gas': self.components['M_gas'][j],
                            'M_dmb': self.components['M_dmb'][j]
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
                'mean_f_cga': df_bcm_params['f_cga'].mean(),
                'std_f_cga': df_bcm_params['f_cga'].std(),
                'mean_f_clm': df_bcm_params['f_clm'].mean(),
                'std_f_clm': df_bcm_params['f_clm'].std(),
                'mean_f_star': df_bcm_params['f_star'].mean(),
                'std_f_star': df_bcm_params['f_star'].std(),
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
                f.write(f"  f_cga:  {analysis_summary['mean_f_cga']:.4f} ± {analysis_summary['std_f_cga']:.4f}\n")
                f.write(f"  f_clm:  {analysis_summary['mean_f_clm']:.4f} ± {analysis_summary['std_f_clm']:.4f}\n")
                f.write(f"  f_star: {analysis_summary['mean_f_star']:.4f} ± {analysis_summary['std_f_star']:.4f}\n")
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
        disp_file = os.path.join(csv_dir, "batch_displacement.csv")
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
        all_dmb_positions = []
        
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
                disp_vals_new = halo_disp_data['displacement_new'].values
                
                # Interpolate displacement for particle radii
                disp = np.interp(r, r_vals, disp_vals)
                disp_new = np.interp(r, r_vals, disp_vals_new)
                
                # Apply displacement
                with np.errstate(invalid='ignore', divide='ignore'):
                    direction = np.zeros_like(rel_pos)
                    mask = r > 0
                    direction[mask] = rel_pos[mask] / r[mask, np.newaxis]
                
                new_rel_pos = rel_pos + direction * disp[:, np.newaxis]
                new_pos = new_rel_pos + center
                
                new_rel_pos_new = rel_pos + direction * disp_new[:, np.newaxis]
                new_pos_new = new_rel_pos_new + center
                
                # Store positions
                all_dmo_positions.append(particles['pos'])
                all_bcm_positions.append(new_pos)
                all_dmb_positions.append(new_pos_new)
                
            except Exception as e:
                print(f"Warning: Error processing halo {halo_id}: {e}")
                continue
        
        if not all_dmo_positions:
            print("Error: No valid particle data found")
            return None
        
        # Combine all positions
        dmo_positions = np.vstack(all_dmo_positions)
        bcm_positions = np.vstack(all_bcm_positions)
        dmb_positions = np.vstack(all_dmb_positions)
        
        print(f"Total particles: DMO={len(dmo_positions)}, BCM={len(bcm_positions)}, DMB={len(dmb_positions)}")
        
        # Calculate power spectra
        k_dmo, Pk_dmo, k_bcm, Pk_bcm, k_dmb, Pk_dmb = ut.compare_power_spectra(
            dmo_positions, bcm_positions, dmb_positions, self.boxsize, output_file
        )
        
        return k_dmo, Pk_dmo, k_bcm, Pk_bcm, k_dmb, Pk_dmb, dmo_positions, bcm_positions, dmb_positions
    
    def calculate_gravitational_potential_cic(self, positions, grid_size=128, smoothing_length=None):
        """
        Calculate gravitational potential from particle positions using CIC assignment.
        
        This method assigns particles to a grid using Cloud-In-Cell interpolation,
        computes the density field, and then calculates the gravitational potential
        using Poisson's equation in Fourier space.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Particle positions in shape (N, 3) in Mpc/h
        grid_size : int, default 128
            Size of the cubic grid for CIC assignment
        smoothing_length : float, optional
            Smoothing scale in Mpc/h. If None, uses grid spacing
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'potential': 3D potential grid
            - 'density': 3D density grid  
            - 'grid_coords': Grid coordinate arrays
            - 'grid_spacing': Grid spacing in Mpc/h
        """
        import numpy as np
        from scipy.fft import fftn, ifftn
        
        # Validate input positions
        if len(positions) == 0:
            print("Warning: No particles provided!")
            return None
        
        print(f"Processing {len(positions)} particles...")
        
        # Ensure positions are within box bounds and remove invalid values
        valid_mask = ~np.isnan(positions).any(axis=1) & ~np.isinf(positions).any(axis=1)
        positions = positions[valid_mask]
        positions = np.mod(positions, self.boxsize)
        
        print(f"Valid particles after cleaning: {len(positions)}")
        
        # Initialize density grid
        density_grid = np.zeros((grid_size, grid_size, grid_size))
        grid_spacing = self.boxsize / grid_size
        
        # CIC assignment
        print("Assigning particles to grid using CIC...")
        for pos in tqdm(positions, desc="CIC assignment"):
            # Convert to grid coordinates
            grid_pos = pos / grid_spacing
            
            # Get integer and fractional parts
            i0, j0, k0 = np.floor(grid_pos).astype(int) % grid_size
            i1, j1, k1 = (np.floor(grid_pos).astype(int) + 1) % grid_size
            
            # Fractional distances
            dx, dy, dz = grid_pos - np.floor(grid_pos)
            
            # CIC weights
            w000 = (1 - dx) * (1 - dy) * (1 - dz)
            w001 = (1 - dx) * (1 - dy) * dz
            w010 = (1 - dx) * dy * (1 - dz)
            w011 = (1 - dx) * dy * dz
            w100 = dx * (1 - dy) * (1 - dz)
            w101 = dx * (1 - dy) * dz
            w110 = dx * dy * (1 - dz)
            w111 = dx * dy * dz
            
            # Assign to grid
            density_grid[i0, j0, k0] += w000
            density_grid[i0, j0, k1] += w001
            density_grid[i0, j1, k0] += w010
            density_grid[i0, j1, k1] += w011
            density_grid[i1, j0, k0] += w100
            density_grid[i1, j0, k1] += w101
            density_grid[i1, j1, k0] += w110
            density_grid[i1, j1, k1] += w111
        
        # Convert to density (particles per cell to density)
        rho_crit_0 = 2.775e11  # Critical density at z=0 in Msun/h / (Mpc/h)^3
        E_z_squared = self.Om * (1 + self.z)**3 + (1 - self.Om)  # Simplified for flat ΛCDM
        rho_crit_z = rho_crit_0 * E_z_squared
        particle_mass = (self.Om * rho_crit_z * self.boxsize**3) / len(positions)
        density_grid = density_grid * particle_mass / grid_spacing**3
        
        # Convert to overdensity
        mean_density = np.mean(density_grid)
        delta_grid = density_grid / mean_density - 1.0
        
        print("Calculating potential using FFT...")
        
        # Calculate potential using Poisson equation in Fourier space
        # ∇²φ = 4πGρδ  =>  φ = -4πGρδ/k²
        
        # FFT to k-space
        delta_k = np.fft.fftn(delta_grid)
        
        # Create k-space grid
        kx = np.fft.fftfreq(grid_size, d=grid_spacing) * 2 * np.pi
        ky = np.fft.fftfreq(grid_size, d=grid_spacing) * 2 * np.pi
        kz = np.fft.fftfreq(grid_size, d=grid_spacing) * 2 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        
        # Avoid division by zero at k=0
        K2[0, 0, 0] = (2 * np.pi / self.boxsize)**2  # Fundamental mode (was 1.0)
        
        # FFT of density field
        delta_k = fftn(delta_grid)
        
        # Calculate potential in k-space
        # Using G = 4.301e-9 Mpc/M_sun (km/s)^2 in appropriate units
        G = 4.301e-3  # Mpc³ M_sun⁻¹ (km/s)² - CORRECT for potential
        rho_crit = rho_crit_0 * 1.989e30 / (3.086e24)**3
        #potential_k = -4 * np.pi * G * mean_density * delta_k / K2
        potential_k = -4 * np.pi * G * rho_crit * delta_k / K2
        
        # Set DC mode to zero (arbitrary potential zero point)
        potential_k[0, 0, 0] = 0.0
        
        # Apply smoothing if requested
        if smoothing_length is not None:
            smoothing_k = np.exp(-0.5 * K2 * smoothing_length**2)
            potential_k *= smoothing_k
        
        # Transform back to real space
        potential_grid = np.real(ifftn(potential_k))
        
        # Ensure zero mean (removes any numerical drift)
        potential_grid -= np.mean(potential_grid)
        
        # Create coordinate grids
        x = np.linspace(0, self.boxsize, grid_size, endpoint=False)
        y = np.linspace(0, self.boxsize, grid_size, endpoint=False)
        z = np.linspace(0, self.boxsize, grid_size, endpoint=False)
        
        print(f"Potential calculation complete. Grid size: {grid_size}³")
        print(f"Potential range: {potential_grid.min():.3e} to {potential_grid.max():.3e}")
        
        return {
            'potential': potential_grid,
            'density': density_grid,
            'overdensity': delta_grid,
            'grid_coords': (x, y, z),
            'grid_spacing': grid_spacing,
            'particle_mass': particle_mass,
            'mean_density': mean_density
        }
    
    def compare_potentials_dmo_bcm_dmb(self, output_file=None, grid_size=128):
        """
        Compare gravitational potentials between DMO, BCM, and DMB using CIC.
        
        Parameters
        ----------
        output_file : str, optional
            If provided, saves plots to this file
        grid_size : int, default 128
            Grid resolution for CIC assignment
            
        Returns
        -------
        dict
            Dictionary containing potential results for DMO, BCM, and DMB
        """
        import pandas as pd
        import os
        
        # Load displacement data
        csv_dir = "/Users/Maxi/Desktop/Uni/Master/Masterarbeit/Baryonic_Correction/halo_data/displacement_all_halos/"
        disp_file = os.path.join(csv_dir, "batch_displacement.csv")
        
        if not os.path.exists(disp_file):
            print(f"Error: Displacement file not found: {disp_file}")
            return None
        
        df_disp = pd.read_csv(disp_file)
        print("Loading displacement data from CSV...")
        
        # Get particle positions for all halos
        all_dmo_positions = []
        all_bcm_positions = []
        all_dmb_positions = []
        
        for halo_id in tqdm(df_disp['halo_id'].unique(), desc="Loading particle positions"):
            try:
                particles = self.get_halo_particles(halo_id)
                if particles is None:
                    continue
                    
                center = self.get_halo_center(halo_id)
                rel_pos = particles['pos'] - center
                r = np.linalg.norm(rel_pos, axis=1)
                
                # Get displacements
                halo_disp_data = df_disp[df_disp['halo_id'] == halo_id]
                r_vals = halo_disp_data['r_val'].values
                disp_vals = halo_disp_data['displacement'].values
                disp_vals_new = halo_disp_data['displacement_new'].values
                
                disp = np.interp(r, r_vals, disp_vals)
                disp_new = np.interp(r, r_vals, disp_vals_new)
                
                # Apply displacements
                with np.errstate(invalid='ignore', divide='ignore'):
                    direction = np.zeros_like(rel_pos)
                    mask = r > 0
                    direction[mask] = rel_pos[mask] / r[mask, np.newaxis]
                
                # Calculate displaced positions
                bcm_pos = rel_pos + direction * disp[:, np.newaxis] + center
                dmb_pos = rel_pos + direction * disp_new[:, np.newaxis] + center
                
                all_dmo_positions.append(particles['pos'])
                all_bcm_positions.append(bcm_pos)
                all_dmb_positions.append(dmb_pos)
                
            except Exception as e:
                print(f"Warning: Error processing halo {halo_id}: {e}")
                continue
        
        # Combine positions
        dmo_positions = np.vstack(all_dmo_positions)
        bcm_positions = np.vstack(all_bcm_positions)
        dmb_positions = np.vstack(all_dmb_positions)
        
        print(f"Total particles: DMO={len(dmo_positions)}, BCM={len(bcm_positions)}, DMB={len(dmb_positions)}")
        
        # Calculate potentials
        print("\nCalculating DMO potential...")
        dmo_result = self.calculate_gravitational_potential_cic(dmo_positions, grid_size)
    
        if True: 
            # Plot comparison
            self._plot_potential_comparison(dmo_result, dmo_result, dmo_result, output_file)
            
            return {
                'dmo': dmo_result,
                'bcm': dmo_result, 
                'dmb': dmo_result
            }
        
        print("Calculating BCM potential...")
        bcm_result = self.calculate_gravitational_potential_cic(bcm_positions, grid_size)
        
        print("Calculating DMB potential...")
        dmb_result = self.calculate_gravitational_potential_cic(dmb_positions, grid_size)
        
        # Plot comparison
        self._plot_potential_comparison(dmo_result, bcm_result, dmb_result, output_file)
        
        return {
            'dmo': dmo_result,
            'bcm': bcm_result, 
            'dmb': dmb_result
        }
    
    def _plot_potential_comparison_yt(self, dmo_result, bcm_result, dmb_result, output_file=None):
        """Plot comparison using YT for better cosmological visualization."""
        try:
            import yt
            import matplotlib.pyplot as plt
            
            # Create YT datasets from density grids
            def create_yt_dataset(density_grid, boxsize, name):
                data = {
                    ('gas', 'density'): (density_grid, 'g/cm**3'),
                }
                
                bbox = np.array([[-boxsize/2, boxsize/2],
                            [-boxsize/2, boxsize/2], 
                            [-boxsize/2, boxsize/2]])
                
                ds = yt.load_uniform_grid(data, density_grid.shape, 
                                        length_unit="Mpc", 
                                        bbox=bbox, 
                                        dataset_name=name)
                return ds
            
            # Convert density to proper units
            rho_crit = 2.775e11 * 1.989e30 / (3.086e24)**3
            
            dmo_density = dmo_result['density'] * rho_crit
            bcm_density = bcm_result['density'] * rho_crit  
            dmb_density = dmb_result['density'] * rho_crit
            
            # Apply power-law enhancement to improve low-density visibility
            # This compresses high values and stretches low values
            power = 0.4  # Values < 1 enhance low density regions
            
            def enhance_low_density(density_data):
                """Apply power-law transformation to enhance low-density structures."""
                # Normalize to [0, 1] range
                normalized = (density_data - density_data.min()) / (density_data.max() - density_data.min())
                # Apply power transformation
                enhanced = np.power(normalized, power)
                # Scale back to physical units (but with enhanced contrast)
                return enhanced * (density_data.max() - density_data.min()) + density_data.min()
            
            dmo_density_enhanced = enhance_low_density(dmo_density)
            bcm_density_enhanced = enhance_low_density(bcm_density)
            dmb_density_enhanced = enhance_low_density(dmb_density)
            
            # Create YT datasets
            ds_dmo = create_yt_dataset(dmo_density_enhanced, self.boxsize, "DMO")
            ds_bcm = create_yt_dataset(bcm_density_enhanced, self.boxsize, "BCM") 
            ds_dmb = create_yt_dataset(dmb_density_enhanced, self.boxsize, "DMB")
        
            # Create matplotlib figure FIRST
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Top row: Gravitational potential
            mid = dmo_result['potential'].shape[0] // 2
            
            im1 = axes[0,0].imshow(dmo_result['potential'][mid,:,:], origin='lower', cmap='viridis')
            axes[0,0].set_title('DMO Potential')
            plt.colorbar(im1, ax=axes[0,0])
            
            im2 = axes[0,1].imshow(bcm_result['potential'][mid,:,:], origin='lower', cmap='viridis')
            axes[0,1].set_title('BCM Potential') 
            plt.colorbar(im2, ax=axes[0,1])
            
            im3 = axes[0,2].imshow(dmb_result['potential'][mid,:,:], origin='lower', cmap='viridis')
            axes[0,2].set_title('DMB Potential')
            plt.colorbar(im3, ax=axes[0,2])
            
            # Bottom row: Use YT but extract data manually instead of using YT's plotting
            datasets = [ds_dmo, ds_bcm, ds_dmb]
            titles = ['DMO Cosmic Web', 'BCM Cosmic Web', 'DMB Cosmic Web']
            
            for i, (ds, title) in enumerate(zip(datasets, titles)):
                # Create YT projection but don't plot it
                proj = yt.ProjectionPlot(ds, 'z', ('gas', 'density'), 
                                    width=(self.boxsize, 'Mpc'),
                                    center='c')
                
                # Set colormap optimized for low-density visibility
                proj.set_cmap(('gas', 'density'), 'inferno')  # Better for low values
                
                # Calculate adaptive z-limits based on data percentiles
                frb_temp = proj.data_source.to_frb((self.boxsize, 'Mpc'), 128)
                density_proj_temp = np.array(frb_temp[('gas', 'density')])
                
                # Use percentiles to set limits that enhance low-density regions
                vmin = np.percentile(density_proj_temp[density_proj_temp > 0], 5)   # 5th percentile
                vmax = np.percentile(density_proj_temp, 95)  # 95th percentile
                
                # Apply asinh scaling for better low-value visibility
                proj.set_zlim(('gas', 'density'), vmin, vmax)
                
                # Extract the data from YT projection with higher resolution
                frb = proj.data_source.to_frb((self.boxsize, 'Mpc'), 800)  # Higher resolution
                density_proj = np.array(frb[('gas', 'density')])
                
                # Apply additional enhancement for matplotlib display
                # Use asinh scaling which is excellent for astronomical data
                asinh_data = np.arcsinh(density_proj / vmin) / np.arcsinh(vmax / vmin)
                
                # Plot on our matplotlib axes with enhanced visualization
                im = axes[1, i].imshow(asinh_data, origin='lower', cmap='inferno',
                                    extent=[-self.boxsize/2, self.boxsize/2, 
                                            -self.boxsize/2, self.boxsize/2])
                axes[1, i].set_title(f'{title} (Enhanced Low-Density)')
                axes[1, i].set_xlabel('X [Mpc/h]')
                axes[1, i].set_ylabel('Y [Mpc/h]')
                
                # Add contours to highlight low-density structure
                contour_levels = np.linspace(asinh_data.min(), asinh_data.max(), 8)
                axes[1, i].contour(asinh_data, levels=contour_levels, colors='white', 
                                alpha=0.3, linewidths=0.4, 
                                extent=[-self.boxsize/2, self.boxsize/2, 
                                        -self.boxsize/2, self.boxsize/2])
                
                plt.colorbar(im, ax=axes[1, i], label='Enhanced Density (asinh scaled)')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"YT cosmic web comparison saved to: {output_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"YT visualization failed: {e}")
            print("Using fallback visualization...")
            self._plot_potential_comparison_fallback(dmo_result, bcm_result, dmb_result, output_file)
    
    def _plot_potential_comparison_fallback(self, dmo_result, bcm_result, dmb_result, output_file=None):
        """Improved fallback visualization without YT."""
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: Potential slices
        mid = dmo_result['potential'].shape[0] // 2
        
        im1 = axes[0,0].imshow(dmo_result['potential'][mid,:,:], origin='lower', cmap='viridis')
        axes[0,0].set_title('DMO Potential')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(bcm_result['potential'][mid,:,:], origin='lower', cmap='viridis')
        axes[0,1].set_title('BCM Potential')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[0,2].imshow(dmb_result['potential'][mid,:,:], origin='lower', cmap='viridis')
        axes[0,2].set_title('DMB Potential')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Bottom row: Improved cosmic web visualization
        def plot_cosmic_web_improved(density_data, ax, title):
            """Create better cosmic web visualization."""
            # Sum projection along z-axis (column density)
            column_density = np.sum(density_data, axis=0)
            
            # Apply logarithmic scaling
            log_column = np.log10(column_density + column_density.max() * 1e-6)
            
            # Apply Gaussian smoothing to enhance structure
            smoothed = gaussian_filter(log_column, sigma=1.5)
            
            # Enhanced contrast
            vmin = np.percentile(smoothed, 10)
            vmax = np.percentile(smoothed, 99.5)
            
            # Use 'hot' colormap for classic astronomy look
            im = ax.imshow(smoothed, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
            ax.set_title(f'{title} (Log Column Density)')
            
            # Add contours to highlight structure
            levels = np.linspace(vmin, vmax, 8)
            ax.contour(smoothed, levels=levels, colors='cyan', alpha=0.3, linewidths=0.5)
            
            return im
        
        # Plot cosmic web for each dataset
        datasets = [(dmo_result['density'], 'DMO'), 
                    (bcm_result['density'], 'BCM'),
                    (dmb_result['density'], 'DMB')]
        
        for i, (density, title) in enumerate(datasets):
            im = plot_cosmic_web_improved(density, axes[1, i], title)
            plt.colorbar(im, ax=axes[1, i], label='Log₁₀(Column Density)')
            
            # Add scale information
            grid_size = density.shape[0]
            axes[1, i].set_xlabel(f'Grid X (Box: {self.boxsize:.1f} Mpc/h)')
            axes[1, i].set_ylabel(f'Grid Y (Box: {self.boxsize:.1f} Mpc/h)')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Cosmic web comparison saved to: {output_file}")
        
        plt.show()
    
    def _plot_potential_comparison_old(self, dmo_result, bcm_result, dmb_result, output_file=None):
        """Plot comparison of gravitational potentials and density distributions."""
        from matplotlib.colors import LogNorm
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Central slices through the grids
        mid = dmo_result['potential'].shape[0] // 2
        
        # Top row: Potential slices
        im1 = axes[0,0].imshow(dmo_result['potential'][mid,:,:], origin='lower', cmap='viridis')
        axes[0,0].set_title('DMO Potential')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(bcm_result['potential'][mid,:,:], origin='lower', cmap='viridis')
        axes[0,1].set_title('BCM Potential')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[0,2].imshow(dmb_result['potential'][mid,:,:], origin='lower', cmap='viridis')
        axes[0,2].set_title('DMB Potential')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Bottom row: Cosmic web filament structure (overdensity field)
        def plot_cosmic_web(overdensity_data, ax, title):
            """Plot cosmic web structure using 3D projection."""
            # Project the 3D volume along the z-axis (sum all slices)
            projected_density = np.sum(overdensity_data, axis=0)  # Sum along z-axis
            
            # Apply smoothing to enhance filament visibility
            from scipy.ndimage import gaussian_filter
            projected_smooth = gaussian_filter(projected_density, sigma=1.0)
            
            # Create filament-enhancing visualization
            vmin = np.percentile(projected_smooth, 5)
            vmax = np.percentile(projected_smooth, 95)
            vabsmax = max(abs(vmin), abs(vmax))
            
            # Use RdBu_r colormap for classic cosmology visualization
            im = ax.imshow(projected_smooth, origin='lower', cmap='RdBu_r', 
                        vmin=-vabsmax, vmax=vabsmax)
            ax.set_title(f'{title} Cosmic Web (3D Projected)')
            
            # Add contour lines to emphasize filament structure
            contour_levels = [0.5, 1.0, 2.0, 5.0] * np.std(projected_smooth)
            ax.contour(projected_smooth, levels=contour_levels, colors='black', 
                    linewidths=0.5, alpha=0.6)
            
            return im
        
        # Alternative: Show density with filament-optimized colormap
        def plot_filament_density(density_data, ax, title):
            """Plot density field optimized for filament visualization."""
            # Log-transform density for better dynamic range
            log_density = np.log10(density_data + 1e-10)  # Add small value to avoid log(0)
            
            # Use 'hot' or 'afmhot' colormap for classic filament appearance
            im = ax.imshow(log_density, origin='lower', cmap='afmhot')
            ax.set_title(f'{title} Filaments (Log Density)')
            
            return im
        
        # Choose visualization method based on data quality
        try:
            # Method 1: Use overdensity for cosmic web structure
            im4 = plot_cosmic_web(dmo_result['overdensity'][mid,:,:], axes[1,0], 'DMO')
            plt.colorbar(im4, ax=axes[1,0], label='δ (overdensity)')
            
            im5 = plot_cosmic_web(bcm_result['overdensity'][mid,:,:], axes[1,1], 'BCM')
            plt.colorbar(im5, ax=axes[1,1], label='δ (overdensity)')
            
            im6 = plot_cosmic_web(dmb_result['overdensity'][mid,:,:], axes[1,2], 'DMB')
            plt.colorbar(im6, ax=axes[1,2], label='δ (overdensity)')
            
        except:
            # Fallback: Use density-based filament visualization
            print("Using density-based filament visualization")
            
            im4 = plot_filament_density(dmo_result['density'][mid,:,:], axes[1,0], 'DMO')
            plt.colorbar(im4, ax=axes[1,0], label='Log₁₀(ρ)')
            
            im5 = plot_filament_density(bcm_result['density'][mid,:,:], axes[1,1], 'BCM')
            plt.colorbar(im5, ax=axes[1,1], label='Log₁₀(ρ)')
            
            im6 = plot_filament_density(dmb_result['density'][mid,:,:], axes[1,2], 'DMB')
            plt.colorbar(im6, ax=axes[1,2], label='Log₁₀(ρ)')
        
        # Add grid lines and axis labels for better readability
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel('Grid X')
                ax.set_ylabel('Grid Y')
                # Optional: Add physical scale if known
                # ax.set_xlabel(f'X [Mpc/h] (Box: {self.boxsize:.1f})')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Potential and density comparison saved to: {output_file}")
        
        plt.show()
    
    def _plot_potential_comparison(self, dmo_result, bcm_result, dmb_result, output_file=None):
        """Plot comparison with YT for better cosmic web visualization."""
        try:
            self._plot_potential_comparison_yt(dmo_result, bcm_result, dmb_result, output_file)
        except:
            self._plot_potential_comparison_fallback(dmo_result, bcm_result, dmb_result, output_file)
    
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
        
        if self.verbose:
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
            
            # Calculate new fraction parameters first
            fractions_new = af.new_fractions(M200)
            
            # store all parameters in a dictionary
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
                'f_egas': None,  # Will calculate after others
                'f_star': fractions_new[0],
                'f_cga': fractions_new[1],
                'f_sga': fractions_new[2]
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
            for i, (r, disp, disp_new) in enumerate(zip(results['r_vals'], results['displacement'],results['displacement_new'])):
                displacement_data.append([halo_idx, i, r, disp, disp_new])
            
            # BCM parameters
            bcm_data.append([
                halo_idx, results['M200'], results['r200'], results['c'],
                results['fbar'], results['f_rdm'], results['f_bgas'], 
                results['f_cgal'], results['f_egas'], results['f_star'],
                results['f_cga'], results['f_sga']
            ])
            
            # Summary data
            summary_data.append([
                halo_idx, results['M200'], results['r200'], results['c'],
                results['particle_count'], self.z, self.h, self.Om, self.Ob
            ])
        
        # Write all data at once (much faster than individual writes)
        if displacement_data:
            df = pd.DataFrame(displacement_data, 
                            columns=['halo_id', 'r_index', 'r_val', 'displacement', 'displacement_new'])
            df.to_csv(os.path.join(output_dir, 'batch_displacement.csv'), 
                    mode='a', header=False, index=False)
        
        if bcm_data:
            df = pd.DataFrame(bcm_data, 
                            columns=['halo_id', 'M200', 'r200', 'c', 'fbar',
                                    'f_rdm', 'f_bgas', 'f_cgal', 'f_egas','f_star', 'f_cga', 'f_sga'])
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
                                    desc="Processing batches",
                                    unit="batch"):
                batch_end = min(batch_start + batch_size, len(valid_halos))
                batch_halos = valid_halos[batch_start:batch_end]
                
                #print(f"\rBatch {batch_start//batch_size + 1}: Loading particles for {len(batch_halos)} halos...", end='', flush=True)
                
                # Step 1: Batch load particles (major speedup)
                particles_batch = self._batch_load_particles(batch_halos)
                
                # Step 2: Pre-calculate BCM parameters
                bcm_params_batch = self._batch_calculate_bcm_parameters(batch_halos)
                
                #print(f"\rBatch {batch_start//batch_size + 1}: Processing BCM calculations...", end='', flush=True)
                
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
                        self.f_star = bcm_params['f_star'] 
                        self.f_cga = bcm_params['f_cga']
                        self.f_sga = bcm_params['f_sga']
                        
                        # Calculate BCM (this is still the slow part, but unavoidable)
                        self.calculate(n_points=n_points)  # Reduced resolution
                        
                        # Store results
                        batch_results[halo_idx] = {
                            'r_vals': self.r_vals.copy(),
                            'displacement': self.components['disp'].copy(),
                            'displacement_new': self.components['disp_new'].copy(),
                            'particle_count': len(particles['pos']),
                            **bcm_params
                        }
                        
                    except Exception as e:
                        print(f"\rWarning: Error processing halo {halo_idx}: {e}")
                        continue
                
                # Step 4: Batch write results
                if save_individual and batch_results:
                    self._batch_write_results(batch_results, output_dir)
                
                all_results.update(batch_results)
                
                #print(f"\rCompleted batch {batch_start//batch_size + 1}: {len(batch_results)}/{len(batch_halos)} successful")
        
        finally:
            self.index = original_index
        
        print(f"\nCompleted processing: {len(all_results)}/{len(valid_halos)} halos successful")
        return all_results
    
    def _initialize_csv_files(self, output_dir):
        """Initialize CSV files with proper headers."""
        import pandas as pd
        
        # Create empty DataFrames with headers and save
        pd.DataFrame(columns=['halo_id', 'r_index', 'r_val', 'displacement', 'displacement_new']).to_csv(
            os.path.join(output_dir, 'batch_displacement.csv'), index=False)
        
        pd.DataFrame(columns=['halo_id', 'M200', 'r200', 'c', 'fbar',
                             'f_rdm', 'f_bgas', 'f_cgal', 'f_egas']).to_csv(
            os.path.join(output_dir, 'batch_bcm_parameters.csv'), index=False) 
                             
    # visualization methods
    def visualize_potential_3d(self, method='interactive', density_percentile = 15, output_file=None, grid_size=128):
        """
        Create 3D visualizations of gravitational potential.
        
        Parameters
        ----------
        method : str, default 'interactive'
            Visualization method: 'interactive', 'volume', 'slices', or 'pyvista'
        output_file : str, optional
            Output file path
        grid_size : int, default 128
            Grid resolution
        """
        
        def enhance_potential_display(potential_grid):
            """Enhance potential for better visualization."""
            # Shift to make all values positive for colormap
            shifted = potential_grid - potential_grid.min()
            
            # Apply power scaling to enhance contrast
            power = 0.5  # Adjust between 0.1-1.0 for different contrast
            normalized = (shifted / shifted.max()) ** power
            
            return normalized * (potential_grid.max() - potential_grid.min()) + potential_grid.min()
                
        # Calculate potential for DMO case
        print("Loading DMO particle positions...")
        all_positions = []
        for halo_id in self.halo['id']:
            particles = self.get_halo_particles(halo_id)
            if particles is not None:
                all_positions.append(particles['pos'])
        
        if not all_positions:
            print("No particles found")
            return
        
        dmo_positions = np.vstack(all_positions)
        dmo_result = self.calculate_gravitational_potential_cic(dmo_positions, grid_size)
        
        dmo_result['potential'] = enhance_potential_display(dmo_result['potential'])
        
        # Choose visualization method
        if method == 'interactive':
            self.plot_potential_3d_interactive(dmo_result, density_percentile, output_file)
        elif method == 'volume':
            self.plot_potential_3d_volume(dmo_result, output_file)
        elif method == 'slices':
            self.plot_potential_3d_slices(dmo_result, output_file)
        elif method == 'pyvista':
            self.plot_potential_3d_pyvista(dmo_result, density_percentile, output_file)
        else:
            print(f"Unknown method: {method}")
            print("Available methods: 'interactive', 'volume', 'slices', 'pyvista'")
    
    def plot_potential_3d_interactive(self, potential_result, density_percentile = 15, output_file=None):
        """Create interactive 3D visualization of gravitational potential."""
        try:
            import plotly.graph_objects as go
            from skimage import measure
            
            potential_grid = potential_result['potential']
            density_grid = potential_result['density']
            
            # FILTER LOW DENSITY REGIONS
            # Method 1: Percentile-based filtering
            density_threshold = np.percentile(density_grid[density_grid > 0], density_percentile)
            
            # Create mask for significant structures
            structure_mask = density_grid > density_threshold
            
            # Apply 3D morphological operations to clean up the mask
            from scipy.ndimage import binary_opening, binary_closing
            
            # Remove small isolated regions
            cleaned_mask = binary_opening(structure_mask, iterations=1)
            # Fill small holes
            cleaned_mask = binary_closing(cleaned_mask, iterations=1)
            
            # Apply mask to data
            filtered_potential = potential_grid.copy()
            filtered_density = density_grid.copy()
            
            filtered_potential[~cleaned_mask] = np.nan
            filtered_density[~cleaned_mask] = 0
            
            # Create isosurfaces at different potential levels
            fig = go.Figure()
            
            # Define potential levels for isosurfaces
            potential_levels = np.percentile(filtered_density[filtered_density > density_threshold], 
                                     [25, 50, 75, 95])
            colors = ['blue', 'cyan', 'orange', 'red']
            
            # Create coordinate grids
            x, y, z = np.mgrid[0:self.boxsize:potential_grid.shape[0]*1j,
                                0:self.boxsize:potential_grid.shape[1]*1j,
                                0:self.boxsize:potential_grid.shape[2]*1j]
            
            for i, (level, color) in enumerate(zip(potential_levels, colors)):
                # Extract isosurface using marching cubes
                try:
                    # Extract isosurface using marching cubes on density
                    verts, faces, _, _ = measure.marching_cubes(filtered_density, level)
                    
                    if len(verts) > 0:
                        # Scale vertices to physical coordinates
                        verts = verts * self.boxsize / potential_grid.shape[0]
                        
                        # Get potential values at surface vertices for coloring
                        vert_indices = (verts / self.boxsize * potential_grid.shape[0]).astype(int)
                        vert_indices = np.clip(vert_indices, 0, potential_grid.shape[0]-1)
                        
                        surface_potentials = filtered_potential[
                            vert_indices[:, 0], vert_indices[:, 1], vert_indices[:, 2]
                        ]
                        
                        fig.add_trace(go.Mesh3d(
                            x=verts[:, 0],
                            y=verts[:, 1], 
                            z=verts[:, 2],
                            i=faces[:, 0],
                            j=faces[:, 1],
                            k=faces[:, 2],
                            intensity=surface_potentials,
                            colorscale='Viridis',
                            opacity=0.6,
                            name=f'Density {level:.2e}',
                            showscale=(i == 0),
                            colorbar=dict(title="Potential") if i == 0 else None
                        ))
                except Exception as e:
                    print(f"Skipping isosurface at level {level}: {e}")
            
            fig.update_layout(
                title="3D Gravitational (Low-Density Filtered)",
                scene=dict(
                    xaxis_title='X [Mpc/h]',
                    yaxis_title='Y [Mpc/h]',
                    zaxis_title='Z [Mpc/h]',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=800
            )
            
            if output_file:
                fig.write_html(output_file.replace('.png', '.html'))
                print(f"Interactive 3D plot saved to: {output_file.replace('.png', '.html')}")
            
            fig.show()
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly scikit-image")        
        
    def plot_potential_3d_volume(self, potential_result, output_file=None):
        """Create volume rendering of gravitational potential."""
        try:
            from mayavi import mlab
            
            potential_grid = potential_result['potential']
            
            # Normalize potential for better visualization
            pot_norm = (potential_grid - potential_grid.min()) / (potential_grid.max() - potential_grid.min())
            
            # Create volume rendering
            mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
            
            # Volume rendering
            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(pot_norm))
            vol.module_manager.scalar_lut_manager.lut.table = cm.viridis(np.linspace(0, 1, 256)) * 255
            
            # Add contour surfaces
            contours = mlab.contour3d(pot_norm, contours=8, transparent=True, opacity=0.4)
            
            # Add axes and labels
            mlab.axes(xlabel='X [Mpc/h]', ylabel='Y [Mpc/h]', zlabel='Z [Mpc/h]')
            mlab.colorbar(title='Normalized Potential')
            mlab.title('3D Gravitational Potential Volume')
            
            if output_file:
                mlab.savefig(output_file.replace('.png', '_3d.png'))
                print(f"3D volume plot saved to: {output_file.replace('.png', '_3d.png')}")
            
            mlab.show()
            
        except ImportError:
            print("Mayavi not available. Install with: pip install mayavi")
    
    def plot_potential_3d_slices(self, potential_result, output_file=None):
        """Create 3D visualization with multiple slices and projections."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LinearSegmentedColormap
        
        potential_grid = potential_result['potential']
        fig = plt.figure(figsize=(15, 12))
        
        # Create 2x2 subplot arrangement
        # Top left: 3D slices
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Sample slices through the volume
        nx, ny, nz = potential_grid.shape
        slice_indices = [nx//4, nx//2, 3*nx//4]
        
        x = np.linspace(0, self.boxsize, nx)
        y = np.linspace(0, self.boxsize, ny)
        z = np.linspace(0, self.boxsize, nz)
        
        X, Y = np.meshgrid(x, y)
        
        # Plot slices at different z-levels
        colors = ['blue', 'green', 'red']
        alphas = [0.4, 0.6, 0.4]
        
        for i, (slice_idx, color, alpha) in enumerate(zip(slice_indices, colors, alphas)):
            Z_slice = np.full_like(X, z[slice_idx])
            pot_slice = potential_grid[slice_idx, :, :]
            
            ax1.plot_surface(X, Y, Z_slice, 
                            facecolors=plt.cm.viridis(pot_slice/pot_slice.max()),
                            alpha=alpha, shade=False)
        
        ax1.set_xlabel('X [Mpc/h]')
        ax1.set_ylabel('Y [Mpc/h]')
        ax1.set_zlabel('Z [Mpc/h]')
        ax1.set_title('3D Potential Slices')
        
        # Top right: Maximum projection
        ax2 = fig.add_subplot(222)
        max_proj = np.max(potential_grid, axis=0)
        im2 = ax2.imshow(max_proj, origin='lower', cmap='viridis', 
                         extent=[0, self.boxsize, 0, self.boxsize])
        ax2.set_title('Maximum Projection (Z-axis)')
        ax2.set_xlabel('X [Mpc/h]')
        ax2.set_ylabel('Y [Mpc/h]')
        plt.colorbar(im2, ax=ax2)
        
        # Bottom left: Central slice with contours
        ax3 = fig.add_subplot(223)
        central_slice = potential_grid[nx//2, :, :]
        im3 = ax3.imshow(central_slice, origin='lower', cmap='viridis',
                         extent=[0, self.boxsize, 0, self.boxsize])
        
        # Add contour lines
        Y_2d, Z_2d = np.meshgrid(y, z)
        contours = ax3.contour(Y_2d, Z_2d, central_slice, levels=10, 
                              colors='white', alpha=0.6, linewidths=0.8)
        ax3.clabel(contours, inline=True, fontsize=8)
        
        ax3.set_title('Central Slice with Contours')
        ax3.set_xlabel('Y [Mpc/h]')
        ax3.set_ylabel('Z [Mpc/h]')
        plt.colorbar(im3, ax=ax3)
        
        # Bottom right: 1D profile through center
        ax4 = fig.add_subplot(224)
        
        # Extract 1D profiles through center
        center_idx = nx // 2
        profile_x = potential_grid[center_idx, center_idx, :]
        profile_y = potential_grid[center_idx, :, center_idx]
        profile_z = potential_grid[:, center_idx, center_idx]
        
        ax4.plot(z, profile_x, 'b-', label='X-direction', linewidth=2)
        ax4.plot(y, profile_y, 'g-', label='Y-direction', linewidth=2)
        ax4.plot(x, profile_z, 'r-', label='Z-direction', linewidth=2)
        
        ax4.set_xlabel('Distance [Mpc/h]')
        ax4.set_ylabel('Potential')
        ax4.set_title('1D Profiles Through Center')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file.replace('.png', '_3d_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"3D analysis plot saved to: {output_file.replace('.png', '_3d_analysis.png')}")
        
        plt.show()
    
    def plot_potential_3d_pyvista(self, potential_result, density_percentile=15, output_file=None):
        """Create advanced 3D visualization using PyVista."""
        try:
            import pyvista as pv
            
            potential_grid = potential_result['potential']
            density_grid = potential_result['density']
            
            # Ensure density_percentile is numeric
            density_percentile = float(density_percentile)
            
            # Create PyVista structured grid
            grid = pv.ImageData(dimensions=potential_grid.shape)
            grid.spacing = (self.boxsize/potential_grid.shape[0],) * 3
            grid.point_data['potential'] = potential_grid.flatten(order='F')
            grid.point_data['density'] = density_grid.flatten(order='F')
            
            # Calculate density threshold
            valid_density = density_grid[density_grid > 0]
            if len(valid_density) == 0:
                print("Warning: No positive density values found")
                density_threshold = density_grid.min()
            else:
                density_threshold = np.percentile(valid_density, density_percentile)
            
            print(f"Using density threshold: {density_threshold:.2e} (percentile: {density_percentile})")
            
            # Create plotter
            plotter = pv.Plotter(window_size=(1000, 800))
            
            # Method 1: Add volume rendering with opacity
            opacity = [0.0, 0.0, 0.1, 0.3, 0.7, 1.0]
            try:
                plotter.add_volume(grid, scalars='density', opacity=opacity, cmap='hot', 
                                scalar_bar_args={'title': 'Density'})
            except Exception as e:
                print(f"Volume rendering failed: {e}")
            
            # Method 2: FIXED - Handle MultiBlock slices properly
            try:
                # Create orthogonal slices
                slices = grid.slice_orthogonal(x=self.boxsize/2, y=self.boxsize/2, z=self.boxsize/2)
                
                # FIXED: Handle MultiBlock - extract and threshold each block
                if hasattr(slices, 'n_blocks') and slices.n_blocks > 0:
                    # Process each slice in the MultiBlock
                    for i in range(slices.n_blocks):
                        slice_block = slices[i]
                        if slice_block.n_points > 0:
                            # Apply threshold to individual slice
                            thresholded_slice = slice_block.threshold(density_threshold, scalars='density')
                            if thresholded_slice.n_points > 0:
                                plotter.add_mesh(thresholded_slice, opacity=0.8, cmap='viridis', 
                                            scalars='potential', name=f'Slice_{i}')
                else:
                    # Fallback: treat as single mesh
                    if slices.n_points > 0:
                        thresholded_slices = slices.threshold(density_threshold, scalars='density')
                        if thresholded_slices.n_points > 0:
                            plotter.add_mesh(thresholded_slices, opacity=0.8, cmap='viridis', 
                                        scalars='potential')
                            
            except Exception as e:
                print(f"Slice rendering failed: {e}")
            
            # Method 3: Add isosurfaces at meaningful density levels
            try:
                high_density_data = density_grid[density_grid > density_threshold]
                if len(high_density_data) > 0:
                    density_levels = np.percentile(high_density_data, [30, 60, 90])
                    colors = ['blue', 'green', 'red']
                    
                    for i, (level, color) in enumerate(zip(density_levels, colors)):
                        iso = grid.contour([level], scalars='density')
                        if iso.n_points > 0:
                            plotter.add_mesh(iso, opacity=0.4, color=color, 
                                        name=f'Density {level:.2e}')
            except Exception as e:
                print(f"Isosurface rendering failed: {e}")
            
            # Configure camera and scene
            try:
                plotter.camera_position = 'iso'
            except:
                # Fallback camera positions
                try:
                    plotter.camera_position = 'xy'
                except:
                    pass
            
            plotter.add_axes()
            plotter.add_text('3D Gravitational Potential (Filtered)', font_size=16)
            
            # Show and save
            if output_file:
                plotter.show(screenshot=output_file.replace('.png', '_pyvista.png'))
                print(f"PyVista 3D plot saved to: {output_file.replace('.png', '_pyvista.png')}")
            else:
                plotter.show()
                
        except ImportError:
            print("PyVista not available. Install with: pip install pyvista")
        except Exception as e:
            print(f"PyVista visualization failed: {e}")
            print("Falling back to interactive visualization...")
            self.plot_potential_3d_interactive(potential_result, density_percentile, output_file)
    
    
    