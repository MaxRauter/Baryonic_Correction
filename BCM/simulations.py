import numpy as np
import os
import h5py
import glob
import hdf5plugin
import matplotlib.pyplot as plt
try:
    from Baryonic_Correction.BCM import density_profiles as dp
    from Baryonic_Correction.BCM import utils as ut
    from Baryonic_Correction.BCM import abundance_fractions as af
except ImportError:
    from BCM import density_profiles as dp
    from BCM import utils as ut
    from BCM import abundance_fractions as af


class CAMELSReader:
    """
    A class for reading and handling CAMELS simulation data.
    Stores important parameters from the simulations.
    """
    
    def __init__(self, path_group=None, path_snapshot=None, index = None, verbose = False):
        """
        Initialize the CAMELSReader with a path to CAMELS data.
        
        Parameters:
        -----------
        path : str
            Path to the directory containing the simulation data.
        """
        self.path_group = path_group
        self.path_snapshot = path_snapshot
        self.index = index
        self.verbose = verbose
        # Only load data if paths are provided
        if path_group:
            self._load_halodata()
        if path_snapshot:
            self._load_simdata()
        #self._load_particles()
        
    def _load_halodata(self):
        """
        Load a CAMELS simulation from the given path.
        
        Parameters:
        -----------
        path : str
            Path to the simulation directory.
            
        Returns:
        --------
        None
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
        Load a CAMELS simulation from the given path.
        
        Parameters:
        -----------
        path : str
            Path to the simulation directory.
            
        Returns:
        --------
        None
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
                        self.boxsize = header.attrs['BoxSize']
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
        if index is None:
            index = self.index
        offset = np.sum(self.halo['lentype_h'][:index], axis=0)[1] #this is the sum of the lengths of all FoF halos previous to the one we consider
        return offset
    
    def _load_particles(self):
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
                        'pos': parttype1['Coordinates'][start:stop],
                        'vel': parttype1['Velocities'][start:stop],
                        'id': parttype1['ParticleIDs'][start:stop],
                    })
        self.particles = particles
    
    def get_halo_particles(self, index = None):
        path = self.path_snapshot 
        print(f"Trying to open snapshot file: {path}")  # Add this line
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
                    'pos': parttype1['Coordinates'][start:stop],
                    'vel': parttype1['Velocities'][start:stop],
                    'id': parttype1['ParticleIDs'][start:stop],
                }
        return particles
    
    def get_halo_center(self, index=None):
        """Get the center of a halo."""
        if index is None:
            index = self.index
        if index is None or not self.halo:
            return None
        return self.halo['pos'][index]
    
    def get_particles_relative_position(self, index=None):
        """Get particle positions relative to halo center."""
        particles = self.get_halo_particles(index)
        if not particles:
            return None
            
        center = self.get_halo_center(index)
        rel_pos = particles['pos'] - center
        return rel_pos
    
    def plot_halo_masses_histogram(self, masslimit=None):
        """Plot a histogram of halo masses."""
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
    
    def init_calculations(self, M200=1e14, r200=0.77, c=None, h=0.6777, z=0, 
                 Omega_m=0.3071, f=None, verbose=False):
        """
        Initialize the Baryonic Correction Model for a given halo.
        
        Parameters:
            M200 (float): Halo mass in Msun/h
            r200 (float): Halo radius in Mpc/h
            c (float): Concentration parameter
            h (float): Hubble parameter 
            z (float): Redshift
            Omega_m (float): Matter density parameter
            f (str or list): Either a preset case ("a", "b", "c") or a list of 
                             abundance fractions [f_rdm, f_bgas, f_cgal, f_egas]
            verbose (bool): Whether to print detailed information
        """
        # Store input parameters
        self.M200 = self.halo['m200'][self.index] if self.index is not None else M200
        self.r200 = self.halo['r200'][self.index] if self.index is not None else r200
        self.h = h
        self.z = z
        self.Om = Omega_m
        self.Ol = 1 - Omega_m
        if c == None:
            c = ut.calc_concentration(self.M200,z)
        self.c = c
        self.fbar = self.Ob / Omega_m if hasattr(self, 'Ob') else self.fbar if hasattr(self, 'fbar') else 0.0483
        Omega_m = self.Om if hasattr(self, 'Om') else Omega_m
        self.verbose = verbose
        
        
        # Derived parameters
        self.r_s = r200 / c  # Scale radius for NFW profile
        self.r_tr = 8 * r200  # Truncation radius
        
        # Set abundance fractions
        self._set_abundance_fractions(f)
        
        # Calculate other parameters if not provided
        if not hasattr(self, 'r_ej'):
            self.r_ej = ut.calc_r_ej2(M200, r200, z=z, Omega_m=Omega_m, 
                                    Omega_Lambda=self.Ol, h=h)
        if not hasattr(self, 'R_h'):
            self.R_h = ut.calc_R_h(M200, r200)
        
        # Initialize component storage
        self.components = {}
        self.r_vals = None
        
        if self.verbose:
            self._print_parameters()
            
        self.calculate()
    
    def _set_abundance_fractions(self, f):
        """Set abundance fractions based on input."""
        # Check if f is list/dict or has to be calculated
        if (isinstance(f, list) and len(f) == 4) or (isinstance(f, dict) and len(f) == 4):
            # Custom abundance fractions
            if self.verbose:
                print("Using custom abundance fractions.")
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
            self.f_bgas = af.f_bgas(self.M200, self.fbar, self.z)
            self.f_cgal = af.f_cgal(self.M200, self.z)
            self.f_egas = af.f_egas(self.f_bgas,self.f_cgal,self.fbar)
        
        # Validate fractions sum to 1.0
        total = self.f_rdm + self.f_bgas + self.f_cgal + self.f_egas
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Abundance fractions do not sum to 1.0 but to {total:.6f}")
    
    def _print_parameters(self):
        """Print the model parameters."""
        print(f"BCM with M200 = {self.M200:.2e} Msun/h, r200 = {self.r200:.3f} Mpc/h, "
              f"c = {self.c:.2f}, h = {self.h:.3f}, z = {self.z:.2f}")
        print("Abundance fractions:")
        print(f"  f_rdm  = {self.f_rdm:.3f}")
        print(f"  f_bgas = {self.f_bgas:.3f}")
        print(f"  f_cgal = {self.f_cgal:.3f}")
        print(f"  f_egas = {self.f_egas:.3f}")
        print(f"Calculated NFW normalization: rho0 = {self.rho0:.3e}")
        
    def _print_components_at(self,r):
        """Print the calculated components at given radius."""
        print(f"Components at r = {r:.3f} Mpc/h:")
        print(f"  rho_dmo = {dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr):.3e}")
        print(f"  rho_bcm = {self.components['rho_bcm'][r]:.3e}")
        print(f"  rho_bkg = {self.components['rho_bkg'][r]:.3e}")
        print(f"  rho_rdm = {self.components['rdm'][r]:.3e}")
        print(f"  rho_bgas = {self.components['bgas'][r]:.3e}")
        print(f"  rho_egas = {self.components['egas'][r]:.3e}")
        print(f"  rho_cgal = {self.components['cgal'][r]:.3e}")
        
    def _create_radius_array(self, r_min, r_max, n_points):
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
        This is done by integrating the NFW profile over a large range.
        """
        # Integrate NFW profile over a large range to approximate total mass
        self.rho0 = ut.bracket_rho0(self.M200, self.r_s, self.r_tr, self.r200)
        rho_nfw = np.array([dp.rho_nfw(r, self.r_s, self.rho0, self.r_tr) for r in self.r_vals])
        M_nfw = ut.cumul_mass(self.r_vals, rho_nfw)
        self.fixed_M_tot = M_nfw[-1]
        return M_nfw

    def _calculate_normalizations(self):

        norm_bgas = ut.normalize_component_total(
            lambda r, r_s, r200, y0, c, rho0, r_tr: dp.y_bgas(r, r_s, r200, y0, c, rho0, r_tr), 
            (self.r_s, self.r200, 1.0, self.c, self.rho0, self.r_tr), self.f_bgas * self.fixed_M_tot, self.r200
        )
        norm_egas = ut.normalize_component_total(
            lambda r, M_tot, r_ej: dp.y_egas(r, M_tot, r_ej), 
            (1.0, self.r_ej), self.f_egas * self.fixed_M_tot, self.r200
        )
        norm_cgal = ut.normalize_component(
            lambda r, M_tot, R_h: dp.y_cgal(r, M_tot, R_h), 
            (1.0, self.R_h), self.f_cgal * self.fixed_M_tot, self.r200
        )
        
        return norm_bgas, norm_egas, norm_cgal

    def _calculate_normalizations_old(self):
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
        """Special normalization for cgal that uses a R_h-dependent integration range"""
        def unnorm_func(r):
            return density_func(r, 1.0, *args)
        
        # Use a much larger radius that adapts to the concentration
        r_max = 1000 * R_h  # This captures essentially all mass for a Hernquist profile
        unnorm_mass = dp.mass_profile(r_max, unnorm_func)
        
        return M_target / unnorm_mass
    
    def _compute_density_profiles(self, norm_bgas, norm_egas, norm_cgal, M_nfw):
        
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
        
        # baryon components for xi
        # Note: y_bgas, y_egas, and y_cgal are already normalized
        baryons = [(self.r_vals, y_bgas_vals), 
           (self.r_vals, y_egas_vals), 
           (self.r_vals, y_cgal_vals)]

        M_i = M_nfw
        
        
        # Calculate unnormalized profile
        rho_dm_contracted = dp.y_rdm_ac(self.r_vals, self.r_s, self.rho0, self.r_tr, 
                                    norm=1.0, a=0.68, f_cdm=0.839, 
                                    baryon_components=baryons)

        # Calculate total mass and correction factor
        M_contracted = ut.cumul_mass(self.r_vals, rho_dm_contracted)[-1]
        target_mass = self.f_rdm * self.fixed_M_tot
        correction_factor = target_mass / M_contracted

        print(f"RDM mass correction factor: {correction_factor:.4f}")

        # Apply correction
        rho_dm_contracted *= correction_factor
        y_rdm_vals = rho_dm_contracted
        rho_bcm = y_rdm_vals + y_bgas_vals + y_egas_vals + y_cgal_vals + rho_bkg_vals
        return rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm

    def _compute_mass_profiles(self, rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, M_nfw):
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
        if True:#self.verbose:
            #self._print_masses_at_r200()
            self._print_masses_at_infinity()
        return M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm

    def _invert_mass_profile(self, M_bcm):
        from scipy.interpolate import interp1d
        return interp1d(M_bcm, self.r_vals, bounds_error=False, fill_value="extrapolate")

    def _compute_displacement(self, M_dmo, f_inv_bcm):
        disp = np.zeros_like(self.r_vals)
        for i, r in enumerate(self.r_vals):
            M_target = M_dmo[i]
            r_bcm_val = f_inv_bcm(M_target)
            disp[i] = r_bcm_val - r
        return disp
    
    def _print_masses_at_infinity(self):
        """Print the masses of all components at infinity."""
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
        Calculate the RDM profile.
        """
        M_f = self.f_rdm * M_i + M_b
        # Calculate the RDM profile
        rho_rdm = dp.y_rdm_ac2(self.r_vals, self.r_s, self.rho0, self.r_tr,
                              M_i, M_f)
        
        # Calculate the total mass and correction factor
        M_contracted = ut.cumul_mass(self.r_vals, rho_rdm)[-1]
        target_mass = self.f_rdm * self.fixed_M_tot
        correction_factor = target_mass / M_contracted
        
        # Apply correction
        rho_rdm *= correction_factor
        
        # Calculate the total mass profile
        M_rdm = ut.cumul_mass(self.r_vals, rho_rdm)
        
        return rho_rdm, M_rdm
    
    def calculate(self, r_min=0.0001, r_max=2000, n_points=1000):
        """
        Calculate all BCM profiles and properties.
        """
        # Create a radius array
        self._create_radius_array(r_min, r_max, n_points)
        
        M_nfw = self._calc_NFW_target_mass()
        
        # Calculate normalizations
        norm_bgas, norm_egas, norm_cgal = self._calculate_normalizations()
        if self.verbose:
            print(f"Component normalizations to contain M200:")
            print(f"  bgas: {norm_bgas:.3e}")
            print(f"  egas: {norm_egas:.3e}")
            print(f"  cgal: {norm_cgal:.3e}")
        
        # Calculate density profiles
        rho_dmo_vals, rho_nfw_vals, rho_bkg_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, y_rdm_vals, rho_bcm = self._compute_density_profiles(norm_bgas, norm_egas, norm_cgal, M_nfw)
        
        # Calculate mass profiles
        M_dmo, M_bkg, M_rdm, M_bgas, M_egas, M_cgal, M_bcm = self._compute_mass_profiles(rho_dmo_vals, rho_bkg_vals, y_rdm_vals, y_bgas_vals, y_egas_vals, y_cgal_vals, rho_bcm, M_nfw)
        
        M_b = M_bgas + M_egas + M_cgal
        M_i = M_nfw
        # Calculate RDM profile
        rho_rdm_vals2, M_rdm_2 = self._calculate_rdm(M_i, M_b)
        
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
    
    
if __name__ == "__main__":
    #test1 = CAMELSReader(path_group = '/Users/Maxi/Desktop/Uni/Master/MasterArbeit/MasterArbeit/Camels/groups_014_dm.hdf5',path_snapshot = '/Users/Maxi/Desktop/Uni/Master/MasterArbeit/MasterArbeit/Camels/snapshot_014_dm.hdf5')
    #print(test1.particles[11]['pos'])
    
    #test2 = CAMELSReader(path_group = '/Users/Maxi/Desktop/Uni/Master/MasterArbeit/MasterArbeit/Camels/groups_014_dm.hdf5',path_snapshot = '/Users/Maxi/Desktop/Uni/Master/MasterArbeit/MasterArbeit/Camels/snapshot_014_dm.hdf5',index = 11)
    print()