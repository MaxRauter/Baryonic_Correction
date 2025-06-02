import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from tqdm import tqdm
from MAS_library import MASL
from Pk_library import PKL  # Fix: Import from Pylians

def calc_concentration(M200, z):
    """
    Calculate halo concentration using the Duffy et al. (2008) relation.
    
    This function uses the concentration-mass relation for relaxed halos from 
    Duffy et al. 2008 to determine the concentration parameter of a dark matter halo.
    
    Parameters
    ----------
    M200 : float
        Halo mass M_200 in Msun/h (solar masses divided by the Hubble parameter)
    z : float
        Redshift
        
    Returns
    -------
    float
        Concentration parameter of the halo, constrained to be between 2.0 and 20.0
    
    Notes
    -----
    The relation is: c = A * ((M200 / Mpivot)**B) * (1 + z)**C
    where A=5.71, B=-0.084, C=-0.47, and Mpivot=2e12 Msun/h
    """
    if z is None:
        raise ValueError(f"Invalid z value in concentration calculation: {z}")  # Fixed: added 'raise'
    
    Mpivot = 2e12  # Msun/h
    A = 5.71
    B = -0.084
    C = -0.47
    c = A * ((M200 / Mpivot)**B) * (1 + z)**C
    c = min(max(c, 2.0), 20.0)
    return c

def calc_r_ej(r200, z=0, mu=0.5, Omega_m=0.3071, h=0.6777, theta=0.5):
    """
    Calculate the characteristic radius for ejected gas using Eqs. 2.22 and 2.23 from Schneider & Teyssier 2016.
    
    The ejection radius is given by:
    r_ej = mu * theta * r_200 * sqrt(delta_200)
    
    where:
    - delta_200 = 200 * (H(z)/H0)² * (1+z)³ / Omega_m
    - theta is a free parameter (default 0.5 from Schneider 2016)
    - mu is a scaling factor for the escape radius (default 0.5)
    
    Parameters
    ----------
    r200 : float
        Halo radius r_200 in Mpc/h
    z : float, optional
        Redshift, defaults to 0
    mu : float, optional
        Scaling factor for escape radius, defaults to 0.5
    Omega_m : float, optional
        Matter density parameter at z=0, defaults to 0.3071
    h : float, optional
        Hubble constant in units of 100 km/s/Mpc, defaults to 0.6777
    theta : float, optional
        Free parameter, default=0.5 (from Schneider & Teyssier 2016)
        
    Returns
    -------
    float
        Ejection radius r_ej in Mpc/h
    """
    Omega_Lambda = 1.0 - Omega_m
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    delta_200 = 200.0 * E_z**2 * (1 + z)**3 / Omega_m
    r_esc = theta * r200 * np.sqrt(delta_200)
    r_ej = r_esc * mu
    return r_ej

def calc_r_ej2(M200, r200, z=0, Omega_m=0.3071, Omega_Lambda=0.6929, h=0.6777, theta=4.0, delta_0=420.0, M_ej0=1.5e12):
    """
    Calculate the characteristic radius for ejected gas using Eqs. 2.22 and 2.23 from Schneider 2015.
    
    Parameters
    ----------
    M200 : float
        Halo mass M_200 in Msun/h
    r200 : float
        Halo radius r_200 in Mpc/h
    z : float, optional
        Redshift, defaults to 0
    Omega_m : float, optional
        Matter density parameter at z=0, defaults to 0.3071
    Omega_Lambda : float, optional
        Dark energy density parameter at z=0, defaults to 0.6929
    h : float, optional
        Hubble constant in units of 100 km/s/Mpc, defaults to 0.6777
    theta : float, optional
        Free parameter, default=4.0
    delta_0 : float, optional
        Characteristic density parameter, default=420.0
    M_ej0 : float, optional
        Characteristic ejected mass, default=1.5e12 Msun/h
    
    Returns
    -------
    float
        Ejection radius r_ej in Mpc/h
    """
    from . import abundance_fractions as fr
    fbar = Omega_m / (Omega_m + Omega_Lambda)
    f_bgas_val = fr.f_bgas(M200, fbar, z)
    f_cgal_val = fr.f_cgal(M200, z)
    f_egas_val = max(0.0, fbar - f_bgas_val - f_cgal_val)
    M_ej = f_egas_val * M200
    rho_crit_0 = 2.775e11
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    rho_crit_z = rho_crit_0 * E_z**2
    rho_m_z = Omega_m * rho_crit_0 * (1 + z)**3
    delta_200 = 200.0 * rho_crit_z / rho_m_z
    if M_ej > 0:
        r_ej = theta * r200 * np.sqrt(M_ej / M_ej0) * np.sqrt(delta_0 / delta_200)
    else:
        r_ej = 0.5 * r200
    return r_ej

def calc_R_h(M200, r200):
    """
    Calculate the half-light radius of the central galaxy based on halo mass.
    
    Uses observational scaling relations to determine a more accurate half-light radius
    that varies with halo mass. For low-mass halos, R_h is relatively larger compared 
    to r200, while for high-mass halos it's relatively smaller.
    
    Parameters
    ----------
    M200 : float
        Halo mass in Msun/h
    r200 : float
        Halo radius in Mpc/h
        
    Returns
    -------
    float
        Half-light radius R_h in Mpc/h
    
    Notes
    -----
    The base relation is R_h = 0.015 * r200, which is then scaled based on halo mass.
    """
    R_h_base = 0.015 * r200
    if M200 < 1e12:
        mass_factor = (M200 / 1e12)**(-0.1)
    elif M200 > 1e14:
        mass_factor = (M200 / 1e14)**(-0.05)
    else:
        mass_factor = 1.0
    return R_h_base * mass_factor

def bracket_rho0(M_target, r_s, r_tr, r200, r_max=None):
    """
    Solve for the NFW density normalization (rho0) that produces a specified enclosed mass.
    
    Uses the Brent method to find the value of rho0 such that the enclosed mass
    at radius r_max equals M_target.
    
    Parameters
    ----------
    M_target : float
        Target enclosed mass at r_max in Msun/h
    r_s : float
        Scale radius of the NFW profile in Mpc/h
    r_tr : float
        Truncation radius in Mpc/h (can be np.inf for untruncated profiles)
    r200 : float
        Halo radius r_200 in Mpc/h
    r_max : float, optional
        Radius at which the enclosed mass should match M_target, defaults to r200
        
    Returns
    -------
    float
        The normalized density parameter rho0 for the NFW profile
        
    Raises
    ------
    ValueError
        If the root finding algorithm cannot bracket the solution
    """
    from . import density_profiles as dp
    if r_max is None:
        r_max = r200
    def f(rho0):
        return dp.mass_profile(r_max, dp.rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr) - M_target
    low, high = 1e-12, 1e-8
    factor = 10
    for _ in range(50):
        high *= factor
        if f(low) * f(high) < 0:
            break
    if f(low) * f(high) > 0:
        raise ValueError("Could not bracket root for rho0.")
    rho0_solved = brentq(f, low, high)
    return rho0_solved

def normalize_component_total(density_func, args, M200_loc, r200_loc):
    """
    Normalize a density function so that its total mass out to a large radius equals M_target.
    
    Parameters
    ----------
    density_func : callable
        The density function to normalize
    args : tuple
        Arguments to pass to the density function
    M200_loc : float
        Target mass (M200) in Msun/h
    r200_loc : float
        Halo radius r200 in Mpc/h, used to determine the maximum integration radius
        
    Returns
    -------
    float
        The normalization factor to apply to the density function
    
    Notes
    -----
    The function integrates the unnormalized density profile out to 100*r200_loc
    to approximate the total mass to infinity.
    """
    from . import density_profiles as dp
    def unnorm_func(r):
        return density_func(r, *args)
    r_max = 100 * r200_loc
    unnorm_mass = dp.mass_profile(r_max, unnorm_func)
    return M200_loc / unnorm_mass

def normalize_component(density_func, args, M200_loc, r200_loc):
    """
    Calculate normalization factor to make a density component contain M200 within r200.
    
    Parameters
    ----------
    density_func : callable
        The density function to normalize
    args : tuple
        Arguments to pass to the density function
    M200_loc : float
        Target mass (M200) in Msun/h to contain within r200
    r200_loc : float
        Halo radius r200 in Mpc/h
        
    Returns
    -------
    float
        The normalization factor to apply to the density function
    """
    from . import density_profiles as dp
    def unnorm_func(r):
        return density_func(r, *args)
    r_max = r200_loc
    unnorm_mass = dp.mass_profile(r_max, unnorm_func)
    return M200_loc / unnorm_mass

def cumul_mass_old(r_array, rho_array):
    """
    Calculate the cumulative mass profile from a density profile.
    
    Integrates the density profile to compute the enclosed mass at each radius.
    Uses adaptive techniques to ensure numerical stability near the origin.
    
    Parameters
    ----------
    r_array : array_like
        Array of radii in Mpc/h
    rho_array : array_like
        Array of density values in Msun/h/Mpc³, corresponding to r_array
        
    Returns
    -------
    array_like
        Array of cumulative masses in Msun/h, corresponding to each radius in r_array
    
    Notes
    -----
    For improved numerical stability, the integration is performed using logarithmic
    spacing near the origin and adaptive methods for different radius ranges.
    """
    mass = np.zeros_like(r_array)
    r_min = 1e-8
    for i, r in enumerate(r_array):
        if r < r_min:
            mass[i] = 0.0
            continue
        if r > 10 * r_min:
            integration_points = np.logspace(np.log10(r_min), np.log10(r), 30)
            segment_masses = np.zeros(len(integration_points) - 1)
            for j in range(len(integration_points) - 1):
                r1, r2 = integration_points[j], integration_points[j + 1]
                segment_r = np.linspace(r1, r2, 20)
                segment_rho = np.interp(segment_r, r_array, rho_array)
                segment_integrand = 4 * np.pi * segment_r**2 * segment_rho
                segment_masses[j] = np.trapz(segment_integrand * segment_r, np.log(segment_r))
            mass[i] = np.sum(segment_masses)
        else:
            integrand = lambda s: 4 * np.pi * s**2 * np.interp(s, r_array, rho_array)
            mass[i], _ = quad(integrand, r_min, r, limit=200, epsabs=1e-8, epsrel=1e-8)
    return mass

def cumul_mass(r_array, rho_array):
    """
    Calculate the cumulative mass profile using logarithmic integration.
    
    For spherical symmetry: M(r) = ∫[0 to r] 4π r'² ρ(r') dr'
    Change variables: x = ln(r'), dx = dr'/r', so dr' = r' dx = e^x dx
    Then: M(r) = ∫[ln(r_min) to ln(r)] 4π e^(3x) ρ(e^x) dx
    
    Parameters
    ----------
    r_array : array_like
        Array of radii in Mpc/h
    rho_array : array_like
        Array of density values in Msun/h/Mpc³, corresponding to r_array
        
    Returns
    -------
    array_like
        Array of cumulative masses in Msun/h, corresponding to each radius in r_array
    
    Notes
    -----
    Uses logarithmic integration throughout for numerical stability and
    to avoid scipy.quad integration warnings.
    """
    r_array = np.asarray(r_array, dtype=float)
    rho_array = np.asarray(rho_array, dtype=float)
    mass = np.zeros_like(r_array)
    
    r_min = 1e-10
    
    # Filter valid data points
    valid_mask = (rho_array > 0) & np.isfinite(rho_array) & (r_array > 0) & np.isfinite(r_array)
    if not np.any(valid_mask):
        return mass
    
    r_valid = r_array[valid_mask]
    rho_valid = rho_array[valid_mask]
    
    # Sort by radius for proper interpolation
    sort_idx = np.argsort(r_valid)
    r_sorted = r_valid[sort_idx]
    rho_sorted = rho_valid[sort_idx]
    
    for i, r_max in enumerate(r_array):
        if r_max < r_min:
            mass[i] = 0.0
            continue
        
        # Adaptive number of integration points based on range
        log_range = np.log(r_max / r_min)
        n_points = max(50, min(200, int(log_range * 40)))
        
        # Create logarithmic integration grid
        log_r_min = np.log(r_min)
        log_r_max = np.log(r_max)
        log_r_grid = np.linspace(log_r_min, log_r_max, n_points)
        r_grid = np.exp(log_r_grid)
        
        # Safe interpolation with extrapolation handling
        rho_interp = np.zeros_like(r_grid)
        
        for j, r_val in enumerate(r_grid):
            if r_val < r_sorted[0]:
                # Power-law extrapolation at small radii: ρ ∝ r^(-1)
                rho_interp[j] = rho_sorted[0] * (r_val / r_sorted[0])**(-1)
            elif r_val > r_sorted[-1]:
                # Power-law extrapolation at large radii: ρ ∝ r^(-3)
                rho_interp[j] = rho_sorted[-1] * (r_val / r_sorted[-1])**(-3)
            else:
                rho_interp[j] = np.interp(r_val, r_sorted, rho_sorted)
        
        # Integrand in log space: 4π r³ ρ(r)
        # This accounts for the Jacobian dr = r d(ln r)
        integrand = 4 * np.pi * r_grid**3 * rho_interp
        
        # Filter out any problematic values
        valid_integrand = np.isfinite(integrand) & (integrand >= 0)
        
        if np.any(valid_integrand):
            # Trapezoidal integration in log space
            mass[i] = np.trapz(integrand[valid_integrand], 
                             log_r_grid[valid_integrand])
        else:
            mass[i] = 0.0
    
    return mass

def cumul_mass_single_old(r, rho_array, r_array):
    """
    Calculate cumulative mass from a density profile for a single radius value.
    
    Parameters
    ----------
    r : float
        The radius at which to compute the cumulative mass in Mpc/h
    rho_array : array_like
        The density profile values in Msun/h/Mpc³
    r_array : array_like
        The corresponding radius values for the density profile in Mpc/h
        
    Returns
    -------
    float
        The cumulative mass within radius r in Msun/h
    
    Notes
    -----
    Similar to cumul_mass but optimized for computing the mass at a single radius value.
    """
    r_min = 1e-8
    if r < r_min:
        return 0.0
    if r > 10 * r_min:
        integration_points = np.logspace(np.log10(r_min), np.log10(r), 30)
        segment_mass = 0.0
        for j in range(len(integration_points) - 1):
            r1, r2 = integration_points[j], integration_points[j + 1]
            segment_r = np.linspace(r1, r2, 20)
            segment_rho = np.interp(segment_r, r_array, rho_array)
            segment_integrand = 4 * np.pi * segment_r**2 * segment_rho
            segment_mass += np.trapz(segment_integrand *segment_r, np.log(segment_r))
        return segment_mass
    else:
        integrand = lambda s: 4 * np.pi * s**2 * np.interp(s, r_array, rho_array)
        mass, _ = quad(integrand, r_min, r, limit=200, epsabs=1e-8, epsrel=1e-8)
        return mass

def cumul_mass_single(r, rho_array, r_array):
    """
    Calculate cumulative mass using logarithmic integration.
    
    For spherical symmetry: M(r) = ∫[0 to r] 4π r'² ρ(r') dr'
    Change variables: x = ln(r'), dx = dr'/r', so dr' = r' dx = e^x dx
    Then: M(r) = ∫[ln(r_min) to ln(r)] 4π e^(3x) ρ(e^x) dx
    """
    r_min = 1e-8
    if r < r_min:
        return 0.0
    
    # Use logarithmic spacing
    log_r_min = np.log(r_min)
    log_r_max = np.log(r)
    log_r_points = np.linspace(log_r_min, log_r_max, 100)
    r_points = np.exp(log_r_points)
    
    # Interpolate density at these points
    rho_points = np.interp(r_points, r_array, rho_array)
    
    # Integrand in log space: 4π r³ ρ(r) (since dr = r dx in log space)
    integrand = 4 * np.pi * r_points**3 * rho_points
    
    # Integrate using trapezoidal rule in log space
    mass = np.trapz(integrand, log_r_points)
    
    return mass

def plot_bcm_profiles(r_vals, components, title=None, save_path=None):
    """
    Create a three-panel plot of BCM density profiles, cumulative mass, and displacement.
    
    Parameters
    ----------
    r_vals : array_like
        Radius values in Mpc/h
    components : dict
        Dictionary containing the following components:
            - M200 (float): Halo mass in Msun/h
            - r200 (float): Halo radius in Mpc/h
            - r_s (float): Scale radius in Mpc/h
            - rho_dmo (array): DM-only density profile
            - rho_bcm (array): Total BCM density profile
            - rho_bkg (array): Background density values
            - rdm (array): Relaxed DM density values (with fraction applied)
            - bgas (array): Bound gas density values (with fraction applied)
            - egas (array): Ejected gas density values (with fraction applied)
            - cgal (array): Central galaxy density values (with fraction applied)
            - M_dmo (array): DM-only cumulative mass profile
            - M_rdm (array): Relaxed DM cumulative mass profile
            - M_bgas (array): Bound gas cumulative mass profile
            - M_egas (array): Ejected gas cumulative mass profile
            - M_cgal (array): Central galaxy cumulative mass profile
            - M_bcm (array): Total BCM cumulative mass profile
            - M_bgk (array): Background cumulative mass values
            - disp (array): Displacement function values
    title : str, optional
        Custom title for the figure, defaults to auto-generated title with M200
    save_path : str, optional
        Path to save the figure, if not provided the figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    axes : array of matplotlib.axes.Axes
        The array of axes objects
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    M200 = components['M200']
    r200 = components['r200']
    r_s = components['r_s']
    ax1 = axes[0]
    ax1.loglog(r_vals, components['rho_dmo'], 'b-', label='NFW (DM-only)')
    ax1.loglog(r_vals, components['rdm'], 'b--', label='Relaxed DM (rdm)')
    ax1.loglog(r_vals, components['bgas'], 'g--', label='Bound gas (bgas)')
    ax1.loglog(r_vals, components['egas'], 'r--', label='Ejected gas (egas)')
    ax1.loglog(r_vals, components['cgal'], 'm--', label='Central galaxy (cgal)')
    ax1.loglog(r_vals, components['rho_bkg'], 'y--', label='Background')
    ax1.loglog(r_vals, components['rho_bcm'], 'r-', lw=2, label='Total bcm profile')
    ax1.axvline(r200, color='gray', linestyle=':', label='r200')
    ax1.axvline(r_s, color='gray', linestyle='--', label='r_s')
    ax1.set_xlabel("Radius [Mpc/h]")
    ax1.set_ylabel("Density [Msun/h/Mpc³]")
    ax1.set_title("Density Profiles")
    ax1.set_xlim([1e-2, 3e1])
    ax1.set_ylim([1e9, 7e16])
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.3)
    ax2 = axes[1]
    ax2.loglog(r_vals, components['M_dmo'], 'b-', label='NFW (DM-only)')
    ax2.loglog(r_vals, components['M_rdm'], 'b--', label='Relaxed DM (rdm)')
    ax2.loglog(r_vals, components['M_bgas'], 'g--', label='Bound gas (bgas)')
    ax2.loglog(r_vals, components['M_egas'], 'r--', label='Ejected gas (egas)')
    ax2.loglog(r_vals, components['M_cgal'], 'm--', label='Central galaxy (cgal)')
    ax2.loglog(r_vals, components['M_bcm'], 'r-', lw=2, label='Total bcm')
    ax2.loglog(r_vals, components['M_bkg'], 'y--', label='Background')
    ax2.axvline(r200, color='gray', linestyle=':', label='r200')
    ax2.set_xlabel("Radius [Mpc/h]")
    ax2.set_ylabel("Cumulative Mass [Msun/h]")
    ax2.set_title("Cumulative Mass Profiles")
    ax2.set_xlim(1e-2, 1e2)
    ax2.set_ylim(7e10, 7e15)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls=":", alpha=0.3)
    disp = components['disp']
    disp_pos = np.where(disp > 0, disp, 0)
    disp_neg = np.where(disp < 0, disp, 0)
    ax3 = axes[2]
    ax3.loglog(r_vals, np.abs(disp_pos), 'b-', lw=2, label='positive')
    ax3.loglog(r_vals, np.abs(disp_neg), 'b--', lw=2, label='negative')
    ax3.axvline(r200, color='gray', linestyle=':', label='r200')
    ax3.set_xlabel("Radius [Mpc/h]")
    ax3.set_ylabel("Displacement [Mpc/h]")
    ax3.set_title("Displacement Function")
    ax3.set_ylim(1e-4, 1.1)
    ax3.set_xlim(1e-2, 1e2)
    ax3.grid(True, which="both", ls=":", alpha=0.3)
    ax3.legend(fontsize=8)
    if title is None:
        title = f"Baryonic Correction Model for M200 = {M200:.2e} Msun/h"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes

def verify_schneider(verbose=False):
    """
    Verify that the implementation matches Schneider & Teyssier 2016 Fig 1 by plotting cases (a, b, c).
    
    Creates a figure showing density profiles, cumulative mass profiles, and displacement functions
    for the three test cases from the original paper, allowing for verification of the implementation.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print detailed information during calculations, defaults to False
        
    Returns
    -------
    None
        The function saves the output figure as "schneider_match.png"
    
    Notes
    -----
    This function reproduces Figure 1 from Schneider & Teyssier 2016 for verification purposes.
    It uses the default parameters and the case-specific parameters defined in parameters.py.
    """
    from Baryonic_Correction import simulations as sim
    from .parameters import DEFAULTS, CASE_PARAMS
    print("Verifying match to Schneider & Teyssier 2016 Fig 1")
    cases = list(CASE_PARAMS.keys())
    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 18))
    M200 = DEFAULTS['M200']
    r200 = DEFAULTS['r200']
    c = DEFAULTS['c']
    h = DEFAULTS['h']
    z = DEFAULTS['z']
    Omega_m = DEFAULTS['Omega_m']
    r_ej = DEFAULTS['r_ej_factor'] * r200
    R_h = DEFAULTS['R_h_factor'] * r200
    for i, case in enumerate(cases):
        print("\nPaper case ({})".format(case))
        case_params = CASE_PARAMS[case]
        f_rdm = case_params['f_rdm']
        bcm = sim.CAMELSReader(verbose=verbose)
        bcm.r_ej = r_ej
        bcm.R_h = R_h
        bcm.fbar = 1 - f_rdm
        bcm.init_calculations(
            M200=M200,
            r200=r200,
            c=c,
            h=h,
            z=z,
            Omega_m=Omega_m,
            f=case_params,
            verbose=verbose,
        )
        components = bcm.components
        r_vals = bcm.r_vals
        r_s = components['r_s']
        disp = components['disp']
        ax1 = axes[0, i]
        ax1.loglog(r_vals, components['rho_dmo'], 'b-', label='DM-only (NFW+Bkg)')
        ax1.loglog(r_vals, components['rdm'], 'b--', label='Relaxed DM (rdm)')
        ax1.loglog(r_vals, components['bgas'], 'g--', label='Bound gas (bgas)')
        ax1.loglog(r_vals, components['egas'], 'r--', label='Ejected gas (egas)')
        ax1.loglog(r_vals, components['cgal'], 'm--', label='Central galaxy (cgal)')
        ax1.loglog(r_vals, components['rho_bkg'], 'y--', label='Background')
        ax1.loglog(r_vals, components['rho_bcm'], 'r-', lw=2, label='Total bcm profile')
        ax1.axvline(r200, color='gray', linestyle=':', label='r200')
        ax1.axvline(r_s, color='gray', linestyle='--', label='r_s')
        ax1.set_xlabel("Radius [Mpc/h]")
        ax1.set_ylabel("Density [Msun/h/Mpc³]")
        ax1.set_title("Density Profiles")
        ax1.set_xlim([1e-2, 3e1])
        ax1.set_ylim([2e9, 7e16])
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", ls=":", alpha=0.3)
        ax2 = axes[1, i]
        ax2.loglog(r_vals, components['M_dmo'], 'b-', label='DM-only (NFW+Bkg)')
        ax2.loglog(r_vals, components['M_rdm'], 'b--', label='Relaxed DM (rdm)')
        ax2.loglog(r_vals, components['M_bgas'], 'g--', label='Bound gas (bgas)')
        ax2.loglog(r_vals, components['M_egas'], 'r--', label='Ejected gas (egas)')
        ax2.loglog(r_vals, components['M_cgal'], 'm--', label='Central galaxy (cgal)')
        ax2.loglog(r_vals, components['M_bcm'], 'r-', lw=2, label='Total bcm')
        ax2.loglog(r_vals, components['M_bkg'], 'y--', label='Background')
        ax2.axvline(r200, color='gray', linestyle=':', label='r200')
        ax2.set_xlabel("Radius [Mpc/h]")
        ax2.set_ylabel("Cumulative Mass [Msun/h]")
        ax2.set_title("Cumulative Mass Profiles")
        ax2.set_xlim(1e-2, 1e2)
        ax2.set_ylim(7e11, 7e15)
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", ls=":", alpha=0.3)
        ax3 = axes[2, i]
        disp_pos = np.where(disp > 0, disp, 0)
        disp_neg = np.where(disp < 0, disp, 0)
        ax3.loglog(r_vals, np.abs(disp_pos), 'b-', lw=2, label='positive')
        ax3.loglog(r_vals, np.abs(disp_neg), 'b--', lw=2, label='negative')
        ax3.axvline(r200, color='gray', linestyle=':', label='r200')
        ax3.set_xlabel("Radius [Mpc/h]")
        ax3.set_ylabel("Displacement [Mpc/h]")
        ax3.set_title("Displacement Function")
        ax3.set_ylim(1e-4, 1.1)
        ax3.set_xlim(1e-2, 1e2)
        ax3.grid(True, which="both", ls=":", alpha=0.3)
        ax3.legend(fontsize=8)
        print("\n")
    fig.suptitle("Comparison of Cases (a, b, c) from Schneider & Teyssier 2016", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("schneider_match.png", dpi=300, bbox_inches='tight')

def calc_power_spectrum(positions, box_size, mesh_size=512):
    """
    Calculate the matter power spectrum using FFT and NGP mass assignment.
    
    Parameters
    ----------
    positions : array_like
        (N, 3) array of particle positions in the simulation box
    box_size : float
        Size of the simulation box in Mpc/h
    mesh_size : int, optional
        Number of cells along each dimension for the density grid, defaults to 256
        
    Returns
    -------
    k_vals : array_like
        Wavenumbers in h/Mpc
    power_binned : array_like
        Power spectrum values P(k) in (Mpc/h)³
    
    Notes
    -----
    This implementation uses Nearest Grid Point (NGP) assignment to create 
    the density field before applying the FFT.
    """
    grid = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
    indices = np.floor(positions / box_size * mesh_size).astype(int) % mesh_size
    for p in tqdm(range(len(positions)), desc="Binning particles"):
        i, j, k = indices[p]
        grid[i, j, k] += 1
    mean_density = len(positions) / mesh_size**3
    grid = grid / mean_density - 1.0
    fft_grid = np.fft.fftn(grid)
    power = np.abs(fft_grid)**2
    kf = 2 * np.pi / box_size
    kx = kf * np.fft.fftfreq(mesh_size) * mesh_size
    ky = kf * np.fft.fftfreq(mesh_size) * mesh_size
    kz = kf * np.fft.fftfreq(mesh_size) * mesh_size
    kgrid = np.sqrt(kx[np.newaxis, np.newaxis, :]**2 + 
                    ky[np.newaxis, :, np.newaxis]**2 + 
                    kz[:, np.newaxis, np.newaxis]**2)
    kbins = np.linspace(0.1 * kf, kf * mesh_size / 2, 20)
    k_vals = 0.5 * (kbins[1:] + kbins[:-1])
    power_binned = np.zeros_like(k_vals)
    for i in range(len(kbins) - 1):
        mask = (kgrid >= kbins[i]) & (kgrid < kbins[i + 1])
        if np.sum(mask) > 0:
            power_binned[i] = np.mean(power[mask])
    return k_vals, power_binned

def calc_power_spectrum_new(positions, box_size, mass=None, mesh_size=512):
    """
    Calculate the matter power spectrum using FFT and NGP mass assignment.
    
    Parameters
    ----------
    positions : array_like
        (N, 3) array of particle positions in the simulation box
    box_size : float
        Size of the simulation box in Mpc/h
    mesh_size : int, optional
        Number of cells along each dimension for the density grid, defaults to 256
        
    Returns
    -------
    k_vals : array_like
        Wavenumbers in h/Mpc
    power_binned : array_like
        Power spectrum values P(k) in (Mpc/h)³
    
    Notes
    -----
    This implementation uses Nearest Grid Point (NGP) assignment to create 
    the density field before applying the FFT.
    """
    # Ensure proper data types for Pylians3
    positions = np.asarray(positions, dtype=np.float32)
    box_size = float(box_size)
    
    # For DM-only simulations, set equal masses if not provided
    if mass is None:
        # Option 1: Equal masses (most common for DM-only)
        mass = np.ones(len(positions), dtype=np.float32)
    
    delta = np.zeros((mesh_size, mesh_size, mesh_size), dtype=np.float32)
    MAS     = 'CIC'
    verbose = True
    MASL.MA(positions, delta, box_size, MAS, W=mass, verbose=verbose)
    delta /= np.mean(delta, dtype=np.float32)
    delta -= 1.0
    axis     = 0     #specifies the axes along which redshift-space distortions are place. Not used in real-space
    MAS      = 'CIC' #specifies the mass assignment scheme used to construct the density field
    threads  = 1     #number of openmp threads
    verbose  = True
    Pk_class = PKL.Pk(delta, box_size, axis, MAS, threads, verbose)
    k        = Pk_class.k3D
    Pk       = Pk_class.Pk[:,0]
    
    return k, Pk

def calc_power_spectrum2(positions, box_size, grid_size=512):
    """
    Compute the matter power spectrum from particle positions using FFT and CIC assignment.
    
    Parameters
    ----------
    positions : array_like
        (N, 3) array of particle positions in the simulation box in Mpc/h
    box_size : float
        Size of the simulation box in Mpc/h
    grid_size : int, optional
        Number of grid cells along each axis, defaults to 256
    
    Returns
    -------
    k_centers : array_like
        Array of wavenumbers in h/Mpc
    Pk : array_like
        Power spectrum values at k in (Mpc/h)³
    
    Notes
    -----
    This implementation uses Cloud-In-Cell (CIC) assignment to create 
    the density field before applying the FFT.
    """
    N = positions.shape[0]
    dx = box_size / grid_size
    rho = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    for pos in positions:
        ix = (pos / dx).astype(int) % grid_size
        rho[ix[0], ix[1], ix[2]] += 1
    rho_mean = np.mean(rho)
    delta = rho / rho_mean - 1.0
    delta_k = np.fft.fftn(delta)
    delta_k = np.fft.fftshift(delta_k)
    power_spectrum = np.abs(delta_k)**2
    k_vals = np.fft.fftfreq(grid_size, d=dx)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag = np.fft.fftshift(k_mag) * 2 * np.pi
    k_bins = np.linspace(0, np.max(k_mag), grid_size // 2)
    Pk = np.zeros_like(k_bins)
    counts = np.zeros_like(k_bins)
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        Pk[i] = np.mean(power_spectrum[mask]) if np.any(mask) else 0
        counts[i] = np.sum(mask)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    Pk = Pk[:-1]
    return k_centers, Pk

def compare_power_spectra(dmo_positions, bcm_positions, box_size, output_file=None):
    """
    Compare power spectra between DMO and BCM simulations and plot their ratio.
    
    Parameters
    ----------
    dmo_positions : array_like
        (N, 3) array of particle positions from the DMO simulation in Mpc/h
    bcm_positions : array_like
        (N, 3) array of particle positions from the BCM simulation in Mpc/h
    box_size : float
        Size of the simulation box in Mpc/h
    output_file : str, optional
        Path to save the output plot, if not provided the plot is not saved
        
    Returns
    -------
    k_dmo : array_like
        Wavenumbers for the DMO power spectrum in h/Mpc
    Pk_dmo : array_like
        Power spectrum values for DMO simulation in (Mpc/h)³
    k_bcm : array_like
        Wavenumbers for the BCM power spectrum in h/Mpc
    Pk_bcm : array_like
        Power spectrum values for BCM simulation in (Mpc/h)³
        
    Notes
    -----
    If the particle counts don't match, the function will truncate the larger dataset.
    """
    print("Comparing power spectra between DMO and BCM simulations")
    if len(dmo_positions) != len(bcm_positions):
        print("WARNING: Particle counts don't match! Power spectrum comparison may be invalid.")
        if len(bcm_positions) < len(dmo_positions):
            print(f"Using only the first {len(bcm_positions)} DMO particles to match BCM count")
            dmo_positions = dmo_positions[:len(bcm_positions)]
        else:
            print(f"Using only the first {len(dmo_positions)} BCM particles to match DMO count")
            bcm_positions = bcm_positions[:len(dmo_positions)]
    print("Calculating power spectrum for DMO")
    k_dmo, Pk_dmo = calc_power_spectrum_new(dmo_positions, box_size)
    print("Calculating power spectrum for BCM")
    k_bcm, Pk_bcm = calc_power_spectrum_new(bcm_positions, box_size)
    plt.figure(figsize=(10, 6))
    ratio = Pk_bcm / Pk_dmo
    plt.loglog(k_dmo, ratio, label='Ratio BCM/DMO', linestyle='--')
    plt.loglog(k_dmo, Pk_dmo, label='DMO Power Spectrum', color='blue')
    plt.loglog(k_bcm, Pk_bcm, label='BCM Power Spectrum', color='red')
    plt.title('Power Spectrum Comparison: DMO vs BCM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)³]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_file:
        plt.savefig(output_file)
    plt.show()
    return k_dmo, Pk_dmo, k_bcm, Pk_bcm

def plot_power_spectrum(k, Pk, title=None, save_path=None):
    """
    Plot a power spectrum on a log-log scale.
    
    Parameters
    ----------
    k : array_like
        Wavenumbers in h/Mpc
    Pk : array_like
        Power spectrum values in (Mpc/h)³
    title : str, optional
        Title for the plot, defaults to 'Power Spectrum'
    save_path : str, optional
        Path to save the figure, if not provided the figure is not saved
        
    Returns
    -------
    None
        The function displays the plot and optionally saves it
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(k, Pk)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)³]')
    plt.title(title if title else 'Power Spectrum')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def verify_schneider2(verbose=False):
    """
    Verify implementation against Schneider & Teyssier 2016 Fig 1 by plotting cases (a, b).
    
    Creates a simplified comparison of the density profiles and displacement functions
    for the first two test cases from the original paper.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print detailed information during calculations, defaults to False
        
    Returns
    -------
    None
        The function saves the output figure as "Schneider_match_4.png"
    
    Notes
    -----
    This function is a simplified version of verify_schneider that focuses only on
    the first two cases (a and b) and shows only density profiles and displacement functions.
    """
    from Baryonic_Correction import simulations as sim
    from .parameters import DEFAULTS, CASE_PARAMS
    print("Verifying match to Schneider & Teyssier 2016 Fig 1")
    cases = list(CASE_PARAMS.keys())[:2]
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 12))
    M200 = DEFAULTS['M200']
    r200 = DEFAULTS['r200']
    c = DEFAULTS['c']
    h = DEFAULTS['h']
    z = DEFAULTS['z']
    Omega_m = DEFAULTS['Omega_m']
    r_ej = DEFAULTS['r_ej_factor'] * r200
    R_h = DEFAULTS['R_h_factor'] * r200
    for i, case in enumerate(cases):
        print(f"\nPaper case ({case})")
        case_params = CASE_PARAMS[case]
        f_rdm = case_params['f_rdm']
        bcm = sim.CAMELSReader(verbose=verbose)
        bcm.r_ej = r_ej
        bcm.R_h = R_h
        bcm.fbar = 1 - f_rdm
        bcm.init_calculations(
            M200=M200,
            r200=r200,
            c=c,
            h=h,
            z=z,
            Omega_m=Omega_m,
            f=case_params,
            verbose=verbose
        )
        components = bcm.components
        r_vals = bcm.r_vals
        r_s = components['r_s']
        disp = components['disp']
        ax1 = axes[0, i]
        ax1.loglog(r_vals, components['rho_dmo'], 'b-', label='DM-only (NFW+Bkg)')
        ax1.loglog(r_vals, components['rdm'], 'b--', label='Relaxed DM (rdm)')
        ax1.loglog(r_vals, components['bgas'], 'g--', label='Bound gas (bgas)')
        ax1.loglog(r_vals, components['egas'], 'r--', label='Ejected gas (egas)')
        ax1.loglog(r_vals, components['cgal'], 'm--', label='Central galaxy (cgal)')
        ax1.loglog(r_vals, components['rho_bkg'], 'y--', label='Background')
        ax1.loglog(r_vals, components['rho_bcm'], 'r-', lw=2, label='Total bcm profile')
        ax1.axvline(r200, color='gray', linestyle=':', label='r200')
        ax1.axvline(r_s, color='gray', linestyle='--', label='r_s')
        ax1.set_xlabel("Radius [Mpc/h]")
        ax1.set_ylabel("Density [Msun/h/Mpc³]")
        ax1.set_title(f"Density Profiles - Case {case}")
        ax1.set_xlim([1e-2, 3e1])
        ax1.set_ylim([2e9, 7e16])
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", ls=":", alpha=0.3)
        ax2 = axes[1, i]
        disp_pos = np.where(disp > 0, disp, 0)
        disp_neg = np.where(disp < 0, disp, 0)
        ax2.loglog(r_vals, np.abs(disp_pos), 'b-', lw=2, label='positive')
        ax2.loglog(r_vals, np.abs(disp_neg), 'b--', lw=2, label='negative')
        ax2.axvline(r200, color='gray', linestyle=':', label='r200')
        ax2.set_xlabel("Radius [Mpc/h]")
        ax2.set_ylabel("Displacement [Mpc/h]")
        ax2.set_title(f"Displacement Function - Case {case}")
        ax2.set_ylim(1e-4, 1.1)
        ax2.set_xlim(1e-2, 1e2)
        ax2.grid(True, which="both", ls=":", alpha=0.3)
        ax2.legend(fontsize=8)
    fig.suptitle("Comparison of Cases (a, b) from Schneider & Teyssier 2016", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Schneider_match_4.png", dpi=300, bbox_inches='tight')

