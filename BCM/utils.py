import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def calc_concentration(M200, z):
    """Calculate halo concentration using the Duffy et al. (2008) relation"""
    # Convert mass to h^-1 units if needed (M200 should already be in Msun/h)
    Mpivot = 2e12  # Msun/h
    
    # Parameters for relaxed halos, from Duffy et al. 2008
    A = 5.71
    B = -0.084
    C = -0.47
    
    # Calculate concentration
    c = A * ((M200 / Mpivot)**B) * (1 + z)**C
    
    # Apply reasonable limits to catch outliers
    c = min(max(c, 2.0), 20.0)
    
    return c

def calc_r_ej(r200, z=0, mu = 0.5, Omega_m=0.3071, h=0.6777, theta=0.5):
    """
    Calculate the characteristic radius for ejected gas using Eqs. 2.22 and 2.23 from Schneider 2016.
    
    The ejection radius is given by:
    r_ej = theta * r_200 * sqrt(delta_200)
    
    where:
    - delta_200 = 200 * (H(z)/H0)² * (1+z)³ / Omega_m
    - theta is a free parameter (default 0.5 from Schneider 2016)
    
    Parameters:
        M200 (float): Halo mass M_200 in Msun/h
        r200 (float): Halo radius r_200 in Mpc/h
        z (float): Redshift
        Omega_m (float): Matter density parameter at z=0
        h (float): Hubble constant in units of 100 km/s/Mpc
        theta (float): Free parameter, default=0.5 (from paper)
        
    Returns:
        float: Ejection radius r_ej in Mpc/h
    """
    # Calculate E(z) = H(z)/H0 for a flat universe
    Omega_Lambda = 1.0 - Omega_m  # Assuming flat universe
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    
    # Calculate delta_200 using Eq. 2.23
    delta_200 = 200.0 * E_z**2 * (1 + z)**3 / Omega_m
    
    # Calculate r_esc using Eq. 2.22
    r_esc = theta * r200 * np.sqrt(delta_200)
    
    # Calculate the ejection radius r_ej Eq. 2.23
    r_ej = r_esc * mu
    
    return r_ej

def calc_r_ej2(M200, r200, z=0, Omega_m=0.3071, Omega_Lambda=0.6929, h=0.6777, theta=4.0, delta_0=420.0, M_ej0=1.5e12):
    """
    Calculate the characteristic radius for ejected gas using Eqs. 2.22 and 2.23 from Schneider 2015.
    
    Parameters:
        M200 (float): Halo mass M_200 in Msun/h
        r200 (float): Halo radius r_200 in Mpc/h
        z (float): Redshift
        Omega_m (float): Matter density parameter at z=0
        Omega_Lambda (float): Dark energy density parameter at z=0
        h (float): Hubble constant in units of 100 km/s/Mpc
        theta (float): Free parameter, default=4.0
        delta_0 (float): Characteristic density parameter, default=420.0
        M_ej0 (float): Characteristic ejected mass, default=1.5e12 Msun/h
    
    Returns:
        float: Ejection radius r_ej in Mpc/h
    """
    from . import abundance_fractions as fr
    fbar = Omega_m / (Omega_m + Omega_Lambda)  # Cosmic baryon fraction
    # Calculate ejected gas mass
    f_bgas_val = fr.f_bgas(M200,fbar,z)
    f_cgal_val = fr.f_cgal(M200,z)
    f_egas_val = max(0.0, fbar - f_bgas_val - f_cgal_val)
    M_ej = f_egas_val * M200
    
    # Calculate critical density at redshift z
    # ρ_crit(z) = 3H(z)²/(8πG)
    # For H(z) = H₀ × E(z) where E(z) = √(Ω_m(1+z)³ + Ω_Λ)
    # ρ_crit,0 = 2.775e11 M⊙/h / (Mpc/h)³
    rho_crit_0 = 2.775e11  # M⊙/h / (Mpc/h)³
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
    rho_crit_z = rho_crit_0 * E_z**2
    
    # Calculate mean matter density at redshift z
    # ρ_m(z) = ρ_m,0 × (1+z)³ = Ω_m × ρ_crit,0 × (1+z)³
    rho_m_z = Omega_m * rho_crit_0 * (1 + z)**3
    
    # Calculate delta_200 using Eq. 2.23
    # δ_200(M_200) = 200 × ρ_crit(z) / ρ_m(z)
    delta_200 = 200.0 * rho_crit_z / rho_m_z
    
    # At z=0, this simplifies to delta_200 = 200/Omega_m
    # For a cross-check: delta_200_simple = 200.0 / Omega_m
    
    # Calculate r_ej using Eq. 2.22
    if M_ej > 0:
        r_ej = theta * r200 * np.sqrt(M_ej / M_ej0) * np.sqrt(delta_0 / delta_200)
    else:
        # If no ejected gas, set r_ej to a small multiple of r200
        r_ej = 0.5 * r200
        
    return r_ej

def calc_R_h(M200, r200):
    """
    Calculate the half-light radius of the central galaxy based on halo mass.
    
    Uses observational scaling relations to determine a more accurate R_h.
    
    Parameters:
        M200: Halo mass in Msun/h
        r200: Halo radius in Mpc/h
        
    Returns:
        R_h: Half-light radius in Mpc/h
    """
    # Base scaling with r200
    R_h_base = 0.015 * r200
    
    # Mass-dependent adjustment
    # For low-mass halos, R_h is relatively larger compared to r200
    # For high-mass halos, R_h is relatively smaller compared to r200
    if M200 < 1e12:  # Dwarf galaxy territory
        mass_factor = (M200/1e12)**(-0.1)  # Relatively larger R_h at low mass
    elif M200 > 1e14:  # Cluster territory
        mass_factor = (M200/1e14)**(-0.05)  # Slightly smaller R_h at high mass
    else:
        mass_factor = 1.0
        
    return R_h_base * mass_factor

def bracket_rho0(M_target, r_s, r_tr,r200, r_max=None):
        """Solve for rho0 so that the enclosed mass equals M_target at r200"""
        from . import density_profiles as dp
        if r_max is None:
            r_max = r200
        
        def f(rho0):
            return dp.mass_profile(r_max, dp.rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr) - M_target
        
        # Initialize search with wide range
        low, high = 1e-12, 1e-8
        factor = 10
        for _ in range(50):
            high *= factor
            if f(low)*f(high) < 0:
                break
        if f(low)*f(high) > 0:
            raise ValueError("Could not bracket root for rho0.")
        
        rho0_solved = brentq(f, low, high)
        return rho0_solved

def normalize_component_total(density_func, args, M200_loc, r200_loc):
    """Normalize so that the total mass to infinity is M_target."""
    from . import density_profiles as dp
    def unnorm_func(r):
        return density_func(r, *args)
    r_max = 100 * r200_loc  # Large enough to capture all mass
    unnorm_mass = dp.mass_profile(r_max, unnorm_func)
    return M200_loc / unnorm_mass

def normalize_component(density_func, args, M200_loc, r200_loc):
    """Calculate normalization to make the component contain M200 over all radii"""
    from . import density_profiles as dp
    # First calculate the unnormalized mass profile over all radii
    def unnorm_func(r):
        return density_func(r, *args)
    
    # Integrate from r = 0 to r = ∞ (use a large upper limit to approximate ∞)
    r_max = r200_loc  # Use a sufficiently large radius to approximate infinity
    unnorm_mass = dp.mass_profile(r_max, unnorm_func)
    
    # Return factor to scale to M200
    return M200_loc / unnorm_mass

# Calculate cumulative masses
def cumul_mass(r_array, rho_array):
    """Calculates cumulative mass profile from density profile"""
    mass = np.zeros_like(r_array)
    
    # Use a small but non-zero minimum radius to avoid singularity
    r_min = 1e-6
    
    for i, r in enumerate(r_array):
        if r < r_min:
            mass[i] = 0.0
            continue
            
        # Split the integration into smaller segments for better numerical stability
        if r > 10*r_min:
            # Use logarithmic spacing for integration points between r_min and r
            # This gives more points near the origin where the density changes rapidly
            integration_points = np.logspace(np.log10(r_min), np.log10(r), 30)
            
            # Integrate over each segment
            segment_masses = np.zeros(len(integration_points)-1)
            for j in range(len(integration_points)-1):
                r1, r2 = integration_points[j], integration_points[j+1]
                # For each segment, use Simpson's rule with many points
                segment_r = np.linspace(r1, r2, 20)
                segment_rho = np.interp(segment_r, r_array, rho_array)
                segment_integrand = 4 * np.pi * segment_r**2 * segment_rho
                segment_masses[j] = np.trapz(segment_integrand, segment_r)
                
            mass[i] = np.sum(segment_masses)
        else:
            # For very small radii, use straight quad with larger tolerance
            integrand = lambda s: 4 * np.pi * s**2 * np.interp(s, r_array, rho_array)
            mass[i], _ = quad(integrand, r_min, r, limit=200, epsabs=1e-8, epsrel=1e-8)
    
    return mass

def cumul_mass_single(r, rho_array, r_array):
    """
    Calculates cumulative mass profile from density profile for a single radius value.

    Parameters:
        r (float): The radius at which to compute the cumulative mass.
        rho_array (array): The density profile values.
        r_array (array): The corresponding radius values for the density profile.

    Returns:
        float: The cumulative mass within radius r.
    """
    r_min = 1e-6
    if r < r_min:
        return 0.0

    if r > 10 * r_min:
        integration_points = np.logspace(np.log10(r_min), np.log10(r), 30)
        segment_mass = 0.0
        for j in range(len(integration_points) - 1):
            r1, r2 = integration_points[j], integration_points[j + 1]
            segment_r = np.linspace(r1, r2, 20)
            segment_rho = np.interp(segment_r, r_array, rho_array)
            segment_integrand = 4 * np.pi * segment_r ** 2 * segment_rho
            segment_mass += np.trapz(segment_integrand, segment_r)
        return segment_mass
    else:
        integrand = lambda s: 4 * np.pi * s ** 2 * np.interp(s, r_array, rho_array)
        mass, _ = quad(integrand, r_min, r, limit=200, epsabs=1e-8, epsrel=1e-8)
        return mass

# Plot function
def plot_bcm_profiles(r_vals, components, title=None, save_path=None):
    """
    Create a three-panel plot of BCM density profiles, cumulative mass, and displacement.
    
    Parameters:
        r_vals (array): Radius values in Mpc/h
        components (dict): Dictionary containing:
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
        title (str, optional): Custom title for the figure
        save_path (str, optional): Path to save the figure
        
    Returns:
        fig, axes: The matplotlib figure and axes objects
    """
    # Create a single figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract components
    M200 = components['M200']
    r200 = components['r200']
    r_s = components['r_s']
    
    # 1. DENSITY PROFILES PLOT (left subplot)
    # ======================================
    ax1 = axes[0]
    ax1.loglog(r_vals, components['rho_dmo'], 'b-', label='NFW (DM-only)')
    ax1.loglog(r_vals, components['rdm'], 'b--', label='Relaxed DM (rdm)')
    ax1.loglog(r_vals, components['bgas'], 'g--', label='Bound gas (bgas)')
    ax1.loglog(r_vals, components['egas'], 'r--', label='Ejected gas (egas)')
    ax1.loglog(r_vals, components['cgal'], 'm--', label='Central galaxy (cgal)')
    ax1.loglog(r_vals, components['rho_bkg'], 'y--', label='Background')
    ax1.loglog(r_vals, components['rho_bcm'], 'r-', lw=2, label='Total bcm profile')
    
    # Add reference lines
    ax1.axvline(r200, color='gray', linestyle=':', label='r200')
    ax1.axvline(r_s, color='gray', linestyle='--', label='r_s')
    
    ax1.set_xlabel("Radius [Mpc/h]")
    ax1.set_ylabel("Density [Msun/h/Mpc³]")
    ax1.set_title("Density Profiles")
    ax1.set_xlim([1e-2, 3e1])
    ax1.set_ylim([1e9, 7e16])
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.3)
    
    # 2. CUMULATIVE MASS PLOT (middle subplot)
    # =======================================
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
    
    # 3. DISPLACEMENT FUNCTION PLOT (right subplot)
    # ============================================
    # Split displacement into positive and negative parts
    disp = components['disp']
    disp_pos = np.where(disp > 0, disp, 0)
    disp_neg = np.where(disp < 0, disp, 0)

    # Row 3: Displacement function (positive and negative parts)
    ax3 = axes[2]
    ax3.loglog(r_vals, np.abs(disp_pos), 'b-', lw=2, label='positive')
    ax3.loglog(r_vals, np.abs(disp_neg), 'b--', lw=2, label='negative')
    ax3.axvline(r200, color='gray', linestyle=':', label='r200')
    ax3.set_xlabel("Radius [Mpc/h]")
    ax3.set_ylabel("Displacement [Mpc/h]")
    ax3.set_title("Displacement Function")
    ax3.set_ylim(1e-4, 1.1)  # Adjust y-limits to show small displacements
    ax3.set_xlim(1e-2, 1e2)
    ax3.grid(True, which="both", ls=":", alpha=0.3)
    ax3.legend(fontsize=8)

    # Add a common title for the entire figure
    if title is None:
        title = f"Baryonic Correction Model for M200 = {M200:.2e} Msun/h"
    fig.suptitle(title, fontsize=14)
    
    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    
    # Save the figure if desired
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def verify_schneider():
    from BCM import simulations as sim
    from .parameters import DEFAULTS, CASE_PARAMS
    """Verify that our implementation matches Schneider & Teyssier 2016 Fig 1 by plotting cases (a, b, c)."""
    print("Verifying match to Schneider & Teyssier 2016 Fig 1")
    cases = list(CASE_PARAMS.keys())
    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 18))
    
    # global defaults
    M200 = DEFAULTS['M200']
    r200 = DEFAULTS['r200']
    c = DEFAULTS['c']
    h = DEFAULTS['h']
    z = DEFAULTS['z']
    Omega_m = DEFAULTS['Omega_m']
    r_ej = DEFAULTS['r_ej_factor'] * r200
    R_h = DEFAULTS['R_h_factor'] * r200

    for i, case in enumerate(cases):
        # Run BCM calculation for each case
        print(fr"Paper carse ({case}):")
        case_params = CASE_PARAMS[case]
        f_rdm = case_params['f_rdm']
        bcm = sim.CAMELSReader()
        bcm.r_ej = r_ej
        bcm.R_h = R_h
        bcm.init_calculations(
            M200=M200,
            r200=r200,
            c=c,
            h=h,
            z=z,
            Omega_m=Omega_m,
            f=case_params,
            verbose=False
        )
        bcm.fbar = 1 - f_rdm
        bcm.calculate()
        components = bcm.components
        r_vals = bcm.r_vals
        r_s = components['r_s']
        disp = components['disp']

        # Density profiles
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
        # Cumulative mass profiles
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
        # Displacement function
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
    fig.suptitle("Comparison of Cases (a, b, c) from Schneider & Teyssier 2016", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("schneider_match.png", dpi=300, bbox_inches='tight')
    #plt.show()