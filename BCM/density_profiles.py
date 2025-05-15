import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from BCM.parameters import DEFAULTS

# --------------------------------------------------------------------
# Density Profiles
# --------------------------------------------------------------------

def rho_background(v,halo_masses):
    """
    Background density profile (Eq. 2.9).
    ρ_background(v) = ρ_c * Omega_m - (1/v) * Σ_i ρ_i
    where ρ_i is the density of each halo.
    This function computes the background density for a given halo mass.
    
    Parameters:
    - v: volume of the Simulation
    - halo_masses: list of halo masses
    """
    rho_c = 2.775e11 #h^2 M_sun / Mpc^3
    Omega_m = 0.3071
    return rho_c * Omega_m 
    #return 8.5e10  # Placeholder for background density

def rho_nfw(r, r_s, rho0, r_tr, r_0=1e-10):
    """
    Truncated NFW profile (Eq. 2.8)

    ρ_nfw(x, τ) = ρ0 / [ x (1+x)^2 (1 + (x/τ)^2)^2 ]
    with x = r/r_s and τ = r_tr/r_s.
    Best result with τ = 8c (Schneider & Teyssier 2016).

    Parameters:
    - r: radius
    - r_s: scale radius
    - rho0: normalization factor
    - r_tr: truncation radius
    - r_0: minimum radius to avoid singularity (default: 1e-10)
    """
    r = np.maximum(r, r_0)  # Ensure r is at least r_0 to avoid division by zero
    x = r / r_s
    tau = r_tr / r_s  # Corrected - tau is r_tr/r_s
    return rho0 / ( x * (1 + x)**2 * (1 + (x/tau)**2)**2 )

def mass_profile(r, density_func, **kwargs):
    """
    Computes the enclosed mass M(r) using Gauss-Legendre quadrature with log transform.
    """
    # For NFW with r_tr >> r_s, use analytical solution
    if density_func == rho_nfw and 'r_tr' in kwargs and kwargs['r_tr'] > 5*kwargs['r_s']:
        r_s = kwargs['r_s']
        rho0 = kwargs['rho0']
        x = r / r_s
        return 4 * np.pi * rho0 * r_s**3 * (np.log(1 + x) - x/(1 + x))
    
    r_min = 1e-8  # Lower minimum radius to capture more central mass
    
    # Use Gauss-Legendre quadrature with log transformation
    from scipy import special
    
    # Get Gauss-Legendre points and weights
    n_points = 48  # Higher number of points for better accuracy
    x, w = special.roots_legendre(n_points)
    
    # Transform to log space from r_min to r
    s_min = np.log(r_min)
    s_max = np.log(r)
    s = 0.5 * (s_max - s_min) * x + 0.5 * (s_max + s_min)
    radius = np.exp(s)
    
    # Calculate integrand at each point
    integrand = 4 * np.pi * radius**3 * np.array([density_func(r_i, **kwargs) for r_i in radius])
    
    # Apply weights and sum
    M = 0.5 * (s_max - s_min) * np.sum(w * integrand)
    
    return M

def mass_nfw_analytical(r, r_s, rho0):
    x = r/r_s
    return 4*np.pi*rho0*r_s**3*(np.log(1+x) - x/(1+x))

def calculate_total_mass(r_vals, rho_bcm):
    """
    Calculate the total mass by integrating the total density profile.
    
    Parameters:
        r_vals (array): Radius values in Mpc/h
        rho_bcm (array): Total density profile in Msun/h/Mpc³
    
    Returns:
        float: Total mass in Msun/h
    """
    # Integrate 4πr²ρ(r) over the radius range
    integrand = 4 * np.pi * r_vals**2 * rho_bcm
    M_total = np.trapz(integrand, r_vals)  # Use trapezoidal integration
    return M_total

def y_bgas(r, r_s, r200, y0, c, rho0, r_tr):
    sqrt5 = np.sqrt(5)
    r_transition = r200 / sqrt5
    x = r / r_s

    # Gamma_eff (Eq. 2.11)
    Gamma_eff = (1 + 3*c/sqrt5) * np.log(1 + c/sqrt5) / ((1 + c/sqrt5)*np.log(1 + c/sqrt5) - c/sqrt5)
    
    # Inner profile (Eq. 2.10)
    inner_val = y0 * (np.log(1 + x) / x)**Gamma_eff
    
    # Outer NFW profile
    outer_val = rho_nfw(r, r_s, y0, r_tr)
    
    # Value match at transition
    x_trans = r_transition / r_s
    inner_at_trans = y0 * (np.log(1 + x_trans) / x_trans)**Gamma_eff
    outer_at_trans = rho_nfw(r_transition, r_s, y0, r_tr)
    scale_factor = outer_at_trans / inner_at_trans
    #print(f"Scale factor: {scale_factor}")
    inner_scaled = inner_val * scale_factor
    outer_scaled = outer_val / scale_factor
    
    return np.where(r <= r_transition, inner_scaled, outer_val)

def y_egas(r, M_tot, r_ej):
    """
    Ejected gas profile (Eq. 2.13), modeled as a Gaussian.

    y_egas(r) = M_tot / ((2π r_ej^2)^(3/2)) * exp[- r^2 / (2 r_ej^2)]
    """
    norm = M_tot / ((2 * np.pi * r_ej**2)**(1.5))
    return norm * np.exp(- r**2 / (2 * r_ej**2))

def y_cgal(r, M_tot, R_h):
    """
    Central galaxy (stellar) profile (Eq. 2.14).

    A form based on observations:
    y_cgal(r) = M_tot / (4 π^(3/2) R_h) * (1/r^2) * exp[- (r/(2 R_h))^2]
    """
    # Add a tiny core radius to prevent division by zero
    r_safe = max(r, 1e-10)  # Prevent r=0 issues
    
    # Avoid underflow/overflow in the exponential
    exp_term = np.exp(-(r_safe/(2*R_h))**2)
    
    norm = M_tot / (4 * np.pi**1.5 * R_h)
    return norm * (1.0 / r_safe**2) * exp_term

def y_rdm_ac(r, r_s, rho0, r_tr, norm = 1.0,
             a=0.68,                # contraction strength
             f_cdm=0.839,             # CDM fraction of total mass
             baryon_components=None, # list of (r_array, y_vals) tuples
            verbose=False):
    """
    Adiabatically contracted DM profile with radius-dependent xi.

    baryon_components: e.g.
      [
        (r_array, y_bgas_vals),
        (r_array, y_bcg_vals),
        (r_array, y_egas_vals)
      ]
    """

    def xi_of_r(rf):
        import BCM.utils as ut
        # Solve rf/ri - 1 = a*(M_i(ri)/M_f(rf) - 1) for ri/rf = xi
        # where M_i is the mass before contraction and M_f is the mass after contraction
        # and rf is the radius at which we want to evaluate the density.

        # mass in baryons at rf
        M_b_rf = 0.0
        if baryon_components:
            for r_array, rho_array in baryon_components:
                baryon_mass = ut.cumul_mass_single(rf, rho_array, r_array)
                M_b_rf += baryon_mass
        #M_dm_initial_rf = mass_profile(rf, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
        
        def G(ri):
            # mass before contraction (2.16)
            M_i_ri = mass_profile(ri, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
            #f_cdm = DEFAULTS['f_cdm']
            f_cdm = M_i_ri / (M_i_ri + M_b_rf)  # CDM fraction of total mass
            # mass after contraction (2.16)
            M_f_rf = M_i_ri * f_cdm + M_b_rf  # Total mass after contraction
            #M_f_rf = M_dm_initial_rf + M_b_rf
            # Use a modified scaling that ensures some minimum contraction
            #a_min = 0.1 * a  # Minimum level of contraction (10% of normal)
            #scale_radius = 0.2 * r_s 
            #a_new = a_min + (a - a_min) * (1.0 - np.exp(-rf/scale_radius))
            
            a_new = a # set to a fixed value as Schneider
            
            # Avoid division by zero or negative mass
            if M_f_rf <= 1e-9: return np.nan
            
            ratio = M_i_ri / M_f_rf
            
            if ri < 1e-10: return np.inf
            
            return rf/ri - 1.0 - a_new*(ratio - 1.0)

        # bracket ri between a tiny inner radius and rf
        ri_min = max(rf*1e-5, 1e-8)  # Avoid division by zero
        ri_max = max(rf*1e4, r_tr*1.5)    # Don't go beyond truncation radius 
        
        try:
            g_min = G(ri_min)
            g_max = G(ri_max)

            # Handle potential NaN from G function (e.g., if M_f_rf was <= 0)
            if np.isnan(g_min) or np.isnan(g_max):
                 print(f"ERROR: G(ri) returned NaN at bounds for rf={rf:.3e}. ri_min={ri_min:.3e}, ri_max={ri_max:.3e}. Returning xi=1.0")
                 return 1.0

            if g_min * g_max >= 0:
                # Root not bracketed - THIS IS LIKELY THE PROBLEM AREA
                print(f"WARNING: Root not bracketed for rf={rf:.3e}. Bounds=[{ri_min:.3e}, {ri_max:.3e}].")
                print(f"  G(min)={g_min:.3e}, G(max)={g_max:.3e}")
                # Check if root is at the boundary
                if abs(g_min) < 1e-7:
                    print("  Root likely at ri_min.")
                    return rf / ri_min
                if abs(g_max) < 1e-7:
                    print("  Root likely at ri_max.")
                    return rf / ri_max

                # Analyze G's behavior: Is it always positive or always negative?
                # Calculate G at rf for reference: G(rf) should be -a*(M_nfw(rf)/M_f_rf - 1)
                g_at_rf = G(rf)
                print(f"  G(rf) = {g_at_rf:.3e}")
                if g_min > 0 and g_max > 0:
                    print("  G(ri) > 0 in bounds. Implies strong contraction (ri << rf) or issue.")
                elif g_min < 0 and g_max < 0:
                    print("  G(ri) < 0 in bounds. Implies strong expansion (ri >> rf) or issue.")

                # As a fallback, return 1.0, but signal that it's problematic
                print("  Returning default xi=1.0 due to bracketing issue.")
                return 1.0

            # If bracketed, proceed with root finding
            ri = brentq(G, ri_min, ri_max, xtol=1e-6, rtol=1e-6) # Added rtol
            xi = rf / ri

            if xi <= 0:
                 print(f"ERROR: Unphysical xi={xi:.3e} found for rf={rf:.3e}. ri={ri:.3e}. Returning 1.0")
                 return 1.0
            return xi

        except ValueError as e:
             # Catch errors during brentq execution
             print(f"ERROR: brentq failed for rf={rf:.3e}: {e}. Bounds=[{ri_min:.3e}, {ri_max:.3e}]. G(min)={g_min:.3e}, G(max)={g_max:.3e}. Returning xi=1.0")
             return 1.0

    
    # handle scalar or array r
    if np.isscalar(r):
        xi = xi_of_r(r)
        ri = r/xi
        return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)

    # vectorized
    out = np.zeros_like(r)
    xi_vals = np.zeros_like(r)
    for i, rf in enumerate(r):
        xi = xi_of_r(rf)
        xi = 1 if xi > 1 else xi
        diff = 1 - xi 
        #xi = 1 - diff/2
        xi_vals[i] = xi
        ri = rf/xi
        out[i] = norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
    if verbose:  
        print(f"xi_vals: {xi_vals[:10]}")
        idx_almost_one = np.argmax(np.isclose(xi_vals, 1.0, atol=1e-3))
        print(f"First xi ~ 1 at index: {idx_almost_one} and r = {r[idx_almost_one]}")
    return out

def y_rdm_ac2(r, r_s, rho0, r_tr, M_i, M_f,verbose, norm = 1.0,
             a=0.68,                # contraction strength
             f_cdm=0.839,             # CDM fraction of total mass
            ):
    """
    Adiabatically contracted DM profile with radius-dependent xi.

    baryon_components: e.g.
      [
        (y_bgas,    {'r_s':r_s, 'r200':r200, 'y0':y0, 'c':c, 'rho0':rho0, 'r_tr':r_tr}, f_bgas),
        (y_bcg,     {...}, f_bcg),
        ...
      ]
    """

    def xi_of_r(rf):
        # Solve rf/ri - 1 = a*(M_i(ri)/M_f(rf) - 1) for ri in (ε, rf)
        def G(ri):
            # Interpolate M_i(ri) and M_f(rf) from provided arrays
            M_i_interp = np.interp(ri, r, M_i)
            M_f_interp = np.interp(rf, r, M_f)
            return rf/ri - 1.0 - a*(M_i_interp/M_f_interp - 1.0)

        # bracket ri between a tiny inner radius and rf
        ri_min, ri_max = rf*1e-3, rf * 100
        
        # Check if solution exists in this interval
        g_min, g_max = G(ri_min), G(ri_max)
        if g_min * g_max >= 0:
            print(f"WARNING: Root not bracketed for rf={rf:.3e}. Bounds=[{ri_min:.3e}, {ri_max:.3e}].")
            return 1.0
        
        ri = brentq(G, ri_min, ri_max, xtol=1e-6, disp=True)
        return rf/ri

    # vectorized
    out = np.zeros_like(r)
    xi_vals = np.zeros_like(r)
    for i, rf in enumerate(r):
        xi = xi_of_r(rf)
        xi_vals[i] = xi
        ri = rf/xi
        out[i] = norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
    if verbose:
        print(f"xi_vals2: {xi_vals[:10]}")
        idx_almost_one = np.argmax(np.isclose(xi_vals, 1.0, atol=1e-3))
        print(f"First xi ~ 1 at index: {idx_almost_one} and r = {r[idx_almost_one]}")
    return out

def y_rdm_fixed_xi(r, r_s, rho0, r_tr, xi=0.85, norm=1.0,
             a=0.68,                # contraction strength
             f_cdm=0.839,             # CDM fraction of total mass
             baryon_components=None # list of (r_array, y_vals) tuples
            ):
    """
    Adiabatically contracted DM profile with a fixed xi value for all radii.

    Parameters:
        r (float or array): Radius or array of radii.
        r_s (float): Scale radius.
        rho0 (float): Normalization factor.
        r_tr (float): Truncation radius.
        xi (float): Fixed contraction parameter.
        norm (float): Normalization factor.

    Returns:
        float or array: Contracted DM density profile.
    """
    limit = 0.04
    inner_xi = 0.65
    if r < limit:
        # Interpolate xi nonlinearly: slow near limit, faster near 0 (e.g., quadratic)
        xi_interp = 1 - ((1 - inner_xi)) * ((1 - r / limit))
        xi = np.clip(xi_interp, inner_xi, 1.0)
        print(f"xi: {xi} for r: {r}")
    else:
        xi = 1
    ri = r / xi
    #print(f"xi: {xi}")
    return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
