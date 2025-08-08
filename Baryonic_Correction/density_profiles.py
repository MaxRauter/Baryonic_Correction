import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy import special
from Baryonic_Correction import parameters as par

# --------------------------------------------------------------------
# Density Profiles
# --------------------------------------------------------------------

def rho_background(v, halo_masses):
    """
    Calculate the background density profile.
    
    This function computes the cosmic background density using the critical 
    density and matter density parameter. The formula is given by Eq. 2.9:
    ρ_background(v) = ρ_c * Omega_m - (1/v) * Σ_i ρ_i
    where ρ_i is the density of each halo.
    
    Parameters
    ----------
    v : float
        Volume of the simulation in (Mpc/h)^3
    halo_masses : array_like
        List of halo masses in Msun/h
        
    Returns
    -------
    float
        Background density in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    Currently uses fixed values for critical density and Omega_m.
    """
    rho_c = 2.775e11 #h^2 M_sun / Mpc^3
    Omega_m = 0.3071
    return rho_c * Omega_m 

def truncated_nfw_integrand(x, tau):
    return x / ((1 + x)**2 * (1 + (x / tau)**2)**2)

def M_tot_truncated(rho0, rs, rtr):
    tau = rtr / rs
    integral, _ = quad(truncated_nfw_integrand, 0, tau, args=(tau,))
    return 4 * np.pi * rho0 * rs**3 * integral

def M_tot_simple(rho0, r_s, r_tr):
    integrand = lambda r: 4 * np.pi * r**2 * rho_nfw(r, r_s, rho0, r_tr)
    M, _ = quad(integrand, 0, r_tr)
    return M

def rho_nfw(r, r_s, rho0, r_tr, r_0=1e-10):
    """
    Calculate the truncated NFW density profile.
    
    Implements the truncated NFW profile from Eq. 2.8:
    ρ_nfw(x, τ) = ρ0 / [x (1+x)^2 (1 + (x/τ)^2)^2]
    where x = r/r_s and τ = r_tr/r_s.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    r_0 : float, optional
        Minimum radius to avoid singularity, default 1e-10 Mpc/h
    
    Returns
    -------
    float or array_like
        NFW density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    Best results are achieved with τ = r_tr/r_s = 8c (Schneider & Teyssier 2016).
    """
    r = np.maximum(r, r_0)  # Ensure r is at least r_0 to avoid division by zero
    x = r / r_s
    tau = r_tr / r_s 
    return rho0 / ( x * (1 + x)**2 * (1 + (x/tau)**2)**2 )

def mass_profile(r, density_func, **kwargs):
    """
    Compute the enclosed mass within radius r for a given density profile.
    
    This function uses Gauss-Legendre quadrature with log-space transformation
    to efficiently calculate the enclosed mass by integrating 4πr²ρ(r) from a 
    small inner radius to r.
    
    Parameters
    ----------
    r : float
        Radius within which to compute the enclosed mass, in Mpc/h
    density_func : callable
        Function that returns the density at a given radius
    **kwargs : dict
        Additional arguments to pass to the density function
    
    Returns
    -------
    float
        Enclosed mass in Msun/h
    
    Notes
    -----
    For NFW profiles with r_tr >> r_s, this function automatically uses the
    analytical solution for better accuracy and performance.
    """
    """if density_func == rho_nfw and 'r_tr2' in kwargs and kwargs['r_tr'] > 5*kwargs['r_s']:
        r_s = kwargs['r_s']
        rho0 = kwargs['rho0']
        x = r / r_s
        return 4 * np.pi * rho0 * r_s**3 * (np.log(1 + x) - x/(1 + x))
    
    if density_func == mass_nfw_analytical_inf and 'r_tr' in kwargs and kwargs['r_tr'] > 5*kwargs['r_s']:
        r_s = kwargs['r_s']
        rho0 = kwargs['rho0']
        r_tr = kwargs['r_tr']
        return mass_nfw_analytical_inf(r_tr, r_s, rho0)
    """
    r_min = 1e-8  # Lower minimum radius to capture more central mass
    
    n_points = 48  # Higher number of points for better accuracy
    x, w = special.roots_legendre(n_points)
    
    s_min = np.log(r_min)
    s_max = np.log(r)
    s = 0.5 * (s_max - s_min) * x + 0.5 * (s_max + s_min)
    radius = np.exp(s)
    
    integrand = 4 * np.pi * radius**3 * np.array([density_func(r_i, **kwargs) for r_i in radius])
    
    M = 0.5 * (s_max - s_min) * np.sum(w * integrand)
    
    return M

def mass_nfw_analytical_inf(r_tr, r_s, rho0):
    """
    Calculate the enclosed mass for an NFW profile using the analytical formula.
    
    Parameters
    ----------
    r : float
        Radius within which to compute the enclosed mass, in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    
    Returns
    -------
    float
        Enclosed mass in Msun/h
    
    Notes
    -----
    Uses the formula: M(r) = 4πρ0r_s³(ln(1+x) - x/(1+x)) where x = r/r_s
    This is valid for a standard (non-truncated) NFW profile.
    """
    t = r_tr / r_s  # Truncation radius in units of scale radius
    m_nfw_inf = 2 * t**4 * (t**2 - 3) * np.log(t) - t**2 * (3*t**2 - 1) * (t**2 - np.pi * t +1) / (2 * (t**2 + 1)**3)
    return 4 * np.pi * r_s ** 3 * rho0 * m_nfw_inf

def calculate_total_mass(r_vals, rho_bcm):
    """
    Calculate the total mass by integrating the density profile.
    
    This function numerically integrates 4πr²ρ(r) over the radius range to
    compute the total enclosed mass.
    
    Parameters
    ----------
    r_vals : array_like
        Radius values in Mpc/h
    rho_bcm : array_like
        Density profile in Msun/h/(Mpc/h)³
    
    Returns
    -------
    float
        Total mass in Msun/h
    
    Notes
    -----
    Uses trapezoidal integration for the numerical calculation.
    """
    integrand = 4 * np.pi * r_vals**2 * rho_bcm
    M_total = np.trapz(integrand, r_vals)  # Use trapezoidal integration
    return M_total

def y_bgas(r, r_s, r200, y0, c, rho0, r_tr):
    """
    Calculate the baryonic gas density profile.
    
    This function implements a modified profile for baryonic gas that consists
    of an inner profile (power law) and an outer NFW profile, with a smooth
    transition between them at r200/√5.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    r200 : float
        Virial radius in Mpc/h
    y0 : float
        Normalization parameter
    c : float
        Concentration parameter (r200/r_s)
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    
    Returns
    -------
    float or array_like
        Baryonic gas density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    The inner profile is given by Eq. 2.10: y0 * (ln(1+x)/x)^Γ_eff
    The outer profile follows the NFW form.
    The transition radius is r200/√5.
    """
    sqrt5 = np.sqrt(5)
    r_transition = r200 / sqrt5
    x = r / r_s

    Gamma_eff = (1 + 3*c/sqrt5) * np.log(1 + c/sqrt5) / ((1 + c/sqrt5)*np.log(1 + c/sqrt5) - c/sqrt5)
    
    inner_val = y0 * (np.log(1 + x) / x)**Gamma_eff
    
    outer_val = rho_nfw(r, r_s, y0, r_tr)
    
    x_trans = r_transition / r_s
    inner_at_trans = y0 * (np.log(1 + x_trans) / x_trans)**Gamma_eff
    outer_at_trans = rho_nfw(r_transition, r_s, y0, r_tr)
    scale_factor = outer_at_trans / inner_at_trans
    inner_scaled = inner_val * scale_factor
    outer_scaled = outer_val / scale_factor
    
    return np.where(r <= r_transition, inner_scaled, outer_val)

def y_egas(r, M_tot, r_ej):
    """
    Calculate the ejected gas density profile.
    
    This function implements the ejected gas profile as a 3D Gaussian 
    distribution as described in Eq. 2.13:
    y_egas(r) = M_tot / ((2π r_ej²)^(3/2)) * exp[-r²/(2r_ej²)]
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    M_tot : float
        Total mass of the ejected gas component in Msun/h
    r_ej : float
        Characteristic ejection radius in Mpc/h
    
    Returns
    -------
    float or array_like
        Ejected gas density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This component represents gas that has been ejected from the halo due to
    feedback processes such as AGN activity and supernovae.
    """
    norm = M_tot / ((2 * np.pi * r_ej**2)**(1.5))
    return norm * np.exp(- r**2 / (2 * r_ej**2))

def y_cgal(r, M_tot, R_h):
    """
    Calculate the central galaxy (stellar) density profile.
    
    This function implements the central galaxy profile based on observations,
    as described in Eq. 2.14:
    y_cgal(r) = M_tot / (4π^(3/2)R_h) * (1/r²) * exp[-(r/(2R_h))²]
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    M_tot : float
        Total stellar mass in Msun/h
    R_h : float
        Characteristic radius (Hernquist scale radius) in Mpc/h
    
    Returns
    -------
    float or array_like
        Stellar density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This profile is similar to a Hernquist profile but with a Gaussian cutoff
    to better match observed stellar distributions in galaxies.
    A small core radius is added to prevent division by zero at r=0.
    """
    r_safe = max(r, 1e-10)  # Prevent r=0 issues
    
    exp_term = np.exp(-(r_safe/(2*R_h))**2)
    
    norm = M_tot / (4 * np.pi**1.5 * R_h)
    return norm * (1.0 / r_safe**2) * exp_term

def y_rdm_ac(r, r_s, rho0, r_tr, norm=1.0,
             a=0.68,                # contraction strength
             f_cdm=0.839,           # CDM fraction of total mass
             baryon_components=None, # list of (r_array, y_vals) tuples
             verbose=False):
    """
    Calculate the adiabatically contracted dark matter profile with debugging.
    """
    def xi_of_r(rf):
        import Baryonic_Correction.utils as ut
        
        #if verbose:
        #    print(f"\n=== DEBUG xi_of_r for rf={rf:.6e} ===")
        
        # Calculate baryon mass at rf
        M_b_rf = 0.0
        if baryon_components:
            #if verbose:
            #    print(f"Processing {len(baryon_components)} baryon components:")
            
            for i, (r_array, rho_array) in enumerate(baryon_components):
                try:
                    # Debug the input arrays
                    #if verbose:
                    #    print(f"  Component {i}:")
                    #    print(f"    r_array range: {r_array.min():.3e} to #{r_array.max():.3e}")
                    #    print(f"    rho_array range: {rho_array.min():.3e} to {rho_array.max():.3e}")
                    #    print(f"    Valid rho values: {np.sum(np.isfinite#(rho_array) & (rho_array > 0))}/{len(rho_array)}")
                    
                    baryon_mass = ut.cumul_mass_single(rf, rho_array, r_array)
                    
                    #if verbose:
                    #    print(f"    Baryon mass at rf={rf:.3e}: {baryon_mass:.3e}")
                    
                    if not np.isfinite(baryon_mass) or baryon_mass < 0:
                        #if verbose:
                        #    print(f"    WARNING: Invalid baryon mass {baryon_mass}, setting to 0")
                        baryon_mass = 0.0
                    
                    M_b_rf += baryon_mass
                    
                except Exception as e:
                    #if verbose:
                    #    print(f"    ERROR calculating baryon mass for component {i}: {e}")
                    continue
        
        #if verbose:
        #    print(f"Total baryon mass M_b_rf: {M_b_rf:.6e}")
        
        def G(ri):
            #if verbose and ri in [1e-8, 1e8]:  # Only debug at bounds
            #    print(f"  G(ri={ri:.6e}):")
            
            try:
                # Calculate initial mass using mass_profile
                M_i_ri = mass_profile(ri, rho_nfw, r_s=r_s, rho0=rho0, r_tr=r_tr)
                
                #if verbose and ri in [1e-8, 1e8]:
                #    print(f"    M_i_ri: {M_i_ri:.6e}")
                
                # Check for invalid mass
                if not np.isfinite(M_i_ri) or M_i_ri <= 0:
                    #if verbose and ri in [1e-8, 1e8]:
                    #    print(f"    ERROR: Invalid M_i_ri = {M_i_ri}")
                    return np.nan
                
                # Calculate f_cdm - fraction of CDM in initial mass
                total_initial_mass = M_i_ri + M_b_rf
                if total_initial_mass <= 0:
                    #if verbose and ri in [1e-8, 1e8]:
                    #    print(f"    ERROR: Total initial mass <= 0: {total_initial_mass}")
                    return np.nan
                
                f_cdm_actual = M_i_ri / total_initial_mass
                
                # Calculate final mass
                M_f_rf = f_cdm_actual * M_i_ri + M_b_rf
                
                #if verbose and ri in [1e-8, 1e8]:
                    #print(f"    f_cdm_actual: {f_cdm_actual:.6f}")
                    #print(f"    M_f_rf: {M_f_rf:.6e}")
                
                # Check for invalid final mass
                if not np.isfinite(M_f_rf) or M_f_rf <= 0:
                    #if verbose and ri in [1e-8, 1e8]:
                    #    print(f"    ERROR: Invalid M_f_rf = {M_f_rf}")
                    return np.nan
                
                # Calculate the ratio
                ratio = M_i_ri / M_f_rf
                
                #if verbose and ri in [1e-8, 1e8]:
                    #print(f"    ratio M_i/M_f: {ratio:.6f}")
                
                # Calculate G
                result = rf/ri - 1.0 - a*(ratio - 1.0)
                
                #if verbose and ri in [1e-8, 1e8]:
                #    print(f"    rf/ri: {rf/ri:.6f}")
                #    print(f"    a*(ratio-1): {a*(ratio-1.0):.6f}")
                #    print(f"    G result: {result:.6e}")
                
                return result
                
            except Exception as e:
                #if verbose and ri in [1e-8, 1e8]:
                #    print(f"    EXCEPTION in G: {e}")
                return np.nan

        # Set bounds more conservatively
        ri_min = max(rf*1e-3, 1e-10)  # Less aggressive lower bound
        ri_max = min(rf*1e3, r_tr*2)   # Less aggressive upper bound
        
        #if verbose:
        #    print(f"Bounds: ri_min={ri_min:.6e}, ri_max={ri_max:.6e}")
        
        try:
            g_min = G(ri_min)
            g_max = G(ri_max)
            
            #if verbose:
            #    print(f"G(ri_min) = {g_min:.6e}")
            #    print(f"G(ri_max) = {g_max:.6e}")

            # Check for NaN values
            if np.isnan(g_min) or np.isnan(g_max):
                #if verbose:
                #    print(f"ERROR: G(ri) returned NaN at bounds. Returning xi=1.0")
                return 1.0

            # Check if root is bracketed
            if g_min * g_max >= 0:
                #if verbose:
                #    print(f"WARNING: Root not bracketed.")
                #    print(f"  This suggests either:")
                #    print(f"  - Very weak contraction (xi ≈ 1)")
                #    print(f"  - Mass calculation issues")
                #    print(f"  - Need wider bounds")
                
                # Try to find a reasonable solution
                if abs(g_min) < 1e-6:
                    xi = rf / ri_min
                    #if verbose:
                    #    print(f"  Using ri_min solution: xi = {xi:.6f}")
                    return xi
                elif abs(g_max) < 1e-6:
                    xi = rf / ri_max
                    #if verbose:
                    #    print(f"  Using ri_max solution: xi = {xi:.6f}")
                    return xi
                else:
                    #if verbose:
                    #    print(f"  No good solution found, returning xi=1.0")
                    return 1.0

            # Solve for ri
            ri = brentq(G, ri_min, ri_max, xtol=1e-8, rtol=1e-8, maxiter=100)
            xi = rf / ri
            
            #if verbose:
            #    print(f"Solution: ri = {ri:.6e}, xi = {xi:.6f}")

            # Validate result
            if not np.isfinite(xi) or xi <= 0:
                #if verbose:
                #    print(f"ERROR: Unphysical xi={xi:.6e}. Returning xi=1.0")
                return 1.0
            
            # Additional sanity check
            if xi > 10:  # Very strong contraction, probably unphysical
                #if verbose:
                #    print(f"WARNING: Very strong contraction xi={xi:.6f}, capping at 2.0")
                return 2.0

            return xi

        except Exception as e:
            if verbose:
                print(f"ERROR: Exception in xi_of_r: {e}")
                import traceback
                traceback.print_exc()
            return 1.0

    # Main function logic
    if np.isscalar(r):
        xi = xi_of_r(r)
        ri = r/xi
        return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)

    out = np.zeros_like(r)
    xi_vals = np.zeros_like(r)
    
    for i, rf in enumerate(r):
        xi = xi_of_r(rf)
        xi = min(xi, 10.0)  # Cap maximum contraction
        xi = max(xi, 0.1)   # Cap minimum contraction
        xi_vals[i] = xi
        ri = rf/xi
        out[i] = norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)
    
    if verbose:  
        print(f"\nFinal xi_vals: {xi_vals[:10]}")
        idx_almost_one = np.argmax(np.isclose(xi_vals, 1.0, atol=1e-3))
        print(f"First xi ~ 1 at index: {idx_almost_one} and r = {r[idx_almost_one]}")
        
        # Additional diagnostics
        print(f"Xi statistics:")
        print(f"  Min xi: {xi_vals.min():.6f}")
        print(f"  Max xi: {xi_vals.max():.6f}")
        print(f"  Mean xi: {xi_vals.mean():.6f}")
        print(f"  Xi = 1 count: {np.sum(np.isclose(xi_vals, 1.0, atol=1e-3))/ len(xi_vals)}")
    
    return out

def y_rdm_ac2(r, r_s, rho0, r_tr, M_i, M_f, verbose, norm=1.0,
              a=0.68,               # contraction strength
              f_cdm=0.839):         # CDM fraction of total mass
    """
    Calculate the adiabatically contracted dark matter profile using pre-computed mass profiles.
    
    This alternative implementation of adiabatic contraction uses pre-computed
    mass profiles M_i and M_f rather than calculating them on-the-fly.
    
    Parameters
    ----------
    r : array_like
        Radius array in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    M_i : array_like
        Initial mass profile (before contraction) in Msun/h
    M_f : array_like
        Final mass profile (after contraction) in Msun/h
    verbose : bool
        Whether to print diagnostic information
    norm : float, optional
        Normalization factor, default 1.0
    a : float, optional
        Contraction strength parameter, default 0.68
    f_cdm : float, optional
        CDM fraction of total mass, default 0.839
    
    Returns
    -------
    array_like
        Contracted dark matter density at the specified radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This method interpolates the mass profiles to determine the contraction
    parameter ξ at each radius. It then applies the same contraction formula as
    y_rdm_ac but avoids recomputing the mass profiles at each step.
    """
    def xi_of_r(rf):
        def G(ri):
            M_i_interp = np.interp(ri, r, M_i)
            M_f_interp = np.interp(rf, r, M_f)
            return rf/ri - 1.0 - a*(M_i_interp/M_f_interp - 1.0)

        ri_min, ri_max = rf*1e-3, rf * 100
        
        g_min, g_max = G(ri_min), G(ri_max)
        if g_min * g_max >= 0:
            print(f"WARNING: Root not bracketed for rf={rf:.3e}. Bounds=[{ri_min:.3e}, {ri_max:.3e}].")
            return 1.0
        
        ri = brentq(G, ri_min, ri_max, xtol=1e-6, disp=True)
        return rf/ri

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
                   a=0.68,          # contraction strength
                   f_cdm=0.839,     # CDM fraction of total mass
                   baryon_components=None): # list of (r_array, y_vals) tuples
    """
    Calculate the dark matter profile with a fixed contraction parameter.
    
    This simplified version of the adiabatic contraction model uses a fixed
    contraction parameter ξ for all radii instead of solving for it separately
    at each radius.
    
    Parameters
    ----------
    r : float or array_like
        Radius in Mpc/h
    r_s : float
        Scale radius in Mpc/h
    rho0 : float
        Characteristic density in Msun/h/(Mpc/h)^3
    r_tr : float
        Truncation radius in Mpc/h
    xi : float, optional
        Fixed contraction parameter, default 0.85
    norm : float, optional
        Normalization factor, default 1.0
    a : float, optional
        Contraction strength parameter (not used in this function), default 0.68
    f_cdm : float, optional
        CDM fraction of total mass (not used in this function), default 0.839
    baryon_components : list, optional
        Baryon components (not used in this function)
    
    Returns
    -------
    float or array_like
        Contracted dark matter density at the specified radius/radii in Msun/h/(Mpc/h)^3
    
    Notes
    -----
    This simplified model provides a way to apply contraction without the
    computational expense of solving for ξ at each radius. This method uses
    radius-dependent contraction for r < 0.04, transitioning to no contraction (ξ=1)
    for r ≥ 0.04.
    """
    limit = 0.04
    inner_xi = 0.65
    if r < limit:
        xi_interp = 1 - ((1 - inner_xi)) * ((1 - r / limit))
        xi = np.clip(xi_interp, inner_xi, 1.0)
    else:
        xi = 1
    ri = r / xi
    return norm * xi**(-3) * rho_nfw(ri, r_s, rho0, r_tr)

##### Newer Paper Functions #####

def rho_clm(r, f_sga, O_dm, O_m, *args, **kwargs):
    return ((O_dm / O_m) + f_sga) * rho_nfw(r, *args, **kwargs)

def rho_cga(r, R_h, M, f_cga):
    expo = -(r / (2 * R_h))**2
    return (1 / (4 * np.pi**1.5 * R_h * r**(-2))) * M  * f_cga *np.exp(expo)

def rho_gas(r, M, r_vir, f_b, f_star, the_ej = 4.235, delta=6.40, gamma=2.25):

    a = f_b - f_star
    b = (1 + 10 * (r / r_vir))**(beta(M)) 
    c = (1 + (r / (r_vir * the_ej))**(gamma))
    e = (delta - beta(M))/gamma
    return a/(b * c**e)

def beta(M, M_c = 10 ** 13.322, mu = 0.93):
    return 3*(M/M_c)**mu / (1 + (M/M_c)**mu)
    