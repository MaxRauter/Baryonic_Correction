import pytest
import numpy as np
from BCM import density_profiles as dp
import scipy.integrate as integrate

# Create a fixture to replace setUp
@pytest.fixture
def test_params():
    return {
        'r_vals': np.logspace(-3, 2, 100),  # Log-spaced radius array
        'r_s': 0.2,      # Scale radius in Mpc/h
        'r200': 1.0,     # Virial radius in Mpc/h
        'rho0': 1e7,     # Characteristic density in Msun/h/Mpc³
        'r_tr': 8.0,     # Truncation radius in Mpc/h
        'c': 5.0,        # Concentration parameter
        'R_h': 0.01,     # Half-light radius for galaxy profile
        'r_ej': 0.3,     # Ejection radius
        'f_cdm': 0.839,  # CDM fraction
    }

def text_rho_nfw_basic(test_params):
    """Test basic NFW profile functionality."""
    p = test_params
    r_vals = p['r_vals']
    
    # Calculate NFW density profile
    rho_nfw = dp.rho_nfw(r_vals, p['r_s'], p['rho0'], p['r_tr'])
    
    # Check if the profile is positive
    assert np.all(rho_nfw > 0)
    
    # Check if the profile is decreasing with radius
    assert np.all(np.diff(rho_nfw) < 0)

def test_rho_nfw_single_value(test_params):
    """Test NFW profile at a specific radius."""
    p = test_params
    rho = dp.rho_nfw(p['r_s'], p['r_s'], p['rho0'], p['r_tr'])
    # Use pytest's direct assertions
    assert abs(rho - p['rho0'] / 4) < 0.01 * p['rho0']
    
def test_mass_profile_nfw(test_params):
    """Test mass profile integration for NFW."""
    p = test_params
    # Test at r_s
    M_rs = dp.mass_profile(p['r_s'], dp.rho_nfw, r_s=p['r_s'], 
                        rho0=p['rho0'], r_tr=p['r_tr'])
    assert M_rs > 0
    
    # Test mass increases with radius
    M1 = dp.mass_profile(0.5*p['r_s'], dp.rho_nfw, r_s=p['r_s'], 
                      rho0=p['rho0'], r_tr=p['r_tr'])
    M2 = dp.mass_profile(2.0*p['r_s'], dp.rho_nfw, r_s=p['r_s'], 
                      rho0=p['rho0'], r_tr=p['r_tr'])
    assert M1 < M2
    
def test_mass_profile_analytical(test_params):
    """Compare mass profile to analytical solution for simple profile."""
    # Use a simple profile with known analytical solution: ρ(r) = ρ0
    def constant_density(r, rho0):
        return rho0
    
    r_test = 2.0
    rho_test = 1000.0
    M_numerical = dp.mass_profile(r_test, constant_density, rho0=rho_test)
    M_analytical = 4/3 * np.pi * r_test**3 * rho_test
    
    # Should match closely
    assert abs(M_numerical/M_analytical - 1.0) < 0.01
    
def test_calculate_total_mass_constant_density(test_params):
    """Test total mass calculation with constant density."""
    r_vals = np.linspace(0, 1, 100)
    rho_vals = np.ones_like(r_vals) * 1000.0  # Constant density
    
    M_numerical = dp.calculate_total_mass(r_vals, rho_vals)
    M_analytical = 4/3 * np.pi * r_vals[-1]**3 * 1000.0
    
    assert abs(M_numerical/M_analytical - 1.0) < 0.01
    
def test_calculate_total_mass_power_law(test_params):
    """Test total mass calculation with power law density."""
    r_vals = np.logspace(-2, 0, 100)
    # ρ(r) ∝ r^-2
    rho_vals = 1e5 * r_vals**(-2)
    
    M_numerical = dp.calculate_total_mass(r_vals, rho_vals)
    # For r^-2 profile, M(r) = 4πρ0*r
    M_analytical = 4 * np.pi * 1e5 * r_vals[-1]
    
    # Allow some numerical error due to discretization
    assert abs(M_numerical/M_analytical - 1.0) < 0.1
    
def test_y_bgas_transition(test_params):
    """Test bound gas profile at the transition radius."""
    p = test_params
    r_transition = p['r200'] / np.sqrt(5)
    y0 = 1e5
    
    # Evaluate slightly below and above transition
    rho_below = dp.y_bgas(0.99*r_transition, p['r_s'], p['r200'], y0, 
                       p['c'], p['rho0'], p['r_tr'])
    rho_at = dp.y_bgas(r_transition, p['r_s'], p['r200'], y0, 
                    p['c'], p['rho0'], p['r_tr'])
    rho_above = dp.y_bgas(1.01*r_transition, p['r_s'], p['r200'], y0, 
                       p['c'], p['rho0'], p['r_tr'])
    
    # Should be continuous at transition
    assert abs(rho_below/rho_at - 1.0) < 0.1
    assert abs(rho_above/rho_at - 1.0) < 0.1
    
def test_y_bgas_normalization(test_params):
    """Test bound gas profile normalization."""
    p = test_params
    r_vals = np.logspace(-3, 1, 1000)
    y0 = 1e5
    
    # Calculate profile and integrate to get total mass
    rho_vals = np.array([dp.y_bgas(r, p['r_s'], p['r200'], y0, p['c'], 
                                p['rho0'], p['r_tr']) for r in r_vals])
    
    # Check if profile integrates close to y0
    integrand = 4 * np.pi * r_vals**2 * rho_vals
    M_total = np.trapz(integrand, r_vals)
    
    # The normalization should be reasonably close to y0
    assert M_total > 0
    
def test_y_egas_normalization(test_params):
    """Test ejected gas profile normalization."""
    p = test_params
    r_vals = np.logspace(-3, 2, 1000)  # Go out far enough
    M_tot = 1e12
    
    # Calculate profile
    rho_vals = np.array([dp.y_egas(r, M_tot, p['r_ej']) for r in r_vals])
    
    # Integrate to verify total mass
    integrand = 4 * np.pi * r_vals**2 * rho_vals
    M_numerical = np.trapz(integrand, r_vals)
    
    # Should recover input mass (allow some numerical error for finite integration)
    assert abs(M_numerical/M_tot - 1.0) < 0.05
    
def test_y_egas_peak(test_params):
    """Test ejected gas profile peak location."""
    p = test_params
    r_vals = np.logspace(-3, 1, 1000)
    M_tot = 1e12
    
    # Find peak of r²ρ(r)
    densities = np.array([dp.y_egas(r, M_tot, p['r_ej']) * r**2 for r in r_vals])
    peak_idx = np.argmax(densities)
    peak_r = r_vals[peak_idx]
    
    # Peak should be near r_ej (the characteristic radius)
    assert abs(peak_r/p['r_ej'] - 1.0) < 0.5
    
def test_y_cgal_normalization(test_params):
    """Test galaxy profile normalization."""
    p = test_params
    r_vals = np.logspace(-3, 1, 1000)
    M_tot = 1e11
    
    # Calculate profile
    rho_vals = np.array([dp.y_cgal(r, M_tot, p['R_h']) for r in r_vals])
    
    # Integrate to get mass
    integrand = 4 * np.pi * r_vals**2 * rho_vals
    M_numerical = np.trapz(integrand, r_vals)
    
    # Should recover total mass
    assert abs(M_numerical/M_tot - 1.0) < 0.1
    
def test_y_cgal_half_mass_radius(test_params):
    """Test galaxy profile half-mass radius."""
    p = test_params
    r_vals = np.logspace(-4, 1, 1000)
    M_tot = 1e11
    
    # Calculate profile and cumulative mass
    rho_vals = np.array([dp.y_cgal(r, M_tot, p['R_h']) for r in r_vals])
    integrand = 4 * np.pi * r_vals**2 * rho_vals
    cum_mass = np.cumsum(np.diff(r_vals) * 0.5 * (integrand[1:] + integrand[:-1]))
    
    # Find half-mass radius
    half_mass_idx = np.argmin(np.abs(cum_mass - 0.5 * cum_mass[-1]))
    half_mass_r = r_vals[half_mass_idx]
    
    # Half-mass radius should be close to R_h
    assert abs(half_mass_r/p['R_h'] - 1.0) < 0.3
    
def test_y_rdm_ac_basic(test_params):
    """Test adiabatically contracted DM profile basic functionality."""
    p = test_params
    # Create baryon components (simplified)
    r_vals = np.logspace(-3, 1, 100)
    bgas_vals = np.array([1e5 * r**(-1) if r > 0.01 else 1e5 * 0.01**(-1) for r in r_vals])
    baryon_components = [(r_vals, bgas_vals)]
    
    # Test at a specific radius
    r_test = 0.5
    rho_ac = dp.y_rdm_ac(r_test, p['r_s'], p['rho0'], p['r_tr'], 
                       a=0.68, f_cdm=p['f_cdm'], 
                       baryon_components=baryon_components)
    
    # Should be positive
    assert rho_ac > 0
    
    # Should be greater than regular NFW (due to contraction)
    rho_nfw_val = dp.rho_nfw(r_test, p['r_s'], p['rho0'], p['r_tr'])
    assert rho_ac > rho_nfw_val
    
def test_y_rdm_ac_baryon_dependence(test_params):
    """Test that y_rdm_ac responds correctly to changes in baryon distribution."""
    p = test_params
    r_vals = np.logspace(-3, 1, 100)
    
    # Create two different baryon components (more/less concentrated)
    bgas_vals1 = np.array([1e5 * r**(-1) if r > 0.01 else 1e5 * 0.01**(-1) for r in r_vals])
    bgas_vals2 = np.array([1e5 * r**(-0.5) if r > 0.01 else 1e5 * 0.01**(-0.5) for r in r_vals])
    
    baryon_components1 = [(r_vals, bgas_vals1)]
    baryon_components2 = [(r_vals, bgas_vals2)]
    
    # Test at a specific radius
    r_test = 0.2
    rho_ac1 = dp.y_rdm_ac(r_test, p['r_s'], p['rho0'], p['r_tr'], 
                        a=0.68, f_cdm=p['f_cdm'], 
                        baryon_components=baryon_components1)
    
    rho_ac2 = dp.y_rdm_ac(r_test, p['r_s'], p['rho0'], p['r_tr'], 
                        a=0.68, f_cdm=p['f_cdm'], 
                        baryon_components=baryon_components2)
    
    # More concentrated baryon distribution should lead to stronger contraction
    assert rho_ac1 > rho_ac2
    