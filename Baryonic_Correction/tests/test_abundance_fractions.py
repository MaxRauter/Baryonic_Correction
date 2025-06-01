import pytest
import numpy as np
from Baryonic_Correction import abundance_fractions as af

@pytest.fixture
def test_params():
    return {
        'M': 1e14,  # Halo mass in Msun/h
        'z': 0.0,   # Redshift
        'Omega_m': 0.3,
        'Omega_b': 0.05,
        'fbar_loc': 0.05/0.3  # Omega_b/Omega_m
    }

# Define a basic cosmic baryon fraction function for testing
def cosmic_baryon_fraction(Omega_m, Omega_b):
    """Cosmic baryon fraction = Omega_b/Omega_m"""
    return Omega_b / Omega_m

def test_f_bgas_basic(test_params):
    """Basic test for bound gas fraction."""
    p = test_params
    f_bgas = af.f_bgas(p['M'], p['fbar_loc'])
    
    # Check if value is in physical range
    assert 0.0 <= f_bgas <= p['fbar_loc']
    

def test_f_bgas_mass_dependence(test_params):
    """Test mass dependence of bound gas fraction."""
    p = test_params
    
    # Higher mass halos should retain more gas
    low_mass = 1e12
    high_mass = 1e15
    
    f_bgas_low = af.f_bgas(low_mass, p['fbar_loc'])
    f_bgas_high = af.f_bgas(high_mass, p['fbar_loc'])
    
    # Higher mass halos should have more bound gas
    assert f_bgas_high > f_bgas_low

def test_g_func_basic():
    """Test basic functionality of g(x) function."""
    # g(x) should be a monotonically decreasing function near x=1
    x_values = np.array([0.8, 1.0, 1.2])
    g_values = [af.g_func(x) for x in x_values]
    
    # Check if decreasing
    assert g_values[0] > g_values[1] > g_values[2]

def test_g_func_parameter_dependence():
    """Test parameter dependence of g(x) function."""
    # Default parameters
    g_default = af.g_func(1.0)
    
    # Custom parameters
    alpha = -2.0  # More negative
    delta = 5.0   # Larger
    gamma = 0.6   # Larger
    
    g_custom = af.g_func(1.0, alpha=alpha, delta=delta, gamma=gamma)
    
    # Parameters should affect output
    assert g_custom != g_default

def test_f_cgal_basic(test_params):
    """Basic test for central galaxy fraction."""
    p = test_params
    f_cgal = af.f_cgal(p['M'])
    
    # Check if value is in physical range
    assert 0.0 <= f_cgal <= p['fbar_loc']
    
    # For cluster-mass halos like 1e14, stellar fraction should be low
    assert f_cgal < 0.25 * p['fbar_loc']

def test_f_cgal_mass_dependence():
    """Test mass dependence of central galaxy fraction."""
    # Test different halo masses
    low_mass = 1e12
    high_mass = 1e15
    
    f_cgal_low = af.f_cgal(low_mass)
    f_cgal_high = af.f_cgal(high_mass)
    
    # Star formation efficiency peaks at Milky Way masses
    # Higher values for medium mass than very high mass
    assert f_cgal_low > f_cgal_high

def test_f_egas_basic(test_params):
    """Basic test for ejected gas fraction."""
    p = test_params
    
    # Calculate ingredients
    f_bgas_val = af.f_bgas(p['M'], p['fbar_loc'])
    f_cgal_val = af.f_cgal(p['M'])
    
    # Calculate ejected gas
    f_egas = af.f_egas(f_bgas_val, f_cgal_val, p['fbar_loc'])
    
    # Check if value is in physical range
    assert 0.0 <= f_egas <= p['fbar_loc']
    
    # Conservation of mass: f_bgas + f_cgal + f_egas = fbar_loc
    assert abs((f_bgas_val + f_cgal_val + f_egas) - p['fbar_loc']) < 1e-10

def test_f_egas_mass_dependence(test_params):
    """Test mass dependence of ejected gas fraction."""
    p = test_params
    
    # Define masses
    low_mass = 1e12
    high_mass = 1e15
    
    # Calculate for low mass
    f_bgas_low = af.f_bgas(low_mass, p['fbar_loc'])
    f_cgal_low = af.f_cgal(low_mass)
    f_egas_low = af.f_egas(f_bgas_low, f_cgal_low, p['fbar_loc'])
    
    # Calculate for high mass
    f_bgas_high = af.f_bgas(high_mass, p['fbar_loc'])
    f_cgal_high = af.f_cgal(high_mass)
    f_egas_high = af.f_egas(f_bgas_high, f_cgal_high, p['fbar_loc'])
    
    # Lower mass halos should have more ejected gas (stronger feedback)
    assert f_egas_low > f_egas_high

def test_f_rdm_basic(test_params):
    """Basic test for dark matter fraction."""
    p = test_params
    f_rdm = af.f_rdm(p['fbar_loc'])
    
    # Check if value is in physical range
    assert 0.7 < f_rdm < 1.0
    
    # DM fraction should be approximately 1 - baryon fraction
    assert abs(f_rdm - (1.0 - p['fbar_loc'])) < 1e-10

def test_f_rdm_cosmic_dependence():
    """Test dark matter fraction with different cosmic baryon fractions."""
    # Create two different baryon fractions
    high_fb = 0.2  # Higher baryon fraction
    low_fb = 0.1   # Lower baryon fraction
    
    f_rdm_high = af.f_rdm(high_fb)
    f_rdm_low = af.f_rdm(low_fb)
    
    # Higher baryon fraction should result in lower DM fraction
    assert f_rdm_high < f_rdm_low
    
    # Should exactly equal 1-fb
    assert abs(f_rdm_high - (1.0 - high_fb)) < 1e-10
    assert abs(f_rdm_low - (1.0 - low_fb)) < 1e-10

def test_component_sum_to_one(test_params):
    """Test that all components sum to 1."""
    p = test_params
    
    # Calculate all mass fractions
    f_bgas_val = af.f_bgas(p['M'], p['fbar_loc'])
    f_cgal_val = af.f_cgal(p['M'])
    f_egas_val = af.f_egas(f_bgas_val, f_cgal_val, p['fbar_loc'])
    f_rdm_val = af.f_rdm(p['fbar_loc'])
    
    # Sum of all fractions should be very close to 1
    total = f_rdm_val + f_bgas_val + f_egas_val + f_cgal_val
    assert abs(total - 1.0) < 1e-10
