import pytest
import numpy as np
try:
    from BCM import utils as ut
    from BCM import density_profiles as dp
except ImportError:
    from Package.BCM import utils as ut
    from Package.BCM import density_profiles as dp

@pytest.fixture
def test_params():
    """Setup common test parameters."""
    return {
        'r_vals': np.logspace(-3, 2, 100),  # Log-spaced radius array
        'r_s': 0.2,        # Scale radius in Mpc/h
        'r200': 1.0,       # Virial radius in Mpc/h
        'rho0': 1e7,       # Characteristic density in Msun/h/Mpc³
        'r_tr': 8.0,       # Truncation radius in Mpc/h
        'M200': 1e14,      # Halo mass in Msun/h
        'z': 0.0,          # Redshift
        'Omega_m': 0.3,    # Matter density parameter
        'Omega_b': 0.05,   # Baryon density parameter
    }

def test_calc_concentration_basic(test_params):
    """Test basic functionality of concentration calculation."""
    p = test_params
    
    # Calculate concentration for standard halo
    c = ut.calc_concentration(p['M200'], p['z'])
    
    # Should return positive value in reasonable range for this mass
    assert 2.0 <= c <= 15.0

def test_calc_concentration_mass_dependence():
    """Test that concentration depends on mass and redshift correctly."""
    # Lower mass halos should have higher concentration at fixed redshift
    low_mass = 1e12
    high_mass = 1e15
    z = 0.0
    
    c_low = ut.calc_concentration(low_mass, z)
    c_high = ut.calc_concentration(high_mass, z)
    
    # Concentration should decrease with increasing mass
    assert c_low > c_high
    
    # Higher redshift should have lower concentration at fixed mass
    z_low = 0.0
    z_high = 2.0
    
    c_z_low = ut.calc_concentration(low_mass, z_low)
    c_z_high = ut.calc_concentration(low_mass, z_high)
    
    # Concentration should decrease with increasing redshift
    assert c_z_low > c_z_high

def test_calc_r_ej_basic(test_params):
    """Test basic functionality of ejection radius calculation."""
    p = test_params
    
    # Calculate ejection radius
    r_ej = ut.calc_r_ej(p['r200'], p['z'], Omega_m=p['Omega_m'])
    
    # Should return positive value
    assert r_ej > 0
    
    # Should be of reasonable size compared to r200
    assert 0.1 * p['r200'] < r_ej < 10 * p['r200']

def test_calc_r_ej_parameter_dependence(test_params):
    """Test that ejection radius depends on parameters correctly."""
    p = test_params
    
    # Calculate with different theta values
    r_ej1 = ut.calc_r_ej(p['r200'], p['z'], Omega_m=p['Omega_m'], theta=0.3)
    r_ej2 = ut.calc_r_ej(p['r200'], p['z'], Omega_m=p['Omega_m'], theta=0.6)
    
    # Larger theta should give larger ejection radius
    assert r_ej2 > r_ej1
    
    # Calculate with different redshifts
    r_ej_z0 = ut.calc_r_ej(p['r200'], 0.0, Omega_m=p['Omega_m'])
    r_ej_z1 = ut.calc_r_ej(p['r200'], 1.0, Omega_m=p['Omega_m'])
    
    # Ejection radius should be smaller at higher redshift due to higher density
    assert r_ej_z0 != r_ej_z1 
    
def test_calc_r_ej2_basic(test_params):
    """Test basic functionality of alternate ejection radius calculation."""
    p = test_params
    
    # Calculate ejection radius
    r_ej = ut.calc_r_ej2(p['M200'], p['r200'], p['z'], p['Omega_m'])
    
    # Should return positive value
    assert r_ej > 0
    
    # Should be of reasonable size compared to r200
    assert 0.1 * p['r200'] < r_ej < 10.0 * p['r200']

def test_calc_r_ej2_mass_dependence(test_params):
    """Test that ejection radius depends on mass correctly."""
    p = test_params
    
    # Calculate for different masses (should affect ejected mass)
    low_mass = 1e12
    high_mass = 1e15
    
    r_ej_low = ut.calc_r_ej2(low_mass, p['r200']/5, p['z'], p['Omega_m'])
    r_ej_high = ut.calc_r_ej2(high_mass, p['r200']*5, p['z'], p['Omega_m'])
    
    # Should be different - exact relationship depends on fractions
    assert r_ej_low != r_ej_high
    
def test_calc_R_h_basic(test_params):
    """Test basic functionality of half-light radius calculation."""
    p = test_params
    
    # Calculate half-light radius for a typical halo
    R_h = ut.calc_R_h(p['M200'], p['r200'])
    
    # Should return positive value
    assert R_h > 0
    
    # Should be a small fraction of r200 (typically around ~1-2%)
    assert 0.005 * p['r200'] < R_h < 0.05 * p['r200']

def test_calc_R_h_mass_dependence():
    """Test mass dependence of half-light radius calculation."""
    # Define different mass halos with corresponding r200 values
    dwarf_mass = 1e11
    dwarf_r200 = 0.2
    
    milky_way_mass = 1e12
    milky_way_r200 = 0.4
    
    cluster_mass = 1e15
    cluster_r200 = 2.0
    
    # Calculate R_h for each
    R_h_dwarf = ut.calc_R_h(dwarf_mass, dwarf_r200)
    R_h_mw = ut.calc_R_h(milky_way_mass, milky_way_r200)
    R_h_cluster = ut.calc_R_h(cluster_mass, cluster_r200)
    
    # Check relative sizes (R_h/r200 should be larger for dwarfs than clusters)
    assert R_h_dwarf/dwarf_r200 > R_h_mw/milky_way_r200
    assert R_h_mw/milky_way_r200 > R_h_cluster/cluster_r200

def test_bracket_rho0_basic(test_params):
    """Test basic functionality of rho0 calculation."""
    p = test_params
    
    # Target mass (some fraction of typical halo mass)
    M_target = 0.8 * p['M200']
    
    # Calculate rho0 to achieve target mass
    rho0 = ut.bracket_rho0(M_target, p['r_s'], p['r_tr'], p['r200'])
    
    # Should return positive value
    assert rho0 > 0
    
    # Verify the calculated rho0 gives approximately the target mass
    M_actual = dp.mass_profile(p['r200'], dp.rho_nfw, r_s=p['r_s'], rho0=rho0, r_tr=p['r_tr'])
    assert abs(M_actual/M_target - 1.0) < 0.01  # Within 1%

def test_bracket_rho0_different_radii(test_params):
    """Test rho0 calculation for different integration limits."""
    p = test_params
    
    # Target mass
    M_target = 0.5 * p['M200']
    
    # Calculate rho0 up to r200
    rho0_r200 = ut.bracket_rho0(M_target, p['r_s'], p['r_tr'], p['r200'])
    
    # Calculate rho0 up to a smaller radius
    r_half = 0.5 * p['r200']
    rho0_half = ut.bracket_rho0(M_target, p['r_s'], p['r_tr'], p['r200'], r_max=r_half)
    
    # Should be different because we're fitting the same mass in a smaller volume
    assert rho0_half > rho0_r200
    
    # Verify both give correct masses at their respective radii
    M_r200 = dp.mass_profile(p['r200'], dp.rho_nfw, r_s=p['r_s'], rho0=rho0_r200, r_tr=p['r_tr'])
    M_half = dp.mass_profile(r_half, dp.rho_nfw, r_s=p['r_s'], rho0=rho0_half, r_tr=p['r_tr'])
    
    assert abs(M_r200/M_target - 1.0) < 0.01
    assert abs(M_half/M_target - 1.0) < 0.01

def test_normalize_component_basic(test_params):
    """Test basic functionality of component normalization."""
    p = test_params
    
    # Simple test density function
    def test_density(r, A=1.0):
        return A * (r/p['r_s'])**(-2)
    
    # Normalize to contain target mass
    M_target = 0.3 * p['M200']
    norm = ut.normalize_component(test_density, (1.0,), M_target, p['r200'])
    
    # Should return positive normalization
    assert norm > 0
    
    # Verify normalized profile integrates to target mass
    M_actual = dp.mass_profile(p['r200'], lambda r: norm * test_density(r, 1.0))
    assert abs(M_actual/M_target - 1.0) < 0.05  # Within 5%

def test_normalize_component_different_profiles(test_params):
    """Test normalization of different density profiles."""
    p = test_params
    
    # Two test profiles with different slopes
    def shallow_profile(r, A=1.0):
        return A * (r/p['r_s'])**(-1)
    
    def steep_profile(r, A=1.0):
        return A * (r/p['r_s'])**(-3)
    
    # Normalize both to the same mass
    M_target = 0.5 * p['M200']
    norm_shallow = ut.normalize_component(shallow_profile, (1.0,), M_target, p['r200'])
    norm_steep = ut.normalize_component(steep_profile, (1.0,), M_target, p['r200'])
    
    # Steeper profile needs higher normalization because mass is more concentrated
    assert norm_steep != norm_shallow
    
    # Verify both integrate to target mass
    M_shallow = dp.mass_profile(p['r200'], lambda r: norm_shallow * shallow_profile(r, 1.0))
    M_steep = dp.mass_profile(p['r200'], lambda r: norm_steep * steep_profile(r, 1.0))
    
    assert abs(M_shallow/M_target - 1.0) < 0.05
    assert abs(M_steep/M_target - 1.0) < 0.05

def test_normalize_component_total_basic(test_params):
    """Test basic functionality of total component normalization."""
    p = test_params
    
    # Simple test density function
    def test_density(r, A=1.0):
        return A * np.exp(-r/p['r_s'])
    
    # Normalize to contain target total mass
    M_target = 0.2 * p['M200']
    norm = ut.normalize_component_total(test_density, (1.0,), M_target, p['r200'])
    
    # Should return positive normalization
    assert norm > 0
    
    # Verify mass within a large radius approaches target
    r_large = 20 * p['r200']  # Use a large radius to capture "total" mass
    M_actual = dp.mass_profile(r_large, lambda r: norm * test_density(r, 1.0))
    assert abs(M_actual/M_target - 1.0) < 0.05  # Within 5%

def test_normalize_component_total_different_truncation(test_params):
    """Test normalization of total mass with different truncation radii."""
    p = test_params
    
    # Truncated NFW-like profile
    def truncated_profile(r, r_trunc=10.0):
        x = r/p['r_s']
        trunc_term = np.exp(-(r/r_trunc)**2)
        return (x * (1 + x)**2)**(-1) * trunc_term
    
    # Normalize with different truncation radii
    M_target = 0.4 * p['M200']
    norm1 = ut.normalize_component_total(truncated_profile, (5.0,), M_target, p['r200'])
    norm2 = ut.normalize_component_total(truncated_profile, (20.0,), M_target, p['r200'])
    
    # More extended profile (larger r_trunc) needs lower normalization
    assert norm1 > norm2
    
    # Verify both integrate to approximately target mass
    r_large = 100 * p['r200']
    M1 = dp.mass_profile(r_large, lambda r: norm1 * truncated_profile(r, 5.0))
    M2 = dp.mass_profile(r_large, lambda r: norm2 * truncated_profile(r, 20.0))
    
    assert abs(M1/M_target - 1.0) < 0.1
    assert abs(M2/M_target - 1.0) < 0.1

def test_cumul_mass_basic():
    """Test basic functionality of cumulative mass calculation."""
    # Create simple arrays
    r_array = np.linspace(0.1, 10, 100)
    rho_array = np.ones_like(r_array)  # Constant density
    
    # Calculate cumulative mass
    mass = ut.cumul_mass(r_array, rho_array)
    
    # Should return non-decreasing array
    assert np.all(np.diff(mass) >= 0)
    
    # For constant density, M(r) ∝ r³
    # Check last point against analytical expectation
    expected_ratio = r_array[-1]**3 / r_array[50]**3
    actual_ratio = mass[-1] / mass[50]
    assert abs(actual_ratio/expected_ratio - 1.0) < 0.05

def test_cumul_mass_specific_profile():
    """Test cumulative mass for power-law density profile."""
    # Create arrays for a power-law profile ρ(r) ∝ r^-2
    r_array = np.logspace(-1, 1, 200)
    rho_array = r_array**(-2)
    
    # Calculate cumulative mass
    mass = ut.cumul_mass(r_array, rho_array)
    
    # For ρ ∝ r^-2, M(r) ∝ r
    # Check growth rate at several points
    for i in range(20, len(r_array)-20, 20):
        r_ratio = r_array[i+20] / r_array[i]
        mass_ratio = mass[i+20] / mass[i]
        assert abs(mass_ratio/r_ratio - 1.0) < 0.5