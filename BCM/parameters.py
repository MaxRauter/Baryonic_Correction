# parameters.py
# Default cosmological and halo parameters for BCM verification
# Schneider & Teyssier (2015/2016) 

DEFAULTS = {
    'M200': 1e14,          # Halo mass in Msun/h
    'r200': 0.77,          # Halo radius in Mpc/h
    'c': 3.2,              # Concentration parameter
    'h': 0.6777,           # Dimensionless Hubble parameter
    'z': 0.0,              # Redshift
    'Omega_m': 0.3071,     # Matter density parameter
    'Omega_b': 0.0483,     # Baryon density parameter
    'r_ej_factor': 3.5,    # r_ej = factor * r200
    'R_h_factor': 0.015,   # R_h = factor * r200
    'Xi_contraction': 0.85 # Contraction factor
}

# Test cases for abundance fractions
CASE_PARAMS = {
    'a': {
        'f_rdm': 0.839,
        'f_bgas': 0.151,
        'f_cgal': 0.005,
        'f_egas': 0.005
    },
    'b': {
        'f_rdm': 0.839,
        'f_bgas': 0.078,
        'f_cgal': 0.005,
        'f_egas': 0.078
    },
    'c': {
        'f_rdm': 0.839,
        'f_bgas': 0.005,
        'f_cgal': 0.005,
        'f_egas': 0.151
    }
}