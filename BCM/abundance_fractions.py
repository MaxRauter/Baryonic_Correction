import numpy as np

# --------------------------------------------------------------------
# Abundance Fractions
# --------------------------------------------------------------------

def f_bgas(M, fbar_loc, Mc=1.2e14, beta=0.6):
    """
    Bound gas fraction as a function of halo mass M (Eq. 2.19).

    f_bgas(M) = (Omega_b/Omega_m) / (1 + (Mc/M)^beta)
    """
    return fbar_loc / (1 + (Mc/M)**beta)

def g_func(x, alpha=-1.779, delta=4.394, gamma=0.547):
    """
    Helper function for the central galaxy abundance (Eq. 2.20).

    g(x) = - log10(10^alpha * x + 1) + delta * (log10(1 + exp(x)))^gamma / (1+exp(10*x))
    """
    return - np.log10(10**(alpha) * x + 1) + delta * (np.log10(1 + np.exp(x)))**gamma / (1 + np.exp(10*x))

def f_cgal(M, epsilon=0.023, M1=1.526e11):
    """
    Central galaxy stellar fraction (Eq. 2.20).

    fcgal(M) = epsilon * (M1/M) * 10^(g(log10(M/M1)) - g(0))
    """
    x = np.log10(M/M1)
    return epsilon * (M1/M) * 10**( g_func(x) - g_func(0) )

def f_egas(fbgas, fcgal, fbar_loc):
    """
    Ejected gas fraction (Eq. 2.21).

    f_egas(M) = fbar - f_bgas(M) - f_cgal(M)
    """
    return fbar_loc - fbgas - fcgal

def f_rdm(fbar_loc):
    """
    Relaxed dark matter fraction is given by:

    f_rdm = 1 - fbar
    """
    return 1 - fbar_loc
