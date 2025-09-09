import math
import numpy as np

def fov_angle_from_swath(swath_m, altitude_m):

    theta_rad = 2 * math.atan(swath_m / (2 * altitude_m))
    theta_deg = math.degrees(theta_rad)
    return theta_deg

def compute_orbital_period(a_m):
    """Computes orbital period in seconds from semi-major axis in meters"""
    mu_earth = 3.986004418e14  # m^3/s^2
    T = 2 * math.pi * math.sqrt(a_m**3 / mu_earth)
    return T

def estimate_box_inertia(m_kg, Lx_m, Ly_m, Lz_m):
    """
    Quick inertia estimate for a rectangular box about its principal axes, through COM.
    Jx = (1/12) m (Ly^2 + Lz^2), etc.
    Returns np.array([Jx, Jy, Jz]) in kg·m^2.
    """
    Jx = (1.0/12.0) * m_kg * (Ly_m**2 + Lz_m**2)
    Jy = (1.0/12.0) * m_kg * (Lx_m**2 + Lz_m**2)
    Jz = (1.0/12.0) * m_kg * (Lx_m**2 + Ly_m**2)
    return np.array([Jx, Jy, Jz], dtype=float)
