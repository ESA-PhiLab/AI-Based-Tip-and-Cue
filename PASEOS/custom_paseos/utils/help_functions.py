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

import numpy as np

import numpy as np

def pass_time_from_nadir(alt_m, el_min_deg=10.0, R_earth_m=6378137.0, mu=3.986004418e14):
    """
    Compute the time [s] from satellite nadir (90° elevation)
    to when the target drops below a given minimum elevation angle.

    Parameters
    ----------
    alt_m : float
        Satellite altitude above Earth [m]
    el_min_deg : float
        Minimum elevation angle [deg] (default 10°)
    R_earth_m : float
        Earth radius [m] (default WGS84 mean = 6378137.0 m)
    mu : float
        Earth's gravitational parameter [m^3/s^2] (default 3.986004418e14)

    Returns
    -------
    t_half : float
        Time [s] from nadir to cutoff elevation
    t_full : float
        Approximate full visible duration [s] above elevation cutoff
    """

    r = R_earth_m + alt_m
    el_min = np.deg2rad(el_min_deg)

    # Orbital angular velocity [rad/s]
    omega = np.sqrt(mu / r**3)

    # Central angle to cutoff point
    cos_theta = (R_earth_m / r) * np.cos(el_min)
    theta_vis = np.arccos(cos_theta)  # rad

    # Time from nadir to cutoff
    t_half = theta_vis / omega

    # Full pass duration
    t_full = 2 * t_half

    return t_half, t_full


