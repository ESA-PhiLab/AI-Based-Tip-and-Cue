import math

def fov_angle_from_swath(swath_m, altitude_m):

    theta_rad = 2 * math.atan(swath_m / (2 * altitude_m))
    theta_deg = math.degrees(theta_rad)
    return theta_deg

def compute_orbital_period(a_m):
    """Computes orbital period in seconds from semi-major axis in meters"""
    mu_earth = 3.986004418e14  # m^3/s^2
    T = 2 * math.pi * math.sqrt(a_m**3 / mu_earth)
    return T

