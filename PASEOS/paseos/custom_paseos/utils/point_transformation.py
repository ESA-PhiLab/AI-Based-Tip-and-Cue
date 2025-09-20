"""
Point transformations between Geodetic, ECEF, and ECI frames.
"""

import numpy as np
import pymap3d as pm


def Point_Geodetic2ECI(lat, lon, alt, time):
    """Convert (lat,lon,alt) → ECI position vector at datetime `time`."""
    x, y, z = pm.geodetic2ecef(lat, lon, alt)
    rx, ry, rz = pm.ecef2eci(x, y, z, time)
    return np.array([[rx], [ry], [rz]])


def Point_ECI2Geodetic(x, y, z, time):
    """Convert ECI position → (lat,lon,alt)."""
    x_ecef, y_ecef, z_ecef = pm.eci2ecef(x, y, z, time)
    lat, lon, h = pm.ecef2geodetic(x_ecef, y_ecef, z_ecef)
    return np.array([[lat], [lon], [h]])


def Point_ECEF2Geodetic(x, y, z):
    """Convert ECEF position → (lat,lon,alt)."""
    lat, lon, h = pm.ecef2geodetic(x, y, z)
    return np.array([[lat], [lon], [h]])


def Point_Geodetic2ECEF(lat, lon, alt):
    """Convert (lat,lon,alt) → ECEF position vector."""
    x, y, z = pm.geodetic2ecef(lat, lon, alt)
    return np.array([[x], [y], [z]])
