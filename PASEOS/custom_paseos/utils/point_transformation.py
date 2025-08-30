import numpy as np
import pykep as pk
import pymap3d as pm

def Point_Geodetic2ECI(lat,lon,alt,time):
    x, y, z = pm.geodetic2ecef(lat, lon, alt)
    #u1 = np.array([[x], [y], [z]]),
    #time = pm.datetime2jd(time)
    T2 = pm.ecef2eci(x, y, z, time)
    x, y, z = T2
    P =  np.array([[x], [y], [z]])
    return P

def Point_ECI2Geodetic(x_ECI, y_ECI, z_ECI,time):
    x_ECEF, y_ECEF, z_ECEF = pm.eci2ecef(x_ECI, y_ECI, z_ECI, time)
    lat, long, h = pm.ecef2geodetic(x_ECEF, y_ECEF, z_ECEF)
    P = np.array([[lat], [long], [h]])
    return P

def Point_ECEF2Geodetic(x_ECEF, y_ECEF, z_ECEF):
    lat, long, h = pm.ecef2geodetic(x_ECEF, y_ECEF, z_ECEF)
    P = np.array([[lat], [long], [h]])
    return P

def Point_Geodetic2ECEF(lat, lon, alt):

    x, y, z = pm.geodetic2ecef(lat, lon, alt)
    P = np.array([[x], [y], [z]])
    return P