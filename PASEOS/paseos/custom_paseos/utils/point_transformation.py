"""
Point transformations between Geodetic, ECEF (ITRF), and ECI (EME2000) frames.

Inputs:
- lat/lon in DEGREES
- alt in METERS
- time is a Python datetime (treated as UTC if naive)

Frames:
- ECEF: ITRF (Earth-fixed)
- ECI:  EME2000 (inertial)

Public functions (kept identical names):
- Point_Geodetic2ECI
- Point_ECI2Geodetic
- Point_ECEF2Geodetic
- Point_Geodetic2ECEF
"""

from __future__ import annotations

import numpy as np
from math import radians, degrees

import orekit
from orekit.pyhelpers import setup_orekit_curdir
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

from org.orekit.bodies import GeodeticPoint
from org.orekit.frames import FramesFactory
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import IERSConventions

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates


_OREKIT_CACHE = {}


def build_orekit_frames() -> dict:
    """build_orekit_frames() -> dict: Build and cache Orekit frames + WGS84 ellipsoid."""
    global _OREKIT_CACHE
    if _OREKIT_CACHE:
        return _OREKIT_CACHE

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    eme2000 = FramesFactory.getEME2000()
    wgs84 = ReferenceEllipsoid.getWgs84(itrf)
    utc = TimeScalesFactory.getUTC()

    # Stable "dummy" date for ECEF-only conversion where caller provides no time.
    j2000 = AbsoluteDate(2000, 1, 1, 12, 0, 0.0, utc)

    _OREKIT_CACHE = {"itrf": itrf, "eme2000": eme2000, "wgs84": wgs84, "utc": utc, "j2000": j2000}
    return _OREKIT_CACHE


def _to_absolutedate(time) -> AbsoluteDate:
    """_to_absolutedate(time) -> AbsoluteDate: Python datetime -> Orekit AbsoluteDate (UTC)."""
    frames = build_orekit_frames()
    utc = frames["utc"]
    dt = time
    return AbsoluteDate(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        dt.second + dt.microsecond / 1e6,
        utc
    )


def _transform_position(tf, pos_vec3d: Vector3D) -> Vector3D:
    """_transform_position(tf,pos_vec3d) -> Vector3D: Robust position transform via PVCoordinates."""
    pv_in = PVCoordinates(pos_vec3d, Vector3D(0.0, 0.0, 0.0))
    pv_out = tf.transformPVCoordinates(pv_in)
    return pv_out.getPosition()


def Point_Geodetic2ECI(lat, lon, alt, time):
    """Point_Geodetic2ECI(lat,lon,alt,time) -> np.ndarray: (deg,deg,m,datetime) -> ECI position (m) as (3,1)."""
    frames = build_orekit_frames()
    wgs84 = frames["wgs84"]
    itrf = frames["itrf"]
    eme2000 = frames["eme2000"]

    gp = GeodeticPoint(radians(float(lat)), radians(float(lon)), float(alt))
    p_ecef = wgs84.transform(gp)  # Vector3D in ITRF

    date = _to_absolutedate(time)
    tf = itrf.getTransformTo(eme2000, date)
    p_eci = _transform_position(tf, p_ecef)

    return np.array([[p_eci.getX()], [p_eci.getY()], [p_eci.getZ()]], dtype=float)


def Point_ECI2Geodetic(x, y, z, time):
    """Point_ECI2Geodetic(x,y,z,time) -> np.ndarray: ECI position (m) -> (lat_deg,lon_deg,alt_m) as (3,1)."""
    frames = build_orekit_frames()
    wgs84 = frames["wgs84"]
    itrf = frames["itrf"]
    eme2000 = frames["eme2000"]

    date = _to_absolutedate(time)

    p_eci = Vector3D(float(x), float(y), float(z))
    tf = eme2000.getTransformTo(itrf, date)
    p_ecef = _transform_position(tf, p_eci)

    # IMPORTANT: in many Orekit versions this signature is required:
    #   GeodeticPoint gp = ellipsoid.transform(Vector3D position, Frame bodyFrame, AbsoluteDate date)
    gp = wgs84.transform(p_ecef, itrf, date)

    lat_deg = degrees(gp.getLatitude())
    lon_deg = degrees(gp.getLongitude())
    alt_m = float(gp.getAltitude())

    return np.array([[lat_deg], [lon_deg], [alt_m]], dtype=float)


def Point_ECEF2Geodetic(x, y, z):
    """Point_ECEF2Geodetic(x,y,z) -> np.ndarray: ECEF position (m) -> (lat_deg,lon_deg,alt_m) as (3,1)."""
    frames = build_orekit_frames()
    wgs84 = frames["wgs84"]
    itrf = frames["itrf"]
    date = frames["j2000"]  # any date works for ITRF->geodetic, but Orekit API requires one.

    p_ecef = Vector3D(float(x), float(y), float(z))
    gp = wgs84.transform(p_ecef, itrf, date)

    lat_deg = degrees(gp.getLatitude())
    lon_deg = degrees(gp.getLongitude())
    alt_m = float(gp.getAltitude())

    return np.array([[lat_deg], [lon_deg], [alt_m]], dtype=float)


def Point_Geodetic2ECEF(lat, lon, alt):
    """Point_Geodetic2ECEF(lat,lon,alt) -> np.ndarray: (deg,deg,m) -> ECEF position (m) as (3,1)."""
    frames = build_orekit_frames()
    wgs84 = frames["wgs84"]

    gp = GeodeticPoint(radians(float(lat)), radians(float(lon)), float(alt))
    p = wgs84.transform(gp)

    return np.array([[p.getX()], [p.getY()], [p.getZ()]], dtype=float)
