import numpy as np
import pyproj

import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.time import AbsoluteDate, DateComponents, TimeComponents, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import IERSConventions


# WGS84 geodetic (lat,lon,alt) <-> ECEF (x,y,z)
_GEOD2ECEF = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
_ECEF2GEOD = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m) -> np.ndarray:
    """geodetic_to_ecef(lat_deg,lon_deg,alt_m) -> np.ndarray: WGS84 geodetic -> ECEF (meters)."""
    x, y, z = _GEOD2ECEF.transform(float(lon_deg), float(lat_deg), float(alt_m))
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(ecef_xyz) -> tuple[float, float, float]:
    """ecef_to_geodetic(ecef_xyz) -> (lat_deg,lon_deg,alt_m): ECEF (meters) -> WGS84 geodetic."""
    x, y, z = np.asarray(ecef_xyz, float).reshape(3)
    lon, lat, alt = _ECEF2GEOD.transform(float(x), float(y), float(z))
    return float(lat), float(lon), float(alt)


def get_ecef_from_lat_lon(satellite_lat, satellite_lon, satellite_alt,
                          target_lat, target_lon, target_alt,
                          datetime_utc,
                          generate_nadir: bool = False):
    """get_ecef_from_lat_lon(satellite_lat,satellite_lon,satellite_alt,target_lat,target_lon,target_alt,datetime_utc,generate_nadir=False) -> tuple: (sat_ecef,tgt_ecef,sun_ecef)."""
    # Target ECEF (WGS84)
    target_ecef = geodetic_to_ecef(target_lat, target_lon, target_alt)

    # Satellite ECEF
    if generate_nadir:
        # Force geocentric nadir: sat on target's Earth-center radial line
        rt = float(np.linalg.norm(target_ecef))
        if rt <= 0.0:
            raise ValueError("target_ecef is zero; cannot define radial direction")
        u = target_ecef / rt
        delta_h = float(satellite_alt) - float(target_alt)
        satellite_ecef = u * (rt + delta_h)
    else:
        satellite_ecef = geodetic_to_ecef(satellite_lat, satellite_lon, satellite_alt)

    # Sun ECEF (keep Orekit for this; still returns ECEF/ITRF)
    vm = orekit.initVM()
    setup_orekit_curdir(from_pip_library=True)

    utc = TimeScalesFactory.getUTC()
    date = DateComponents(datetime_utc.year, datetime_utc.month, datetime_utc.day)
    time = TimeComponents(datetime_utc.hour, datetime_utc.minute, float(datetime_utc.second))
    abs_date = AbsoluteDate(date, time, utc)

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    sun = CelestialBodyFactory.getSun()
    sun_pv_icrf = sun.getPVCoordinates(abs_date, FramesFactory.getICRF())
    transform = FramesFactory.getICRF().getTransformTo(itrf, abs_date)
    sun_pv_ecef = transform.transformPVCoordinates(sun_pv_icrf)

    sun_ecef = np.array([
        sun_pv_ecef.getPosition().getX(),
        sun_pv_ecef.getPosition().getY(),
        sun_pv_ecef.getPosition().getZ()
    ], dtype=float)

    return satellite_ecef, target_ecef, sun_ecef


def get_lat_lon_alt_from_ecef(satellite_ecef):
    """get_lat_lon_alt_from_ecef(satellite_ecef) -> tuple: (lat_deg,lon_deg,alt_m) from ECEF."""
    return ecef_to_geodetic(satellite_ecef)


def compute_max_glint_satellite_ecef(target_ecef, sun_ecef, glint_distance_m):
    # Step 1: Sun-to-target direction vector (incoming light direction)
    sun_dir = sun_ecef - target_ecef
    sun_dir = sun_dir / np.linalg.norm(sun_dir)

    # Step 2: Surface normal at target (using WGS84 ellipsoid normal)
    transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)
    lon, lat, _ = transformer.transform(*target_ecef)

    phi = np.radians(lat)
    lam = np.radians(lon)
    surface_normal = np.array([
        np.cos(phi) * np.cos(lam),
        np.cos(phi) * np.sin(lam),
        np.sin(phi)
    ])

    # Step 3: Reflect sun direction around surface normal
    glint_dir = 2 * np.dot(surface_normal, sun_dir) * surface_normal - sun_dir
    glint_dir = glint_dir / np.linalg.norm(glint_dir)  # normalize

    # Step 4: Move satellite along reflected direction
    satellite_ecef = target_ecef + glint_dir * glint_distance_m

    return satellite_ecef

def sat_ecef_geocentric_over_target(tgt_ecef, sat_alt_m, tgt_alt_m=0.0) -> np.ndarray:
    """sat_ecef_geocentric_over_target(tgt_ecef,sat_alt_m,tgt_alt_m=0.0) -> np.ndarray: Satellite ECEF on Earth-center radial line for zero geocentric off-nadir."""
    tgt = np.asarray(tgt_ecef, float).reshape(3)
    rt = float(np.linalg.norm(tgt))
    if rt <= 0.0:
        raise ValueError("tgt_ecef must be non-zero")
    u = tgt / rt
    return u * (rt + (float(sat_alt_m) - float(tgt_alt_m)))


