import math
import numpy as np

import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.time import AbsoluteDate, DateComponents, TimeComponents, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint
from org.orekit.utils import Constants, PVCoordinates, IERSConventions
from org.hipparchus.geometry.euclidean.threed import Vector3D
import pyproj

def get_ecef_from_lat_lon(satellite_lat, satellite_lon, satellite_alt, target_lat, target_lon, target_alt, datetime_utc):
    vm = orekit.initVM()
    setup_orekit_curdir(from_pip_library=True)

    # === Orekit Time & Frames ===
    utc = TimeScalesFactory.getUTC()
    date = DateComponents(datetime_utc.year, datetime_utc.month, datetime_utc.day)
    time = TimeComponents(datetime_utc.hour, datetime_utc.minute, float(datetime_utc.second))
    abs_date = AbsoluteDate(date, time, utc)

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    icrf = FramesFactory.getICRF()
    earth = ReferenceEllipsoid.getWgs84(itrf)

    # === Compute Target in ECEF ===
    geo_point_target = GeodeticPoint(math.radians(target_lat), math.radians(target_lon), target_alt)
    target_ecef_vec = earth.transform(geo_point_target)
    target_ecef = np.array([
        target_ecef_vec.getX(),
        target_ecef_vec.getY(),
        target_ecef_vec.getZ()
    ])

    geo_point_satellite = GeodeticPoint(math.radians(satellite_lat), math.radians(satellite_lon), satellite_alt)
    satellite_ecef_vec = earth.transform(geo_point_satellite)
    satellite_ecef = np.array([
        satellite_ecef_vec.getX(),
        satellite_ecef_vec.getY(),
        satellite_ecef_vec.getZ()
    ])

    # === Compute Sun in IRF, then convert to ECEF ===
    sun = CelestialBodyFactory.getSun()
    sun_pv_irf = sun.getPVCoordinates(abs_date, FramesFactory.getICRF())

    # Transform full PV from IRF to ECEF
    transform = FramesFactory.getICRF().getTransformTo(itrf, abs_date)
    sun_pv_ecef = transform.transformPVCoordinates(sun_pv_irf)

    # Extract just the position in ECEF
    sun_ecef = np.array([
        sun_pv_ecef.getPosition().getX(),
        sun_pv_ecef.getPosition().getY(),
        sun_pv_ecef.getPosition().getZ()
    ])

    return satellite_ecef, target_ecef, sun_ecef

def get_lat_lon_alt_from_ecef(satellite_ecef):
    # === Orekit init ===
    vm = orekit.initVM()
    setup_orekit_curdir(from_pip_library=True)

    # Use ITRF and WGS84 ellipsoid
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    icrf = FramesFactory.getICRF()
    earth = ReferenceEllipsoid.getWgs84(itrf)

    # ECEF Vector3D
    vec = Vector3D(float(satellite_ecef[0]),
                   float(satellite_ecef[1]),
                   float(satellite_ecef[2]))

    # === Transform ECEF → Geodetic ===
    geo_point = earth.transform(vec, earth.getBodyFrame(), None)

    lat_deg = math.degrees(geo_point.getLatitude())
    lon_deg = math.degrees(geo_point.getLongitude())
    alt_m = geo_point.getAltitude()

    return lat_deg, lon_deg, alt_m

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