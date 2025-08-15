import numpy as np
import math
import pyproj
from .convert_reference_frames import get_lat_lon_alt_from_ecef

def rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

def normalize(v):
    norm = math.sqrt(sum(x**2 for x in v))
    return [x / norm for x in v]

def get_scene_characteristics(satellite_ecef, target_ecef, sun_ecef, img_height, img_width, GSD):
    lat_sat, lon_sat, altitude = get_lat_lon_alt_from_ecef(satellite_ecef)

    half_extent_nadir = (img_width* GSD) / 2

    fov_rad = 2 * np.arctan2(half_extent_nadir, altitude)
    fov_deg = math.degrees(fov_rad)

    transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4326")
    target_lat, target_lon, target_alt = transformer.transform(target_ecef[0], target_ecef[1], target_ecef[2])

    phi = np.radians(target_lat)
    lam = np.radians(target_lon)

    R = np.array([
        [-np.sin(lam), np.cos(lam), 0],
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    ])

    satellite_local = R @ (satellite_ecef - target_ecef)
    target_local = R @ (target_ecef - target_ecef)
    sun_local = R @ (sun_ecef - target_ecef)

    # Translate to target frame
    sun_direction = target_local - sun_local  # From sun to target
    sun_direction = normalize(sun_direction)

    off_nadir_rad = np.arccos(np.dot(satellite_local/np.linalg.norm(satellite_local), np.array([0,0,1])))
    azimuth_rad = np.arctan2(satellite_local[0], satellite_local[1]) % (2 * np.pi)

    return satellite_local, target_local, sun_direction, fov_deg, off_nadir_rad, azimuth_rad


def is_dark_from_sun_dir(target_ecef, sun_ecef, *,
                     threshold_deg=-18.0,   # apparent horizon (~sunrise/sunset)
                     model="spherical",      # "spherical" or "wgs84"
                     dir_type="target_to_sun"):
    """
    Returns (dark, sun_elevation_deg, threshold_deg).

    target_ecef    : 3-vector in ECEF [m]
    sun_direction  : 3-vector in same ECEF frame; direction from target toward Sun
                     (if it's Sun->target, set dir_type='sun_to_target')
    threshold_deg  : choose your cutoff:
                     0.0   = geometric horizon
                    -0.566 = apparent horizon (refraction + solar radius)
                    -6/-12/-18 = civil/nautical/astronomical night
    """
    r = np.asarray(target_ecef, float).reshape(3)
    s = np.asarray(sun_ecef, float).reshape(3)

    # local up at target
    if model == "spherical":
        u = r / np.linalg.norm(r)
    else:  # WGS-84 ellipsoidal normal
        a = 6378137.0
        e2 = 6.69437999014e-3
        b = a * math.sqrt(1.0 - e2)
        u = np.array([r[0] / (a * a), r[1] / (a * a), r[2] / (b * b)], float)
        u /= np.linalg.norm(u)

    # direction from target to Sun (normalize)
    v = s - r
    v /= np.linalg.norm(v)

    # elevation
    dot = float(np.clip(np.dot(v, u), -1.0, 1.0))
    elev_deg = math.degrees(math.asin(dot))

    return (elev_deg < float(threshold_deg)), elev_deg, float(threshold_deg)

import numpy as np, math

def dbg_sun_elevation(target_ecef, sun_ecef):
    r = np.asarray(target_ecef, float).reshape(3)
    s = np.asarray(sun_ecef, float).reshape(3)
    u = r / np.linalg.norm(r)  # spherical 'up'
    # two possible directions (flip to test sign)
    v = s - r
    v /= np.linalg.norm(v)
    elev1 = math.degrees(math.asin(np.clip(np.dot(v, u), -1.0, 1.0)))
    elev2 = -elev1
    # large-sun-distance approx (if s really is ECEF)
    elev_approx = math.degrees(math.asin(np.clip(np.dot(s/np.linalg.norm(s), u), -1.0, 1.0)))
    print(f"elev(target→Sun) = {elev1:.2f}°  |  flipped = {elev2:.2f}°  |  approx(using ŝ) = {elev_approx:.2f}°")
