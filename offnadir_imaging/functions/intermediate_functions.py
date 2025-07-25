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