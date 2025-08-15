# whales_ocean_sim_realtime.py
# Ocean-only "whale" targets with realistic speeds/headings.
# Real-time update (meters per real second), mask-based land avoidance, and PyVista globe rendering.

import os
import time
import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
from datetime import datetime, timezone

from settings import R_earth, max_abs_lat


# --------------------------- UTILS ---------------------------

def wrap_lon_deg(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0

def clamp_lat_deg(lat: float) -> float:
    return max(-90.0, min(90.0, lat))

def direct_geodesic(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """
    Move along a great-circle from (lat, lon) by 'distance_m' at 'bearing_deg' (0=N, 90=E).
    Spherical Earth model. Returns (lat2_deg, lon2_deg).
    """
    if distance_m == 0.0:
        return lat_deg, lon_deg
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    ang = distance_m / R_earth  # central angle (radians)

    sin_lat2 = math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brng)
    sin_lat2 = max(-1.0, min(1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)

    y = math.sin(brng) * math.sin(ang) * math.cos(lat1)
    x = math.cos(ang) - math.sin(lat1) * math.sin(lat2)
    lon2 = lon1 + math.atan2(y, x)

    lat2_deg = math.degrees(lat2)
    lon2_deg = wrap_lon_deg(math.degrees(lon2))
    return lat2_deg, lon2_deg

# --------------------------- LAND MASK ---------------------------

def build_land_mask(shp_dir: str, res_deg: float, tif_name: str, npy_name: str) -> np.ndarray:
    """
    Rasterize GSHHG Level 1 (land masses) + Level 5 (Antarctica ice front) into a binary mask.
    Output: numpy array with shape (height, width), values: land=1, water=0.
    """
    lon_min, lon_max = -180.0, 180.0
    lat_min, lat_max = -90.0, 90.0

    gdf_l1 = gpd.read_file(os.path.join(shp_dir, "GSHHS_f_L1.shp"))
    gdf_l5 = gpd.read_file(os.path.join(shp_dir, "GSHHS_f_L5.shp"))  # Antarctica ice front -> treat as land
    land = gpd.GeoDataFrame(pd.concat([gdf_l1, gdf_l5], ignore_index=True), crs=gdf_l1.crs)

    width = int(round((lon_max - lon_min) / res_deg))
    height = int(round((lat_max - lat_min) / res_deg))
    transform = rasterio.transform.from_origin(lon_min, lat_max, res_deg, res_deg)

    mask = features.rasterize(
        ((geom, 1) for geom in land.geometry),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    tif_path = os.path.join(shp_dir, tif_name)
    npy_path = os.path.join(shp_dir, npy_name)

    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=height, width=width,
        count=1, dtype=np.uint8, crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(mask, 1)

    np.save(npy_path, mask)
    return mask

def load_land_mask(shp_dir: str, npy_name: str, res_deg: float) -> tuple[np.ndarray, float]:
    npy_path = os.path.join(shp_dir, npy_name)
    mask = np.load(npy_path)
    return mask, res_deg

def is_water(lat: float, lon: float, mask: np.ndarray, res_deg: float) -> bool:
    row = int((90.0 - lat) / res_deg)
    col = int((lon + 180.0) / res_deg)
    row = max(0, min(mask.shape[0] - 1, row))
    col = max(0, min(mask.shape[1] - 1, col))
    return mask[row, col] == 0  # 0 = water, 1 = land

# --------------------------- INITIAL TARGETS ---------------------------

def generate_random_water_targets(n: int, mask: np.ndarray, res_deg: float, seed_val: int | None = None,
                                  max_abs_lat_val: float | None = max_abs_lat) -> list[tuple[float, float, float]]:
    if seed_val is not None:
        random.seed(seed_val)
    targets = []
    while len(targets) < n:
        lon = random.uniform(-180.0, 180.0)
        z = random.uniform(-1.0, 1.0)  # uniform in sin(lat)
        lat = math.degrees(math.asin(z))
        if max_abs_lat_val is not None and abs(lat) > max_abs_lat_val:
            continue
        if is_water(lat, lon, mask, res_deg):
            targets.append((lat, lon, 0.0))  # alt in meters
    return targets

# --------------------------- WHALE STATE & DYNAMICS ---------------------------

def init_whales(known_targets: list[tuple[float, float, float]], seed_val: int | None = None) -> dict[int, dict]:
    if seed_val is not None:
        random.seed(seed_val)
    whales = {}
    for idx, (lat, lon, alt_m) in enumerate(known_targets):
        whales[idx] = {
            "lat": lat,
            "lon": lon,
            "alt": alt_m,                   # meters
            "speed": random.uniform(0.5, 3.0),  # m/s
            "heading": random.uniform(0.0, 360.0)  # deg (0=N, 90=E)
        }
    return whales

def step_whale(whale: dict, mask: np.ndarray, res_deg: float, dt_sec: float, whale_propagation: dict) -> None:

    """
    Advance a whale by dt_sec seconds with OU speed, diffusive heading, and land avoidance.
    All distances in meters.
    """
    if dt_sec <= 0.0:
        return

    # Speed (Ornstein–Uhlenbeck)
    v = whale["speed"]
    k = whale_propagation["speed_mean_reversion_per_s"]
    noise = random.gauss(0.0, whale_propagation["speed_noise_sigma"] * math.sqrt(dt_sec))
    v = v + k * (whale_propagation["speed_mean"] - v) * dt_sec + noise
    v = max(whale_propagation["speed_min"], min(whale_propagation["speed_max"], v))

    # Heading diffusion
    h = whale["heading"]
    h = (h + random.gauss(0.0, whale_propagation["turn_std_deg_per_sqrt_s"] * math.sqrt(dt_sec))) % 360.0

    # Proposed move
    dist = v * dt_sec  # meters
    lat0, lon0 = whale["lat"], whale["lon"]
    lat1, lon1 = direct_geodesic(lat0, lon0, h, dist)

    # Keep within latitude limits
    if abs(lat1) > max_abs_lat:
        h = (h + 180.0) % 360.0
        lat1, lon1 = direct_geodesic(lat0, lon0, h, dist)

    # Land avoidance: steer around and shorten step if necessary
    tries = 0
    while (not is_water(lat1, lon1, mask, res_deg)) or (abs(lat1) > 89.9):
        delta = random.uniform(30.0, 150.0) * (1 if random.random() < 0.5 else -1)
        h = (h + delta) % 360.0
        dist *= 0.7
        lat1, lon1 = direct_geodesic(lat0, lon0, h, dist)
        tries += 1
        if tries >= whale_propagation["land_avoid_max_tries"]:
            lat1, lon1 = lat0, lon0  # stay put
            break

    whale["lat"], whale["lon"] = lat1, lon1
    whale["speed"], whale["heading"] = v, h

    return whale

