# simulation/satellite_config_utils.py

import math
import numpy as np
from paseos.custom_paseos.utils.constants import R_earth
from paseos.custom_paseos.utils.help_functions import compute_orbital_period
from satellite_config import (
    SATELLITE_MODE,
    CONSTELLATION_CONFIG,
    INDEPENDENT_SATELLITES,
    TIP_CUE_ORBIT_LINK,
)


def compute_a_e_from_hp_ha(hp, ha):
    """Convert perigee/apogee altitude [m] to semi-major axis and eccentricity."""
    rp = R_earth + float(hp)
    ra = R_earth + float(ha)
    a = 0.5 * (ra + rp)
    e = (ra - rp) / (ra + rp)
    return a, e


def compute_fov_from_swath(swath_m, a_m):
    """Convert swath width [m] to full FOV angle [deg]."""
    return math.degrees(2.0 * math.atan(float(swath_m) / (2.0 * (float(a_m) - R_earth))))


def shift_mean_anomaly_by_delay(orbit, delay_seconds):
    """Return orbit copy with M_deg shifted by time delay."""
    orbit_new = dict(orbit)
    a, _ = compute_a_e_from_hp_ha(orbit_new["hp"], orbit_new["ha"])
    T = compute_orbital_period(a)
    delta_M_deg = 360.0 * float(delay_seconds) / float(T)
    orbit_new["M_deg"] = (float(orbit_new["M_deg"]) + delta_M_deg) % 360.0
    return orbit_new


def resolve_constellation_orbit(group_name, orbit):
    """Resolve orbit for constellation mode, including optional Tip-from-Cue delay link."""
    if group_name != "Tip":
        return dict(orbit)

    if not TIP_CUE_ORBIT_LINK.get("enabled", False):
        return dict(orbit)

    if TIP_CUE_ORBIT_LINK.get("reference", "Cue") != "Cue":
        return dict(orbit)

    cue_orbit = dict(CONSTELLATION_CONFIG["Cue"]["orbit"])
    return shift_mean_anomaly_by_delay(cue_orbit, TIP_CUE_ORBIT_LINK.get("delay_seconds", 0.0))


def normalize_satellite_entry(entry):
    """Expand one independent satellite with derived a, e, fov_deg."""
    orbit = dict(entry["orbit"])
    sensor = dict(entry["sensor"])
    a, e = compute_a_e_from_hp_ha(orbit["hp"], orbit["ha"])
    fov_deg = compute_fov_from_swath(sensor["swath_m"], a)

    return {
        "name": entry["name"],
        "orbit": orbit,
        "sensor": sensor,
        "a": a,
        "e": e,
        "fov_deg": fov_deg,
    }


def normalize_constellation_group(group_name):
    """Normalize one constellation group."""
    raw = CONSTELLATION_CONFIG[group_name]
    orbit = resolve_constellation_orbit(group_name, raw["orbit"])
    sensor = dict(raw["sensor"])
    a, e = compute_a_e_from_hp_ha(orbit["hp"], orbit["ha"])
    fov_deg = compute_fov_from_swath(sensor["swath_m"], a)

    return {
        "build_constellation": True,
        "nPlanes": int(raw["nPlanes"]),
        "nSats": int(raw["nSats"]),
        "orbit": orbit,
        "sensor": sensor,
        "a": a,
        "e": e,
        "fov_deg": fov_deg,
        "satellites": None,
    }


def normalize_independent_group(group_name):
    """Normalize one independent-satellite group."""
    sats = [normalize_satellite_entry(entry) for entry in INDEPENDENT_SATELLITES[group_name]]
    a_vals = [sat["a"] for sat in sats]
    i_vals = [sat["orbit"]["i_deg"] for sat in sats]

    return {
        "build_constellation": False,
        "nPlanes": len(sats),
        "nSats": 1,
        "orbit": None,
        "sensor": None,
        "a": float(np.mean(a_vals)) if a_vals else None,
        "e": None,
        "fov_deg": None,
        "i_deg": float(np.mean(i_vals)) if i_vals else None,
        "satellites": sats,
    }

def get_group_config(group_name):
    """Return normalized config for Tip or Cue."""
    return normalize_constellation_group(group_name) if SATELLITE_MODE[group_name] else normalize_independent_group(group_name)


def get_satellite_group_configs():
    """Return normalized Tip/Cue config dict."""
    return {
        "Tip": get_group_config("Tip"),
        "Cue": get_group_config("Cue"),
    }