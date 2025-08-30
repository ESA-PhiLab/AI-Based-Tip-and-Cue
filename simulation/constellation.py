import numpy as np
import pykep as pk

import re
from collections import defaultdict

def build_constellation(params, label, t0_pykep):
    if params["nSats"] != 0 and params["nPlanes"] != 0:
        return get_constellation(
            params["a"], params["e"], params["i"], params["RAAN"],
            params["argp"], params["M"], params["nSats"], params["nPlanes"],
            t0_pykep, label, verbose=True
        )
    return [], [], []

def get_constellation(a, e, i_deg, RAAN_deg, argp_deg, M_deg,
                      nSats, nPlanes, t0, sat_name, verbose=True):
    # ---extra-params----------------------------------------------------------------------------------------------------
    W_area = 360.0       # RAAN span (deg) across all planes
    # -------------------------------------------------------------------------------------------------------------------

    # Angles to radians
    i = i_deg * pk.DEG2RAD
    W0 = RAAN_deg * pk.DEG2RAD
    w = argp_deg * pk.DEG2RAD
    M0 = M_deg * pk.DEG2RAD

    mu_central_body = pk.MU_EARTH
    mu_self = 1.0
    radius = 1.0
    safe_radius = 1.0

    # RAAN spacing between planes
    pStep = (W_area * pk.DEG2RAD) / float(nPlanes)
    # Mean anomaly spacing between satellites in a plane
    sStep = 2.0 * np.pi / float(nSats) if nSats > 0 else 0.0
    # Inter-plane phasing: 1 "slot" offset per plane
    # (like Walker delta = 1), ensures planes are staggered
    interPlaneStep = 2.0 * np.pi / float(nSats * nPlanes) if (nSats * nPlanes) > 0 else 0.0

    planet_list = []
    elements_list = []

    plane_count = 0
    for _ in range(nPlanes):
        W = W0 + plane_count * pStep

        sat_count = 0
        for _ in range(nSats):
            # mean anomaly: base + intra-plane spacing + inter-plane phasing
            M = M0 + sat_count * sStep + plane_count * interPlaneStep

            planet = pk.planet.keplerian(
                t0,
                [a, e, i, W, w, M],
                mu_central_body,
                mu_self,
                radius,
                safe_radius,
                f"{sat_name}_plane{plane_count}_sat{sat_count}",
            )
            planet_list.append(planet)
            elements_list.append([a, e, i, W, w, M])

            sat_count += 1

        plane_count += 1

    # Period of the first satellite (all have the same)
    period = planet_list[0].compute_period(t0) if planet_list else None

    if verbose:
        print(f"Created {len(elements_list)} satellites...")
        print("Computing constellation's positions and velocities...")

    satellites = []
    for elements in elements_list:
        pos, v = pk.par2ic(elements, pk.MU_EARTH)
        satellites.append((np.asarray(pos), np.asarray(v)))

    if verbose:
        print("Done!")

    return planet_list, satellites, period

def analyze_keplerian_constellation(planets):
    """
    Analyzes a list of pykep.planet.keplerian objects to extract:
    - Total number of planes (max plane index + 1)
    - Satellites per plane (max satellite index + 1)
    - Mapping of plane_id -> [sat_ids]
    - Highest semi-major axis in AU

    Returns:
        num_planes (int)
        sats_per_plane (int)
        plane_sat_map (dict)
        max_semi_major_axis (float)
    """


    pattern = r"plane(\d+)_sat(\d+)"
    plane_sat_map = defaultdict(list)
    max_plane_id = -1
    max_sat_id = -1
    max_semi_major_axis = float("-inf")

    for planet in planets:
        name = planet.name

        if name is None:
            continue

        # Get semi-major axis
        try:
            sma = planet.orbital_elements[0]  # semi-major axis in AU
            if sma >= max_semi_major_axis:
                max_semi_major_axis = sma

        except Exception:
            pass

        # Extract IDs from name
        match = re.search(pattern, name)
        if match:
            plane_id = int(match.group(1))
            sat_id = int(match.group(2))

            plane_sat_map[plane_id].append(sat_id)
            max_plane_id = max(max_plane_id, plane_id)
            max_sat_id = max(max_sat_id, sat_id)
        else:
            print(f"Warning: name '{name}' doesn't match expected pattern")

    num_planes = max_plane_id + 1 if max_plane_id >= 0 else 0
    sats_per_plane = max_sat_id + 1 if max_sat_id >= 0 else 0

    return num_planes, sats_per_plane, max_semi_major_axis
