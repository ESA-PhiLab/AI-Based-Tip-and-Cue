import numpy as np
import pykep as pk

import re
from collections import defaultdict

def _interleaved_plane_index(k, P):
    """
    Map plane counter k = 0..P-1 to an interleaved index k'
    that alternates opposite sides of the RAAN circle.
    Even P: 0, P/2, 1, P/2+1, 2, P/2+2, ...
    Odd  P: 0, ceil(P/2), 1, ceil(P/2)+1, ...
    """
    if P <= 1:
        return 0
    if P % 2 == 0:
        half = P // 2
        return (k // 2) if (k % 2 == 0) else (half + k // 2)
    else:
        half_up = (P + 1) // 2
        return (k // 2) if (k % 2 == 0) else (half_up + k // 2) % P

def build_constellation(params, label, t0_pykep):
    if params["nSats"] != 0 and params["nPlanes"] != 0:
        F = int(params.get("F", 1))  # Walker phasing factor (choose gcd(F, nSats)=1 ideally)
        return get_constellation(
            params["a"], params["e"], params["i"], params["RAAN"],
            params["argp"], params["M"], params["nSats"], params["nPlanes"],
            t0_pykep, label, F=F, verbose=False
        )
    return [], [], []

def get_constellation(a, e, i_deg, RAAN_deg, argp_deg, M_deg,
                      nSats, nPlanes, t0, sat_name, F=1, verbose=False):
    """
    Walker-Delta constellation:
    - RAANs evenly spaced over W_area (default 360 deg).
    - Mean-longitude phasing with factor F, then recover M per plane.
    - Interleaved plane indexing to avoid mirrored-plane coincidences.
    """
    # ---config----------------------------------------------------------------------------------------------------------
    W_area = 360.0  # total RAAN span in degrees
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

    # Derived quantities
    P = int(nPlanes)
    S = int(nSats)
    N = S * P
    if P <= 0 or S <= 0:
        return [], [], []

    # RAAN spacing between planes
    pStep = (W_area * pk.DEG2RAD) / float(P)

    # In-plane spacing in mean longitude
    sStep = 2.0 * np.pi / float(S)

    # Inter-plane phasing in mean longitude
    interPlaneStep = (2.0 * np.pi * F) / float(N)

    planet_list = []
    elements_list = []

    # Reference mean longitude that preserves your M0 at plane 0, sat 0
    lambda0 = (M0 + w + W0) % (2.0 * np.pi)

    for plane_count in range(P):
        # Interleaved plane index for better spatial alternation
        k_prime = _interleaved_plane_index(plane_count, P)

        # RAAN of this plane
        W = (W0 + k_prime * pStep) % (2.0 * np.pi)

        for sat_count in range(S):
            # Target mean longitude for this satellite (Walker phasing)
            lam = (lambda0 + sat_count * sStep + k_prime * interPlaneStep) % (2.0 * np.pi)

            # Recover mean anomaly that achieves the desired lambda in this plane
            M = (lam - W - w) % (2.0 * np.pi)

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

    # Orbital period (all identical)
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

        # Semi-major axis
        try:
            sma = planet.orbital_elements[0]  # AU
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
