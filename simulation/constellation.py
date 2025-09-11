import numpy as np
import pykep as pk
import re
from collections import defaultdict
from math import sin, cos, sqrt, atan2
import math

# Orekit imports
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.orbits import WalkerConstellation
WalkerPattern = WalkerConstellation.Pattern

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

from java.util import ArrayList
from org.orekit.orbits import WalkerConstellationSlot


# ---------------------------
# Helpers
# ---------------------------


def _pick_valid_F(nSats, nPlanes, F_user=None):
    """
    Pick a valid Walker-Delta phasing factor F.
    Rule: gcd(F, S) == 1, where S = nSats per plane.
    If user F is provided and valid, return it.
    Otherwise return the smallest valid F >= 1.
    """
    if nPlanes <= 0 or nSats <= 0:
        return 1
    S = int(nSats / nPlanes)
    if S <= 1:
        return 1

    # If user provided a valid F, keep it
    if F_user is not None and 1 <= F_user < S and math.gcd(F_user, S) == 1:
        return F_user

    # Otherwise pick the smallest valid F
    for F in range(1, S):
        if math.gcd(F, S) == 1:
            return F
    return 1

def pykep_to_orekit(t0_pykep):
    """
    Convert pykep.epoch -> Orekit AbsoluteDate (UTC).
    Uses .to_datetime() if available; otherwise falls back to mjd2000 offset.
    """
    utc = TimeScalesFactory.getUTC()
    if hasattr(t0_pykep, "to_datetime"):
        t0_dt = t0_pykep.to_datetime()
        return AbsoluteDate(
            t0_dt.year, t0_dt.month, t0_dt.day,
            t0_dt.hour, t0_dt.minute,
            t0_dt.second + t0_dt.microsecond / 1e6,
            utc
        )
    else:
        days_from_j2000 = t0_pykep.mjd2000
        j2000_tt = AbsoluteDate(2000, 1, 1, 12, 0, 0.0, TimeScalesFactory.getTT())
        return j2000_tt.shiftedBy(days_from_j2000 * 86400.0)


def solve_kepler_equation(M, e, tol=1e-12, max_iter=50):
    """
    Solve Kepler's equation M = E - e*sinE for E (elliptic).
    Inputs/outputs in radians.
    """
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * sin(E) - M
        fp = 1.0 - e * cos(E)
        dE = -f / fp
        E = E + dE
        if abs(dE) < tol:
            break
    return E


def keplerian_to_pv(a_m, e, i_rad, raan_rad, argp_rad, M_rad, mu):
    """
    Classical Keplerian elements -> PV in ECI.
    a [m], e [-], i, RAAN, argp, M [rad], mu [m^3/s^2]
    Returns: r_vec [m], v_vec [m/s]
    """
    # 1) Solve for eccentric anomaly E
    E = solve_kepler_equation(M_rad % (2.0*np.pi), e)

    # 2) Position/velocity in perifocal (PQW)
    cosE, sinE = cos(E), sin(E)
    # true anomaly
    cos_nu = (cosE - e) / (1.0 - e * cosE)
    sin_nu = (sqrt(1.0 - e**2) * sinE) / (1.0 - e * cosE)
    nu = atan2(sin_nu, cos_nu)

    p = a_m * (1.0 - e**2)
    r_pqw = np.array([p * cos(nu) / (1.0 + e * cos(nu)),
                      p * sin(nu) / (1.0 + e * cos(nu)),
                      0.0])
    v_pqw = np.array([-sqrt(mu / p) * sin(nu),
                       sqrt(mu / p) * (e + cos(nu)),
                       0.0])

    # 3) Rotation PQW -> ECI (Rz(RAAN) * Rx(i) * Rz(ω))
    cO, sO = cos(raan_rad), sin(raan_rad)
    ci, si = cos(i_rad),   sin(i_rad)
    cw, sw = cos(argp_rad), sin(argp_rad)

    RzO = np.array([[ cO, -sO, 0.0],
                    [ sO,  cO, 0.0],
                    [0.0, 0.0, 1.0]])
    Rxi = np.array([[1.0, 0.0, 0.0],
                    [0.0,  ci, -si],
                    [0.0,  si,  ci]])
    Rzw = np.array([[ cw, -sw, 0.0],
                    [ sw,  cw, 0.0],
                    [0.0, 0.0, 1.0]])

    Q = RzO @ Rxi @ Rzw
    r_eci = Q @ r_pqw
    v_eci = Q @ v_pqw
    return r_eci, v_eci


def vec3(x, y=None, z=None):
    """
    Robust Vector3D wrapper.
    """
    if y is None and z is None:
        x, y, z = x
    return Vector3D(float(x), float(y), float(z))



# ---------------------------
# Constellation entry points
# ---------------------------

def build_constellation(params, label, t0_pykep):
    """
    Build constellation using Orekit WalkerConstellation (Delta pattern).
    Picks a valid phasing factor F if not specified.
    """
    nSats = int(params.get("nSats", 0))
    nPlanes = int(params.get("nPlanes", 0))
    if nSats != 0 and nPlanes != 0:
        F_user = params.get("F", None)
        F = _pick_valid_F(nSats, nPlanes, int(F_user) if F_user is not None else None)

        print(f"Generated {label} constellation with nPlanes {nPlanes}, nSats {nSats}, F {F}")
        return get_constellation(
            params["a"], params["e"], params["i"], params["RAAN"],
            params["argp"], params["M"], nSats, nPlanes,
            t0_pykep, label, F=F, verbose=False
        )
    return [], [], []


def get_constellation(a, e, i_deg, RAAN_deg, argp_deg, M_deg,
                      nSats, nPlanes, t0_pykep, sat_name, F=11, verbose=False):
    """
    Walker-Delta constellation using Orekit WalkerConstellation.
    Returns the same format as the PyKEP implementation:
      - planet_list: list of pk.planet.keplerian
      - satellites: list of (pos [m], vel [m/s])
      - period: orbital period [s]
    """

    frame = FramesFactory.getEME2000()
    orekit_epoch = pykep_to_orekit(t0_pykep)
    mu = Constants.WGS84_EARTH_MU  # [m^3/s^2]

    # Angles to radians
    i = np.deg2rad(i_deg)
    raan = np.deg2rad(RAAN_deg)
    argp = np.deg2rad(argp_deg)
    M = np.deg2rad(M_deg)

    # Reference orbit from elements
    a_m = float(a)
    r_eci, v_eci = keplerian_to_pv(a_m, e, i, raan, argp, M, mu)
    pv_ref = PVCoordinates(vec3(r_eci), vec3(v_eci))
    ref_orbit = KeplerianOrbit(pv_ref, frame, orekit_epoch, mu)

    # Walker (Delta pattern)
    P = int(nPlanes)
    S = int(nSats)
    T = P * S
    walker = WalkerConstellation(T, P, int(F), WalkerPattern.DELTA)

    # Build slots
    regularSlots = walker.buildRegularSlots(ref_orbit)
    slot_list_list = list(ArrayList.cast_(regularSlots))
    slot_list_list = [list(ArrayList.cast_(slot_list)) for slot_list in slot_list_list]
    slot_list_list = [[WalkerConstellationSlot.cast_(slot) for slot in slot_list]
                      for slot_list in slot_list_list]

    planet_list = []
    satellites = []

    for p_idx, slots_in_plane in enumerate(slot_list_list):
        for s_idx, slot in enumerate(slots_in_plane):
            sat_orbit = slot.getOrbit()
            pv = sat_orbit.getPVCoordinates()

            pos = np.array([pv.getPosition().getX(),
                            pv.getPosition().getY(),
                            pv.getPosition().getZ()])
            vel = np.array([pv.getVelocity().getX(),
                            pv.getVelocity().getY(),
                            pv.getVelocity().getZ()])

            sat_orbit = slot.getOrbit()
            kep_orbit = KeplerianOrbit.cast_(sat_orbit)  # cast to KeplerianOrbit

            a_out = kep_orbit.getA()
            e_out = kep_orbit.getE()
            i_out = kep_orbit.getI()
            raan_out = kep_orbit.getRightAscensionOfAscendingNode()
            argp_out = kep_orbit.getPerigeeArgument()
            M_out = kep_orbit.getMeanAnomaly()

            # Wrap into PyKEP planet.keplerian (same as old API)
            planet = pk.planet.keplerian(
                t0_pykep,
                [a_out, e_out, i_out, raan_out, argp_out, M_out],
                pk.MU_EARTH,   # central body mu
                1.0,           # mu_self
                1.0,           # radius
                1.0,           # safe radius
                f"{sat_name}_plane{p_idx}_sat{s_idx}"
            )

            planet_list.append(planet)
            satellites.append((pos, vel))

    # Orbital period (all identical)
    period = planet_list[0].compute_period(t0_pykep) if planet_list else None

    if verbose:
        print(f"Created {len(planet_list)} satellites with Orekit WalkerConstellation (Δ)")

    return planet_list, satellites, period


def analyze_keplerian_constellation(planets):
    """
    Analyze constellation metadata from Orekit-based planet_list.
    """
    pattern = r"plane(\d+)_sat(\d+)"
    plane_sat_map = defaultdict(list)
    max_plane_id = -1
    max_sat_id = -1
    max_semi_major_axis = float("-inf")

    for entry in planets:
        name = entry.get("name", None)
        orbit = entry.get("orbit", None)

        # Semi-major axis
        try:
            sma = orbit.getA()  # meters
            if sma >= max_semi_major_axis:
                max_semi_major_axis = sma
        except Exception:
            pass

        # Extract IDs from name (planeX_satY)
        if name is not None:
            match = re.search(pattern, name)
            if match:
                plane_id = int(match.group(1))
                sat_id = int(match.group(2))
                plane_sat_map[plane_id].append(sat_id)
                max_plane_id = max(max_plane_id, plane_id)
                max_sat_id = max(max_sat_id, sat_id)

    num_planes = max_plane_id + 1 if max_plane_id >= 0 else 0
    sats_per_plane = max_sat_id + 1 if max_sat_id >= 0 else 0
    return num_planes, sats_per_plane, max_semi_major_axis
