import numpy as np
import pykep as pk
import math
import re
from collections import defaultdict
from math import sin, cos, sqrt, atan2

from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.orbits import WalkerConstellation
WalkerPattern = WalkerConstellation.Pattern

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

from java.util import ArrayList
from org.orekit.orbits import WalkerConstellationSlot
from paseos.custom_paseos.utils.constants import R_earth


def _pick_valid_F(nSats, nPlanes, F_user=None):
    """Pick valid Walker phasing factor using sats-per-plane convention."""
    if nPlanes <= 0 or nSats <= 0:
        return 1
    S = int(nSats)
    if S <= 1:
        return 1
    if F_user is not None and 1 <= F_user < S and math.gcd(F_user, S) == 1:
        return F_user
    for F in range(1, S):
        if math.gcd(F, S) == 1:
            return F
    return 1


def pykep_to_orekit(t0_pykep):
    """Convert pykep epoch to Orekit AbsoluteDate UTC."""
    utc = TimeScalesFactory.getUTC()
    if hasattr(t0_pykep, "to_datetime"):
        t0_dt = t0_pykep.to_datetime()
        return AbsoluteDate(t0_dt.year, t0_dt.month, t0_dt.day, t0_dt.hour, t0_dt.minute, t0_dt.second + t0_dt.microsecond / 1e6, utc)
    days_from_j2000 = t0_pykep.mjd2000
    j2000_tt = AbsoluteDate(2000, 1, 1, 12, 0, 0.0, TimeScalesFactory.getTT())
    return j2000_tt.shiftedBy(days_from_j2000 * 86400.0)


def solve_kepler_equation(M, e, tol=1e-12, max_iter=50):
    """Solve elliptic Kepler equation for E [rad]."""
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
    """Convert classical Keplerian elements to ECI state."""
    E = solve_kepler_equation(M_rad % (2.0 * np.pi), e)

    cosE, sinE = cos(E), sin(E)
    cos_nu = (cosE - e) / (1.0 - e * cosE)
    sin_nu = (sqrt(1.0 - e ** 2) * sinE) / (1.0 - e * cosE)
    nu = atan2(sin_nu, cos_nu)

    p = a_m * (1.0 - e ** 2)
    r_pqw = np.array([p * cos(nu) / (1.0 + e * cos(nu)), p * sin(nu) / (1.0 + e * cos(nu)), 0.0])
    v_pqw = np.array([-sqrt(mu / p) * sin(nu), sqrt(mu / p) * (e + cos(nu)), 0.0])

    cO, sO = cos(raan_rad), sin(raan_rad)
    ci, si = cos(i_rad), sin(i_rad)
    cw, sw = cos(argp_rad), sin(argp_rad)

    RzO = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    Rxi = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    Rzw = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])

    Q = RzO @ Rxi @ Rzw
    return Q @ r_pqw, Q @ v_pqw


def vec3(x, y=None, z=None):
    """Convert iterable or xyz values to Vector3D."""
    if y is None and z is None:
        x, y, z = x
    return Vector3D(float(x), float(y), float(z))


def build_constellation(params, label, t0_pykep):
    """Build either a Walker constellation or an explicit independent satellite group."""
    if params.get("build_constellation", True):
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

    satellites_cfg = params.get("satellites", [])
    if not satellites_cfg:
        return [], [], []

    print(f"Generated {label} independent satellites: {len(satellites_cfg)}")
    return get_independent_satellites(satellites_cfg, t0_pykep, label)


def get_independent_satellites(satellites_cfg, t0_pykep, sat_name, verbose=False):
    """Build explicitly defined satellites one by one, returning the same format as the Walker builder."""
    planet_list = []
    satellites = []

    for idx, sat in enumerate(satellites_cfg):
        orbit = sat["orbit"]
        name = sat.get("name", f"{sat_name}_plane{idx}_sat0")

        hp = float(orbit["hp"])
        ha = float(orbit["ha"])
        i_deg = float(orbit["i_deg"])
        RAAN_deg = float(orbit["RAAN_deg"])
        argp_deg = float(orbit["argp_deg"])
        M_deg = float(orbit["M_deg"])

        rp = R_earth + hp
        ra = R_earth + ha
        a = 0.5 * (ra + rp)
        e = (ra - rp) / (ra + rp)

        i = np.deg2rad(i_deg)
        raan = np.deg2rad(RAAN_deg)
        argp = np.deg2rad(argp_deg)
        M = np.deg2rad(M_deg)

        r_eci, v_eci = keplerian_to_pv(a, e, i, raan, argp, M, Constants.WGS84_EARTH_MU)

        planet = pk.planet.keplerian(
            t0_pykep,
            [a, e, i, raan, argp, M],
            pk.MU_EARTH,
            1.0,
            1.0,
            1.0,
            name
        )

        planet_list.append(planet)
        satellites.append((np.array(r_eci), np.array(v_eci)))

    period = planet_list[0].compute_period(t0_pykep) if planet_list else None

    if verbose:
        print(f"Created {len(planet_list)} independent satellites for {sat_name}")

    return planet_list, satellites, period

def get_constellation(a, e, i_deg, RAAN_deg, argp_deg, M_deg, nSats, nPlanes, t0_pykep, sat_name, F=1, verbose=False):
    """Build Walker-Delta constellation."""
    frame = FramesFactory.getEME2000()
    orekit_epoch = pykep_to_orekit(t0_pykep)
    mu = Constants.WGS84_EARTH_MU

    i = np.deg2rad(i_deg)
    raan = np.deg2rad(RAAN_deg)
    argp = np.deg2rad(argp_deg)
    M = np.deg2rad(M_deg)

    r_eci, v_eci = keplerian_to_pv(float(a), float(e), i, raan, argp, M, mu)
    pv_ref = PVCoordinates(vec3(r_eci), vec3(v_eci))
    ref_orbit = KeplerianOrbit(pv_ref, frame, orekit_epoch, mu)

    P = int(nPlanes)
    S = int(nSats)
    T = P * S

    walker = WalkerConstellation(T, P, int(F), WalkerPattern.DELTA)
    regularSlots = walker.buildRegularSlots(ref_orbit)
    slot_list_list = list(ArrayList.cast_(regularSlots))
    slot_list_list = [list(ArrayList.cast_(slot_list)) for slot_list in slot_list_list]
    slot_list_list = [[WalkerConstellationSlot.cast_(slot) for slot in slot_list] for slot_list in slot_list_list]

    planet_list = []
    satellites = []

    for p_idx, slots_in_plane in enumerate(slot_list_list):
        for s_idx, slot in enumerate(slots_in_plane):
            sat_orbit = slot.getOrbit()
            pv = sat_orbit.getPVCoordinates()

            pos = np.array([pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()])
            vel = np.array([pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()])

            kep_orbit = KeplerianOrbit.cast_(sat_orbit)
            a_out = kep_orbit.getA()
            e_out = kep_orbit.getE()
            i_out = kep_orbit.getI()
            raan_out = kep_orbit.getRightAscensionOfAscendingNode()
            argp_out = kep_orbit.getPerigeeArgument()
            M_out = kep_orbit.getMeanAnomaly()

            planet = pk.planet.keplerian(
                t0_pykep,
                [a_out, e_out, i_out, raan_out, argp_out, M_out],
                pk.MU_EARTH,
                1.0,
                1.0,
                1.0,
                f"{sat_name}_plane{p_idx}_sat{s_idx}"
            )

            planet_list.append(planet)
            satellites.append((pos, vel))

    period = planet_list[0].compute_period(t0_pykep) if planet_list else None

    if verbose:
        print(f"Created {len(planet_list)} Walker satellites for {sat_name}")

    return planet_list, satellites, period

def analyze_keplerian_constellation(planets):
    """Analyze constellation metadata from pk.planet.keplerian objects."""
    pattern = r"plane(\d+)_sat(\d+)"
    plane_sat_map = defaultdict(list)
    max_plane_id = -1
    max_sat_id = -1
    max_semi_major_axis = float("-inf")

    for entry in planets:
        name = getattr(entry, "name", None)

        try:
            sma = float(entry.orbital_elements[0])
            if sma >= max_semi_major_axis:
                max_semi_major_axis = sma
        except Exception:
            pass

        if name is None:
            continue

        match = re.search(pattern, name)
        if match:
            plane_id = int(match.group(1))
            sat_id = int(match.group(2))
            plane_sat_map[plane_id].append(sat_id)
            max_plane_id = max(max_plane_id, plane_id)
            max_sat_id = max(max_sat_id, sat_id)

    num_planes = max_plane_id + 1 if max_plane_id >= 0 else 0
    sats_per_plane = max_sat_id + 1 if max_sat_id >= 0 else 0

    if max_semi_major_axis == float("-inf"):
        max_semi_major_axis = None

    return num_planes, sats_per_plane, max_semi_major_axis