from custom_paseos.observation.EarthObservation import EOTools
from datetime import timedelta

import numpy as np
import pykep as pk

from custom_paseos.utils.help_functions import compute_orbital_period

from org.orekit.time import AbsoluteDate
from org.orekit.bodies import GeodeticPoint
from org.orekit.utils import PVCoordinates, Constants
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit
from org.hipparchus.geometry.euclidean.threed import Vector3D

from custom_paseos.utils.point_transformation import Point_Geodetic2ECEF
from custom_paseos.attitude.controller import StabilizedAttitudeController, SO3PIDACS
from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from paseos import ActorBuilder, SpacecraftActor

from shapely.geometry import Polygon
from shapely.ops import unary_union
from pyproj import Transformer, CRS
import numpy as np
import math

import numpy as np
from custom_paseos.utils.point_transformation import Point_Geodetic2ECI

import math
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory

from shapely.geometry import Polygon
from shapely.ops import unary_union
from pyproj import Transformer, CRS
import numpy as np
import math

def init_eo_tools(tip_actors, cue_actors, fov_tip, fov_cue, eul_ang_tip_init, eul_ang_cue_init):

    eo_tools_dict = {}

    for actor in tip_actors:
        eo_tools_dict[actor.name] = EOTools(
            local_actor=actor,
            initial_eul_ang_deg=eul_ang_tip_init,
            fov_act_deg=[fov_tip],
            fov_alt_deg=[fov_tip],
        )

    for actor in cue_actors:
        eo_tools_dict[actor.name] = EOTools(
            local_actor=actor,
            initial_eul_ang_deg=eul_ang_cue_init,
            fov_act_deg=[fov_cue],
            fov_alt_deg=[fov_cue],
        )

    return eo_tools_dict


def _build_one(actor_list, eo_tools_dict, p):

    # Unpack with safe defaults in case a key is missing
    cutoff_freq_gnc   = p.get("cutoff_freq_gnc")
    anti_windup_gain  = p.get("anti_windup_gain")
    Kp_acs            = p.get("Kp_acs")
    Kd_acs            = p.get("Kd_acs")
    Ki_acs            = p.get("Ki_acs")

    ang_vel_max_gnc   = p.get("ang_vel_max_gnc")
    ang_vel_max_acs   = p.get("ang_vel_max_acs")

    ang_accel_max_gnc = p.get("ang_accel_max_gnc")
    tau_max_acs       = p.get("tau_max_acs")
    J_sat             = np.asarray(p.get("J_sat"), dtype=float)

    out = {}
    for actor in actor_list:
        eo_tools = eo_tools_dict[actor.name]

        guidance = StabilizedAttitudeController(
            initial_eul_ang_deg=eo_tools.eul_ang_deg,
            cutoff_freq_gnc=cutoff_freq_gnc,
            ang_accel_max_gnc=ang_accel_max_gnc,
            ang_vel_max_gnc=ang_vel_max_gnc,
        )

        acs = SO3PIDACS(
            eul_ang_deg_init=eo_tools.eul_ang_deg,
            J_sat=J_sat,
            Kp_acs=Kp_acs,
            Kd_acs=Kd_acs,
            Ki_acs=Ki_acs,
            tau_max_acs=tau_max_acs,
            ang_vel_max_acs=ang_vel_max_acs, anti_windup_gain=anti_windup_gain
        )

        out[actor.name] = {"guidance": guidance, "acs": acs}
    return out

def init_attitude_controllers(tip_actors, cue_actors, eo_tools_dict,
                              controller_params_tip, controller_params_cue):

    controllers = {}
    if tip_actors != None:
        controllers.update(_build_one(tip_actors, eo_tools_dict, controller_params_tip))

    if cue_actors != None:
        controllers.update(_build_one(cue_actors, eo_tools_dict, controller_params_cue))
    return controllers

def cleanup_tasked_targets(tasked_targets, current_time, timeout):
    """
    Remove tasks that have timed out, using Whale.tip_time.
    tasked_targets: dict[int, Whale]
    """
    expired = [idx for idx, w in tasked_targets.items()
               if (w.tip_time is not None) and (current_time > w.tip_time + timedelta(seconds=timeout))]

    for idx in expired:
        print("!! Time-out: remove tasking request", idx)
        del tasked_targets[idx]



def propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits):

    r, v = actor.get_position_velocity(t_pykep)

    if show_orbits:
        trajectories[actor.name]["r"][n_steps] = r
        trajectories[actor.name]["v"][n_steps] = v

    r_vec = np.array(r).reshape(3, 1)
    v_vec = np.array(v).reshape(3, 1)

    return r_vec, v_vec, r, v


def satellite_in_shadow(r_vec, sun_vec_eci, earth_radius):
    """
    Cylindrical umbra check (fast and robust for LEO).
    r_vec: satellite ECI position [m], shape (3,) or (3,1)
    sun_vec_eci: Sun ECI vector [m] at the same epoch, shape (3,)
    earth_radius: Earth equatorial radius [m]
    Returns True if the satellite is in Earth's shadow (umbra).
    """
    r = np.asarray(r_vec, dtype=float).reshape(-1)        # (3,)
    s = np.asarray(sun_vec_eci, dtype=float).reshape(-1)  # (3,)

    # Unit vector from Earth to Sun
    s_hat = s / np.linalg.norm(s)

    # If satellite is on the dayside (towards the Sun), it cannot be in shadow
    if np.dot(r, s_hat) > 0.0:
        return False

    # Perpendicular distance from the Earth–Sun axis
    # |r x s_hat| = |r| * sin(theta); in umbra if this is < Earth radius
    d = np.linalg.norm(np.cross(r, s_hat))

    return d < earth_radius


def target_illuminated(lat, lon, alt, t_datetime, earth, sun, iers2010):
    """Return True if target is illuminated (Sun above horizon)."""
    sun_pos = sun.getPVCoordinates(AbsoluteDate(t_datetime),
                                   FramesFactory.getITRF(iers2010, True)).getPosition()

    gp = GeodeticPoint(np.radians(lat), np.radians(lon), alt)
    elev = earth.getElevation(gp, earth.getBodyFrame(), AbsoluteDate(t_datetime), sun_pos)
    return elev > 0

def daylight_mask(targets, sun_vec):
    """
    Compute which targets are in daylight.
    targets: dict[int, Whale]
    sun_vec: Sun position vector in ECEF (np.array of shape (3,))
    Returns: set of ids that are illuminated
    """
    illuminated = set()
    sun_unit = sun_vec / np.linalg.norm(sun_vec)

    for tid, whale in targets.items():
        lat, lon, alt = whale.lat, whale.lon, whale.alt
        r = Point_Geodetic2ECEF(lat, lon, alt).flatten()
        r_unit = r / np.linalg.norm(r)
        if np.dot(r_unit, sun_unit) > 0:
            illuminated.add(tid)

    return illuminated


def convert_M_to_lv(orbital_elements, epoch):
    """
    Convert mean anomaly (deg) in orbital_elements to true anomaly (deg).
    orbital_elements = [a, e, i, RAAN, argp, M_deg]
    """
    a, e, i, raan, argp, M_deg = orbital_elements
    inertialFrame = FramesFactory.getEME2000()

    # Temporary orbit with mean anomaly
    temp_orbit = KeplerianOrbit(
        a, e, i,
        argp, raan, math.radians(M_deg),
        PositionAngleType.MEAN,
        inertialFrame, epoch, Constants.WGS84_EARTH_MU
    )
    lv_deg = math.degrees(temp_orbit.getTrueAnomaly())

    return [a, e, i, argp, raan, lv_deg]


def pointing_cost(task, eo_tools, r_vec, v_vec, t_datetime):
    """
    Compute the angular cost of retargeting to this task.
    Lower = easier to point.

    task: dict with at least "coord" (lat, lon, alt)
    eo_tools: EO tools object for the satellite
    r_vec, v_vec: satellite state vectors in ECI
    t_datetime: datetime of observation
    """
    # Convert target to pointing vector
    _, pointing_vec_brf = eo_tools.off_nadir_pointing_angle(
        r_eci=r_vec,
        v_eci=v_vec,
        target_geodetic=task["coord"],
        t_datetime=t_datetime
    )

    # Desired Euler angles for that target
    eul_target = eo_tools.pointing_attitude_brf(pointing_vec_brf)

    # Difference from current pointing
    delta = np.abs(np.array(eul_target) - np.array(eo_tools.eul_ang_deg))
    return np.linalg.norm(delta)

def count_orbits_completed(a_m, sim_duration_seconds):
    """
    Return (n_orbits_float, n_full_orbits, residual_seconds, period_seconds)
      a_m: semi-major axis [m]
      sim_duration_seconds: simulated elapsed time [s]
    """
    T = compute_orbital_period(a_m)          # [s]
    n_float = sim_duration_seconds / T
    n_full = int(n_float)
    residual = sim_duration_seconds - n_full * T
    return n_float, n_full, residual, T

def compute_coverage_fraction(fov_polygons_tip, fov_polygons_cue, R_earth, inclination_deg, a_m, sim_duration_seconds):
    """
    Compute coverage metrics from FOV polygons:
      - Total covered area and fraction of Earth
      - Max possible coverage given inclination
      - Per-orbit coverage
      - Efficiency (total and per orbit)
    """

    # Equal-area projection (Mollweide) for accurate area computation
    crs_equal_area = CRS.from_proj4("+proj=moll +lon_0=0 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs")
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs_equal_area, always_xy=True)

    # Transform polygons into projection and collect valid ones
    polys = []
    for fov_list in (fov_polygons_tip + fov_polygons_cue):
        if fov_list is None: continue
        lons, lats = fov_list[:, 1], fov_list[:, 0]
        if lons[0] != lons[-1] or lats[0] != lats[-1]:  # ensure polygon is closed
            lons, lats = np.append(lons, lons[0]), np.append(lats, lats[0])
        x, y = transformer.transform(lons, lats)
        poly = Polygon(zip(x, y))
        if poly.is_valid and not poly.is_empty: polys.append(poly)

    # Union of all footprints → total covered area
    union_poly = unary_union(polys) if polys else None
    area_covered_m2 = union_poly.area if union_poly else 0.0
    area_covered_km2 = area_covered_m2 / 1e6
    area_total_m2 = 4.0 * math.pi * R_earth**2
    area_total_km2 = area_total_m2 / 1e6

    # Theoretical max coverage based on inclination (sin(i) rule)
    area_mission_fraction_total = math.sin(math.radians(abs(inclination_deg)))
    area_mission_m2 = area_mission_fraction_total * area_total_m2
    area_mission_km2 = area_mission_m2 / 1e6

    area_covered_fraction_total = area_covered_km2 / area_total_km2
    area_covered_fraction_mission = (area_covered_m2 / area_mission_m2) if area_mission_m2 > 0 else 0.0       # Efficiency relative to theoretical maximum

    # Normalize per orbit
    T = compute_orbital_period(a_m)                        # orbital period [s]
    n_orbits = sim_duration_seconds / T if T > 0 else 0.0  # number of completed orbits
    area_covered_per_orbit_km2 = area_covered_km2 / n_orbits if n_orbits > 0 else 0.0
    area_covered_per_orbit_fraction_total = area_covered_per_orbit_km2 / area_total_km2 if n_orbits > 0 else 0.0
    area_covered_per_orbit_fraction_mission = area_covered_per_orbit_km2 / area_mission_km2 if n_orbits > 0 else 0.0

    return area_total_km2, area_mission_km2, area_covered_km2, area_covered_per_orbit_km2, area_mission_fraction_total, area_covered_fraction_total, area_covered_fraction_mission, area_covered_per_orbit_fraction_total, area_covered_per_orbit_fraction_mission




