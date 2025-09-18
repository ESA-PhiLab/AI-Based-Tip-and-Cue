# sim_utils.py
# -----------------------------------------------------------------------------
# Simulation utilities: initialization, cleanup, propagation, coverage analysis
# -----------------------------------------------------------------------------

from datetime import datetime, timedelta
import math
import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union
from pyproj import Transformer, CRS

from org.orekit.time import AbsoluteDate
from org.orekit.bodies import GeodeticPoint
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory

from custom_paseos.utils.help_functions import compute_orbital_period
from custom_paseos.utils.point_transformation import Point_Geodetic2ECEF
from custom_paseos.observation.observation_model import EOTools
from custom_paseos.attitude.attitude_model import AttitudeModel


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def init_eo_tools(tip_actors, cue_actors, fov_tip, fov_cue, offnadir_limit):
    """Initialize EO tools for all actors (geometry only)."""
    eo_tools = {}
    for actor in tip_actors:
        eo_tools[actor.name] = EOTools(actor, [fov_tip], [fov_tip], offnadir_limit)
    for actor in cue_actors:
        eo_tools[actor.name] = EOTools(actor, [fov_cue], [fov_cue], offnadir_limit)
    return eo_tools


def init_attitude_models(tip_actors, cue_actors, eul_ang_tip_init, eul_ang_cue_init):
    """Initialize attitude models for all actors (stateful dynamics)."""
    att_models = {}
    for actor in tip_actors:
        att_models[actor.name] = AttitudeModel(
            actor,
            actor_initial_attitude_deg=eul_ang_tip_init,  # degrees
            actor_initial_angular_velocity=[0.0, 0.0, 0.0],
        )
        att_models[actor.name].set_target_euler(eul_ang_tip_init)

    for actor in cue_actors:
        att_models[actor.name] = AttitudeModel(
            actor,
            actor_initial_attitude_deg=eul_ang_cue_init,  # degrees
            actor_initial_angular_velocity=[0.0, 0.0, 0.0],
        )
        att_models[actor.name].set_target_euler(eul_ang_cue_init)
    return att_models



def link_eo_attitude(eo_tools, att_models):
    """Attach attitude models to EO tools for unified access."""
    for name, eo in eo_tools.items():
        if name in att_models:
            eo.set_attitude_model(att_models[name])


# -----------------------------------------------------------------------------
# Task cleanup
# -----------------------------------------------------------------------------
def cleanup_timeout_targets(all_targets, tasked_targets, current_time, timeout, cleanup_idx,
                            eo_tools_dict, att_models_dict, eul_default=[0.0, 0.0, 0.0]):
    """
    Remove tasks that have timed out or were scheduled for cleanup.
    Also clears the cue actor's local state & MPC.
    """
    expired = [idx for idx, w in tasked_targets.items()
               if ((w.t_observed_tip and current_time > w.t_observed_tip + timedelta(seconds=timeout)) or
                   (w.t_observed_cue and current_time > w.t_observed_cue + timedelta(seconds=timeout)))]

    all_cleanup_idx = list({*expired, *cleanup_idx})

    for idx in expired:
        print(f"!! Time-out Target {idx}")

    for idx in all_cleanup_idx:
        w = all_targets.get(idx)
        if w and getattr(w, "assigned_cue", None):
            _clear_actor_task(w.assigned_cue, idx, eo_tools_dict, att_models_dict, eul_default)

        tasked_targets.pop(idx, None)

        if w:

            w.state_observing = 0
            w.state_tasked = 0
            w.state_confirming = 0
            w.assigned_cue = None

    return all_cleanup_idx


def _clear_actor_task(actor_name, task_id, eo_tools_dict, att_models_dict, eul_default=[0.0, 0.0, 0.0]):
    """Clear a specific task from an actor and reset its attitude target."""
    if actor_name is None or actor_name not in eo_tools_dict:
        return

    eo = eo_tools_dict[actor_name]

    eo.current_task = None
    eo.offnadir_unbound_target = None

    # Always clean the task queue
    if hasattr(eo, "task_queue") and eo.task_queue:
        eo.task_queue = [t for t in eo.task_queue if t.get("target_id") != task_id]

    eo_tools_dict[actor_name] = eo




# -----------------------------------------------------------------------------
# Propagation and shadow/illumination
# -----------------------------------------------------------------------------
def propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits):
    r, v = actor.get_position_velocity(t_pykep)
    if show_orbits:
        trajectories[actor.name]["r"][n_steps] = r
        trajectories[actor.name]["v"][n_steps] = v
    return np.array(r).reshape(3, 1), np.array(v).reshape(3, 1), r, v


def satellite_in_shadow(r_vec, sun_vec_eci, earth_radius):
    """Fast cylindrical umbra check (LEO)."""
    r = np.asarray(r_vec, float).reshape(-1)
    s = np.asarray(sun_vec_eci, float).reshape(-1)
    s_hat = s / np.linalg.norm(s)
    if np.dot(r, s_hat) > 0.0:
        return False
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
    """Return set of ids of targets in daylight."""
    illuminated = set()
    sun_unit = sun_vec / np.linalg.norm(sun_vec)
    for tid, whale in targets.items():
        r = Point_Geodetic2ECEF(whale.lat, whale.lon, whale.alt).flatten()
        if np.dot(r / np.linalg.norm(r), sun_unit) > 0:
            illuminated.add(tid)
    return illuminated


# -----------------------------------------------------------------------------
# Orbital elements
# -----------------------------------------------------------------------------
def convert_M_to_lv(orbital_elements, epoch):
    """
    Convert mean anomaly (deg) in orbital_elements to true anomaly (deg).
    orbital_elements = [a, e, i, RAAN, argp, M_deg]
    """
    a, e, i, raan, argp, M_deg = orbital_elements
    frame = FramesFactory.getEME2000()
    temp_orbit = KeplerianOrbit(
        a, e, i,
        argp, raan, math.radians(M_deg),
        PositionAngleType.MEAN,
        frame, epoch, Constants.WGS84_EARTH_MU
    )
    return [a, e, i, argp, raan, math.degrees(temp_orbit.getTrueAnomaly())]


# -----------------------------------------------------------------------------
# Tasking cost / coverage
# -----------------------------------------------------------------------------
def pointing_cost(task, eo_tools, r_vec, v_vec, t_datetime):
    """
    Compute angular cost of retargeting to task.
    Lower = easier to point.
    """
    d_brf, offnadir_unbound = eo_tools.point_to_target_unbounded(
        r_vec, v_vec, task["coord"], t_datetime)
    eul_target = eo_tools.att_model.pointing_attitude_brf(d_brf)
    delta = np.abs(np.array(eul_target) - eo_tools.att_model._actor_attitude_deg)
    return np.linalg.norm(delta)


def count_orbits_completed(a_m, sim_duration_seconds):
    """Return (n_orbits_float, n_full_orbits, residual_seconds, period_seconds)."""
    T = compute_orbital_period(a_m)
    n_float = sim_duration_seconds / T
    n_full = int(n_float)
    residual = sim_duration_seconds - n_full * T
    return n_float, n_full, residual, T


def compute_coverage_fraction(fov_polygons_tip, fov_polygons_cue,
                              R_earth, inclination_deg, a_m, sim_duration_seconds):
    """
    Compute coverage metrics from FOV polygons:
      - Total covered area and fraction of Earth
      - Max possible coverage given inclination
      - Per-orbit coverage
      - Efficiency (total and per orbit)
    """
    crs_equal_area = CRS.from_proj4(
        "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs")
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs_equal_area, always_xy=True)

    polys = []
    for fov_list in (fov_polygons_tip + fov_polygons_cue):
        if fov_list is None:
            continue
        lons, lats = fov_list[:, 1], fov_list[:, 0]
        if lons[0] != lons[-1] or lats[0] != lats[-1]:
            lons, lats = np.append(lons, lons[0]), np.append(lats, lats[0])
        x, y = transformer.transform(lons, lats)
        try:
            poly = Polygon(zip(x, y))
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            print("Coverage computation: skip FOV polygon")

    union_poly = unary_union(polys) if polys else None
    area_cov_m2 = union_poly.area if union_poly else 0.0
    area_cov_km2 = area_cov_m2 / 1e6
    area_total_m2 = 4.0 * math.pi * R_earth**2
    area_total_km2 = area_total_m2 / 1e6

    area_mission_fraction_total = math.sin(math.radians(abs(inclination_deg)))
    area_mission_m2 = area_mission_fraction_total * area_total_m2
    area_mission_km2 = area_mission_m2 / 1e6

    frac_total = area_cov_km2 / area_total_km2
    frac_mission = (area_cov_m2 / area_mission_m2) if area_mission_m2 > 0 else 0.0

    T = compute_orbital_period(a_m)
    n_orbits = sim_duration_seconds / T if T > 0 else 0.0
    cov_per_orbit_km2 = area_cov_km2 / n_orbits if n_orbits > 0 else 0.0
    frac_total_per_orbit = cov_per_orbit_km2 / area_total_km2 if n_orbits > 0 else 0.0
    frac_mission_per_orbit = cov_per_orbit_km2 / area_mission_km2 if n_orbits > 0 else 0.0

    return (area_total_km2, area_mission_km2, area_cov_km2,
            cov_per_orbit_km2, area_mission_fraction_total,
            frac_total, frac_mission, frac_total_per_orbit, frac_mission_per_orbit)


# -----------------------------------------------------------------------------
# AI flag helpers
# -----------------------------------------------------------------------------
def is_running_ai(actor, whale, parallel_observation_confirmation):
    return actor.running_ai if not parallel_observation_confirmation else whale.running_ai


def set_running_ai(actor, whale, parallel_observation_confirmation, value: bool):
    if not parallel_observation_confirmation:
        actor.running_ai = value
    else:
        whale.running_ai = value
