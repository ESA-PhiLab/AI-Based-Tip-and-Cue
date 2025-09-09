from custom_paseos.observation.EarthObservation import EOTools
from datetime import timedelta

import numpy as np
import pykep as pk

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


import math
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory

def init_eo_tools(tip_actors, cue_actors, fov_tip, fov_cue, eul_ang_tip_target, eul_ang_cue_target):

    eo_tools_dict = {}

    for actor in tip_actors:
        eo_tools_dict[actor.name] = EOTools(
            local_actor=actor,
            initial_eul_ang_deg=eul_ang_tip_target,
            fov_act_deg=[fov_tip],
            fov_alt_deg=[fov_tip],
        )

    for actor in cue_actors:
        eo_tools_dict[actor.name] = EOTools(
            local_actor=actor,
            initial_eul_ang_deg=eul_ang_cue_target,
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

    # delete tasked target list if out of range
    expired = [idx for idx, w in tasked_targets.items()
               if current_time > w["detection_time"] + timedelta(seconds=timeout)]

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


def log_tip_detection(writer, t_datetime, actor, whale_idx, tgt, r, v, in_footprint):
    writer.writerow([
        t_datetime.isoformat(), actor.name,
        whale_idx, f"{tgt[0]:.4f}", f"{tgt[1]:.4f}", f"{tgt[2]:.1f}",
        f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}",
        f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}",
        in_footprint
    ])


def log_cue_evaluation(writer, t_datetime, actor, whale_idx, tgt, r, v,
                       offnadir_angle_deg, in_view, in_footprint,
                       yaw, pitch, roll):
    writer.writerow([
        t_datetime.isoformat(), actor.name,
        whale_idx, f"{tgt[0]:.4f}", f"{tgt[1]:.4f}", f"{tgt[2]:.1f}",
        f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}",
        f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}",
        f"{offnadir_angle_deg:.4f}", in_view, in_footprint,
        f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}"
    ])


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
    targets: dict of {id: {"lat":.., "lon":.., "alt":..}}
    sun_vec: Sun position vector in ECEF (np.array of shape (3,))
    earth: ReferenceEllipsoid
    Returns: set of ids that are illuminated
    """
    illuminated = set()
    sun_unit = sun_vec / np.linalg.norm(sun_vec)

    for tid, t in targets.items():
        r = Point_Geodetic2ECEF(t["lat"], t["lon"], t["alt"]).flatten()
        r_unit = r / np.linalg.norm(r)

        # If angle between position vector and Sun vector is < 90°
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


