from custom_paseos.observation.EarthObservation import EOTools
from datetime import timedelta

import numpy as np
import pykep as pk

from org.orekit.time import AbsoluteDate
from org.orekit.bodies import GeodeticPoint
from org.orekit.frames import FramesFactory

from org.orekit.utils import PVCoordinates, Constants
from org.orekit.frames import FramesFactory
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit

from custom_paseos.utils.point_transformation import Point_Geodetic2ECEF
from custom_paseos.propagation.orekit_propagator import OrekitPropagator

from paseos import ActorBuilder, SpacecraftActor


def init_eo_tools(actors, fov_angle, eul_angles):
    tools = {}
    for actor in actors:
        tools[actor.name] = EOTools(
            local_actor=actor,
            actor_initial_attitude_in_deg=eul_angles,
            actor_FOV_ACT_in_deg=[fov_angle],
            actor_FOV_ALT_in_deg=[fov_angle],
            actor_pointing_vector_body=[0.0, 0.0, 1.0]
        )
    return tools


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

def reset_actor_propagator(actor, current_absdate, current_pykep,
                           satellite_mass, area_s, cr_s, area_d, cd,
                           trajectories=None, n_steps=0, show_orbits=False):

    r_vec, v_vec, _, _ = propagate_actor(actor, current_pykep, trajectories, n_steps, show_orbits)

    pv = PVCoordinates(
        Vector3D(float(r_vec[0]), float(r_vec[1]), float(r_vec[2])),
        Vector3D(float(v_vec[0]), float(v_vec[1]), float(v_vec[2]))
    )
    cart_orbit = CartesianOrbit(pv, FramesFactory.getEME2000(), current_absdate, Constants.WGS84_EARTH_MU)
    kep_orbit = KeplerianOrbit(cart_orbit)

    new_elements = [
        kep_orbit.getA(),
        kep_orbit.getE(),
        kep_orbit.getI(),
        kep_orbit.getRightAscensionOfAscendingNode(),
        kep_orbit.getPerigeeArgument(),
        kep_orbit.getMeanAnomaly()
    ]

    new_propagator = OrekitPropagator(
        orbital_elements=new_elements,
        epoch=current_absdate,
        satellite_mass=satellite_mass,
        area_s=area_s, cr_s=cr_s, area_d=area_d, cd=cd
    )

    def new_orbit(t, p=new_propagator, epoch_ref=current_pykep):
        dt = (t.mjd2000 - epoch_ref.mjd2000) * pk.DAY2SEC
        pv = p.eph(dt).getPVCoordinates()
        return (
            list(pv.getPosition().toArray()),
            list(pv.getVelocity().toArray())
        )

    ActorBuilder.set_custom_orbit(actor, new_orbit, current_pykep)



