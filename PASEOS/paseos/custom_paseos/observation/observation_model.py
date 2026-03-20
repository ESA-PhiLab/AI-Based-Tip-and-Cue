# eotools.py
# -----------------------------------------------------------------------------
# Geometry & EO utilities (no attitude state here).
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pymap3d as pm
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj
from pyproj import Geod
from loguru import logger
from datetime import datetime, timedelta
from paseos.actors.spacecraft_actor import SpacecraftActor

from ..utils.reference_frame_transformation import (
    LVLH2IRF,
    IRF2LVLH,
    LVLH2BRF_eul,
    BRF2LVLH_eul,
    IRF2BRF_eul,
    BRF2IRF_eul,
)
from ..utils.point_transformation import (
    Point_ECI2Geodetic,
    Point_Geodetic2ECI,
)


class EOTools:
    """
    EO geometry tools: FOV construction, ray/ellipsoid intersection, footprint tests,
    quick-visibility checks and polygon coverage helpers.

    This class holds **no** attitude state. It expects an external attitude model
    to be attached (self.att_model) and will query it when a BRF->IRF transform
    needs the current Euler attitude.
    """

    def __init__(self, local_actor, fov_act_deg=[1.0], fov_alt_deg=[1.0], max_offnadir=None):
        assert isinstance(local_actor, SpacecraftActor), "local_actor must be a SpacecraftActor."
        logger.trace("Initializing EOTools")
        self._actor = local_actor
        self.fov_angles = [float(fov_act_deg[0]), float(fov_alt_deg[0])]
        self.max_offnadir = max_offnadir

        # Attitude is owned by AttitudeModel and attached later
        self.att_model = None  # attach via set_attitude_model()

        # --- Back-compat tasking fields expected by the main loop ---
        self.task_queue = []  # list of pending tasks
        self.current_task = None  # currently active task (or None)
        self.offnadir_unbound_target = None  # last computed unconstrained off-nadir (deg)

        self.slew_stab_time = None
        self.move_set = False
        self.return_to_default_announced = False

        # -------------------------------------------------------------------------
    # Link to attitude model
    # -------------------------------------------------------------------------
    def set_attitude_model(self, att_model):
        """Attach an attitude model exposing euler_deg/euler_rad properties."""
        self.att_model = att_model

    # -------------------------------------------------------------------------
    # FOV construction
    # -------------------------------------------------------------------------
    def get_fov_vectors_in_BRF(self):
        """Return 4 unit vectors (BRF) forming the rectangular pyramid FOV."""
        theta_x = np.deg2rad(self.fov_angles[1])
        theta_y = np.deg2rad(self.fov_angles[0])
        V1 = [-np.tan(theta_x / 2), -np.tan(theta_y / 2), 1]
        V2 = [-np.tan(theta_x / 2),  np.tan(theta_y / 2), 1]
        V3 = [ np.tan(theta_x / 2),  np.tan(theta_y / 2), 1]
        V4 = [ np.tan(theta_x / 2), -np.tan(theta_y / 2), 1]
        return np.array([v / np.linalg.norm(v) for v in (V1, V2, V3, V4)], float)

    # -------------------------------------------------------------------------
    # Intersection with WGS-84 ellipsoid
    # -------------------------------------------------------------------------

    def _find_intersection_in_Geodetic(self, ray_dirs_brf, time, r_eci, v_eci, eul_deg_override=None):
        if self.att_model is None and eul_deg_override is None:
            raise RuntimeError("att_model not attached and no eul_deg_override provided.")

        r = np.asarray(r_eci, float).reshape(3)
        v = np.asarray(v_eci, float).reshape(3)

        eul_deg = np.asarray(eul_deg_override, float).reshape(3) if eul_deg_override is not None else self.att_model._actor_attitude_deg

        a, b = 6378137.0, 6356752.314245

        x_ecef, y_ecef, z_ecef = pm.eci2ecef(r[0], r[1], r[2], time)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef], float)


        d_ecef_list = []
        for ray_brf in np.asarray(ray_dirs_brf, float):
            d_eci = BRF2IRF_eul(ray_brf, r, v, eul_deg)
            d_eci /= np.linalg.norm(d_eci)
            dx, dy, dz = pm.eci2ecef(d_eci[0], d_eci[1], d_eci[2], time)
            d_ecef = np.array([dx, dy, dz], float); d_ecef /= np.linalg.norm(d_ecef)
            d_ecef_list.append(d_ecef)

        intersections = []
        for d_ecef in d_ecef_list:
            dx, dy, dz = d_ecef
            A = (dx*dx)/(a*a) + (dy*dy)/(a*a) + (dz*dz)/(b*b)
            B = 2*((r_ecef[0]*dx)/(a*a) + (r_ecef[1]*dy)/(a*a) + (r_ecef[2]*dz)/(b*b))
            C = (r_ecef[0]**2)/(a*a) + (r_ecef[1]**2)/(a*a) + (r_ecef[2]**2)/(b*b) - 1
            delta = B*B - 4*A*C
            if delta < 0:
                continue
            t_candidates = [(-B + np.sqrt(delta))/(2*A), (-B - np.sqrt(delta))/(2*A)]
            ts = [t for t in t_candidates if t > 0]
            if not ts:
                continue
            p_ecef = r_ecef + min(ts) * d_ecef
            lat, lon, alt = pm.ecef2geodetic(*p_ecef)
            intersections.append([lat, lon, alt])

        if not (len(intersections) == 4 or len(intersections) == 1):
            raise ValueError("Expected 4 or 1 intersection(s).")

        P = np.array(intersections, float).T
        lons = P[1, :]
        max_lon_diff = np.max([abs(l1 - l2) for l1 in lons for l2 in lons])
        if max_lon_diff > 300:
            lats, alts = P[0, :], P[2, :]
            lons_w = [lon - 360 if lon > 0 else lon for lon in lons]
            lons_e = [lon + 360 if lon < 0 else lon for lon in lons]
            fov_w = np.vstack((lats, lons_w, alts))
            fov_e = np.vstack((lats, lons_e, alts))
            P = self.pick_west_or_east(np.hstack((fov_w, fov_e)))

        return P

    # -------------------------------------------------------------------------
    # Footprint tests
    # -------------------------------------------------------------------------
    def check_point_in_footprint(self, point, footprint_latlon):
        lat, lon, _ = point
        lats, lons = footprint_latlon[:, 0], footprint_latlon[:, 1]
        if not (np.min(lats) - 1 <= lat <= np.max(lats) + 1 and np.min(lons) - 1 <= lon <= np.max(lons) + 1):
            return 0
        poly = Polygon([(lo, la) for la, lo in zip(lats, lons)])
        return int(poly.contains(Point(lon, lat)))

    def check_point_in_footprint_constrained(self, point, footprint_latlon, dt_sec, t_limit_sec):
        lat, lon, _ = point
        poly = Polygon([(lo, la) for la, lo in zip(footprint_latlon[:, 0], footprint_latlon[:, 1])])
        return bool(poly.contains(Point(lon, lat)) and (dt_sec <= t_limit_sec))

    # -------------------------------------------------------------------------
    # KML
    # -------------------------------------------------------------------------
    @staticmethod
    def load_kml(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        placemarks = []
        for placemark in root.findall(".//kml:Placemark", ns):
            name = placemark.find("kml:name", ns)
            name = name.text if name is not None else None
            timespan = placemark.find(".//kml:TimeSpan", ns)
            begin_time = timespan.find("kml:begin", ns).text if timespan is not None else None
            end_time   = timespan.find("kml:end", ns).text if timespan is not None else None
            polygon = placemark.find(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
            coords = [tuple(map(float, c.split(","))) for c in polygon.text.strip().split()] if polygon is not None else None
            placemarks.append({"name": name, "begin_time": begin_time, "end_time": end_time, "polygon": coords})
        df = pd.DataFrame(placemarks)
        df["begin_time"] = pd.to_datetime(df["begin_time"])
        df["end_time"]   = pd.to_datetime(df["end_time"])
        return df

    # -------------------------------------------------------------------------
    # Pointing vectors
    # -------------------------------------------------------------------------
    def point_to_target_unbounded(self, r_eci, v_eci, tgt_geodetic, t_datetime, frame: str = "LVLH", eul_deg_override=None):
        """point_to_target_unbounded(r_eci,v_eci,tgt_geodetic,t_datetime,frame="LVLH",eul_deg_override=None) -> tuple[np.ndarray,float]: Return LOS unit vector in requested frame ("LVLH" or "BRF") and off-nadir [deg]."""
        P_tgt_eci = np.array(Point_Geodetic2ECI(*tgt_geodetic, t_datetime)).reshape(3)
        r = np.asarray(r_eci, float).reshape(3)
        v = np.asarray(v_eci, float).reshape(3)

        los_eci = P_tgt_eci - r
        n = float(np.linalg.norm(los_eci))
        if n <= 0.0:
            los_lvlh = np.array([0.0, 0.0, 1.0], float)
        else:
            los_eci /= n
            los_lvlh = IRF2LVLH(los_eci, r, v)
            los_lvlh /= float(np.linalg.norm(los_lvlh))

        offnadir = float(np.degrees(np.arccos(np.clip(np.dot([0.0, 0.0, 1.0], los_lvlh), -1.0, 1.0))))

        frame_u = (frame or "LVLH").upper()
        if frame_u == "LVLH":
            return los_lvlh, offnadir

        if frame_u != "BRF":
            raise ValueError(f"frame must be 'LVLH' or 'BRF', got {frame!r}")

        if self.att_model is None and eul_deg_override is None:
            raise RuntimeError("att_model not attached and no eul_deg_override provided.")

        eul_deg = np.asarray(eul_deg_override, float).reshape(3) if eul_deg_override is not None else np.asarray(self.att_model._actor_attitude_deg, float).reshape(3)

        los_brf = LVLH2BRF_eul(los_lvlh, eul_deg)

        los_brf = np.asarray(los_brf, float).reshape(3)
        los_brf /= float(np.linalg.norm(los_brf))
        return los_brf, offnadir

    def point_to_target_bounded(self, r_eci, v_eci, target_geodetic, t_datetime, offnadir_max=None, mode="max", dt_step_coarse=1.0, dt_step_fine=0.5, dt_step_ultrafine=0.1, dt_max=600.0):
        """point_to_target_bounded(r_eci,v_eci,target_geodetic,t_datetime,offnadir_max=None,mode="max",dt_step_coarse=1.0,dt_step_fine=0.5,dt_step_ultrafine=0.1,dt_max=600.0) -> tuple[np.ndarray,float,float,float|None]: Return LVLH LOS unit vector, bounded off-nadir, unbounded off-nadir, and chosen time_to_sight (s)."""
        pointing_vec_lvlh_target, offnadir_unbound = self.point_to_target_unbounded(
            r_eci, v_eci, target_geodetic, t_datetime, frame="LVLH"
        )

        if offnadir_max is not None and offnadir_unbound > float(offnadir_max) + 1e-3:
            pointing_vec_lvlh_target, offnadir_deg_target, time_to_sight = self.set_max_offnadir(
                offnadir_max=float(offnadir_max),
                offnadir_unbound=float(offnadir_unbound),
                pointing_vec_lvlh_target=pointing_vec_lvlh_target,
                r_eci=r_eci,
                v_eci=v_eci,
                target_geodetic=target_geodetic,
                t_datetime=t_datetime,
                dt_step_coarse=float(dt_step_coarse),
                dt_step_fine=float(dt_step_fine),
                dt_step_ultrafine=float(dt_step_ultrafine),
                dt_max=float(dt_max),
                mode=str(mode),
            )
        else:
            offnadir_deg_target = float(offnadir_unbound)
            time_to_sight = 0.0

        return pointing_vec_lvlh_target, offnadir_deg_target, offnadir_unbound, time_to_sight

    def compute_optimal_future_attitude(self,
                                        r_eci,
                                        v_eci,
                                        target_geodetic,
                                        t_datetime,
                                        omega_max_rad,
                                        alpha_max_rad,
                                        zeta,
                                        wn_rad,
                                        offnadir_max,
                                        offnadir_margin=0.0,
                                        dt_step_coarse=5.0,
                                        dt_step_fine=1.0,
                                        dt_max=600.0,
                                        mode="per_axis"):
        """compute_optimal_future_attitude(...) -> tuple[list[float]|None,float|None,float,float|None,np.ndarray|None]: Return earliest slew-feasible future attitude using future LOS geometry only."""
        if self.att_model is None:
            raise RuntimeError("att_model not attached.")

        r0 = np.asarray(r_eci, float).reshape(3)
        v0 = np.asarray(v_eci, float).reshape(3)

        current_eul = np.asarray(self.att_model._actor_attitude_deg, float).reshape(3)
        current_w = np.asarray(self.att_model._actor_angular_velocity, float).reshape(3)
        current_a = np.asarray(self.att_model._actor_angular_acceleration, float).reshape(3)

        _, offnadir_unbound_now = self.point_to_target_unbounded(
            r0, v0, target_geodetic, t_datetime, frame="LVLH"
        )

        dt_step_coarse = float(dt_step_coarse)
        dt_step_fine = min(float(dt_step_fine), dt_step_coarse)
        dt_max = float(dt_max)
        off_limit = float(offnadir_max) + float(offnadir_margin)

        best_solution = None

        dt = 0.0
        while dt <= dt_max + 1e-9:
            t_future = t_datetime + timedelta(seconds=float(dt))
            r_future, v_future = self._kepler_propagate_universal(r0, v0, float(dt))
            r_future = np.asarray(r_future, float).reshape(3)
            v_future = np.asarray(v_future, float).reshape(3)

            pv_future, off_future = self.point_to_target_unbounded(
                r_future, v_future, target_geodetic, t_future, frame="LVLH"
            )
            off_future = float(off_future)

            if off_future <= off_limit + 1e-9:
                target_eul_deg = self.att_model.pointing_attitude_lvlh(pv_future)

                t_need, _, _ = self.att_model.get_pointing_stabilization_time(
                    current_eul=current_eul,
                    target_eul=target_eul_deg,
                    omega_max_rad=omega_max_rad,
                    alpha_max_rad=alpha_max_rad,
                    zeta=zeta,
                    wn_rad=wn_rad,
                    mode=mode,
                    current_w_rad=current_w,
                    current_a_rad=current_a
                )

                if float(dt) >= float(t_need) - 1e-9:
                    pv_future = np.asarray(pv_future, float).reshape(3)
                    pv_future /= float(np.linalg.norm(pv_future))
                    return target_eul_deg, off_future, float(offnadir_unbound_now), float(dt), pv_future

                if best_solution is None:
                    pv_future = np.asarray(pv_future, float).reshape(3)
                    pv_future /= float(np.linalg.norm(pv_future))
                    best_solution = (
                        target_eul_deg,
                        off_future,
                        float(offnadir_unbound_now),
                        float(dt),
                        pv_future,
                        float(t_need),
                    )

            if off_future > offnadir_max + 5.0:
                dt += dt_step_coarse
            else:
                dt += dt_step_fine

        return None, None, float(offnadir_unbound_now), None, None







    # -------------------------------------------------------------------------
    # Visibility
    # -------------------------------------------------------------------------
    def is_in_sight(self, tgt_geodetic, r_eci, v_eci, time, el_min_deg):
        lat, lon, alt = tgt_geodetic
        xs, ys, zs = pm.eci2ecef(r_eci[0], r_eci[1], r_eci[2], time)
        e, n, u = pm.ecef2enu(xs, ys, zs, lat, lon, alt)
        _, el, _ = pm.enu2aer(e, n, u)
        return el > el_min_deg

    def will_be_visible_within(self, target_geodetic, r_eci, v_eci, t_datetime,
                               t_ahead, el_min_deg=10.0, step=600.0):
        """
        Returns (visible, time_to_visibility).

        - visible: True if satellite will see target within t_ahead
        - time_to_visibility: 0.0 if already visible, >0 if visible later, None if never visible
        """
        # 1) Immediate visibility check
        if self.is_in_sight(target_geodetic, r_eci, v_eci, t_datetime, el_min_deg):
            return True, 0.0

        # 2) Fast inclusion marker (cheap yes/no check)
        if self.fast_inclusion_marker(target_geodetic, r_eci, v_eci,
                                      t_datetime, t_ahead, el_min_deg, confirm=True):
            pass  # fall through to coarse scan

        # 3) Coarse stepping propagation
        t = 0.0
        r = np.asarray(r_eci, float).flatten()
        v = np.asarray(v_eci, float).flatten()

        while t < t_ahead:
            h = min(step, t_ahead - t)
            r, v = self._kepler_propagate_universal(r, v, h)

            r = np.asarray(r, float).reshape(3)
            v = np.asarray(v, float).reshape(3)

            t += h
            t_step = t_datetime + timedelta(seconds=float(t))

            if self.is_in_sight(target_geodetic, r, v, t_step, el_min_deg):
                return True, t  # visible in t seconds

        return False, None

    @staticmethod
    def distance_to_target(r_eci, target_geodetic, t_datetime):
        r = np.asarray(r_eci, float).reshape(3)
        tgt_vec = np.array(Point_Geodetic2ECI(*target_geodetic, t_datetime)).reshape(3)
        return np.linalg.norm(tgt_vec - r)

    def is_moving_towards_target(self, r_eci, v_eci, target_geodetic, t_datetime, dt_check: float = 10.0):
        """
        Propagate satellite forward and compare distances.
        Uses larger dt_check to capture if it is past closest approach.
        """
        distance_now = self.distance_to_target(r_eci, target_geodetic, t_datetime)

        # Propagate forward
        r_future, v_future = self._kepler_propagate_universal(r_eci, v_eci, dt_check)
        r_future = np.asarray(r_future).reshape(3,1)
        t_future = t_datetime + timedelta(seconds=dt_check)

        distance_future = self.distance_to_target(r_future, target_geodetic, t_future)

        range_rate = (distance_future - distance_now) / dt_check
        return (range_rate < 0.0), range_rate

    # --- eotools.py: REPLACE compute_viewing_time COMPLETELY ---

    def compute_viewing_time(self, r_eci, v_eci, target_geodetic, t_datetime, offnadir_max, offnadir_margin=0.0, dt_step_coarse=1.0, dt_step_fine=0.5, dt_step_ultrafine=0.1, dt_max=600.0):
        """compute_viewing_time(r_eci,v_eci,target_geodetic,t_datetime,offnadir_max,offnadir_margin=0.0,dt_step_coarse=1.0,dt_step_fine=0.5,dt_step_ultrafine=0.1,dt_max=600.0) -> float: Remaining time (s) target stays within strict off-nadir in LVLH (+Z nadir)."""
        offnadir_max = float(offnadir_max)
        offnadir_margin = float(offnadir_margin)
        dt_step_coarse = float(dt_step_coarse)
        dt_step_fine = min(float(dt_step_fine), dt_step_coarse)
        dt_step_ultrafine = min(float(dt_step_ultrafine), dt_step_coarse)
        dt_max = float(dt_max)

        # off-nadir is defined in LVLH; do NOT use frame="BRF" here
        _, offnadir_now = self.point_to_target_unbounded(r_eci, v_eci, target_geodetic, t_datetime, frame="LVLH")

        if float(offnadir_now) > offnadir_max + offnadir_margin + 1e-9:
            return 0.0

        inside_strict = float(offnadir_now) <= offnadir_max + 1e-9
        last_good = 0.0
        dt = 0.0

        r0 = np.asarray(r_eci, float).reshape(3)
        v0 = np.asarray(v_eci, float).reshape(3)

        while dt <= dt_max + 1e-9:
            t_future = t_datetime + timedelta(seconds=float(dt))
            r_future, v_future = self._kepler_propagate_universal(r0, v0, float(dt))
            _, off_future = self.point_to_target_unbounded(r_future, v_future, target_geodetic, t_future, frame="LVLH")
            off_future = float(off_future)

            if off_future <= offnadir_max + 1e-9:
                inside_strict = True
                last_good = dt
                if off_future < offnadir_max - 1.0:
                    dt += dt_step_coarse
                elif off_future < offnadir_max - 0.3:
                    dt += dt_step_fine
                else:
                    dt += dt_step_ultrafine

            elif off_future <= offnadir_max + offnadir_margin + 1e-9:
                if not inside_strict:
                    dt += dt_step_ultrafine
                else:
                    return float(last_good)

            else:
                return float(last_good)

        return float(last_good)

    # -------------------------------------------------------------------------
    # Footprint helpers
    # -------------------------------------------------------------------------
    def get_center_vector_in_BRF(self):
        return np.array([[0.0, 0.0, 1.0]], float)

    def get_FovPoints(self, r_vec, v_vec, t_datetime, eul_deg_override=None):
        """get_FovPoints(r_vec,v_vec,t_datetime,eul_deg_override=None) -> np.ndarray: Footprint lat/lon points using current or overridden Euler."""
        rays_brf = self.get_fov_vectors_in_BRF()
        P = self._find_intersection_in_Geodetic(rays_brf, t_datetime, r_vec, v_vec, eul_deg_override=eul_deg_override)
        return P[:2, :].T

    def get_CenterRay_Intersection(self, r_vec, v_vec, t_datetime):
        """Intersection of the nadir ray with the reference ellipsoid (sub-satellite point)."""
        # Sub-satellite point is simply geodetic projection of the spacecraft position.
        lat, lon, _alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
        return np.array([[lat], [lon], [0.0]], float)

    def get_CenterRay_Intersection_Attitude(self, r_vec, v_vec, t_datetime):
        if self.att_model is None:
            raise RuntimeError("att_model not attached.")
        return self._find_intersection_in_Geodetic([[0, 0, 1]], t_datetime, r_vec, v_vec)

    # -------------------------------------------------------------------------
    # Dateline
    # -------------------------------------------------------------------------
    def pick_west_or_east(self, P):
        lats, lons, alts = P
        west, east = P[:, lons < 0], P[:, lons > 0]
        if west.shape[1] != 4 or east.shape[1] != 4:
            raise ValueError("Expected two groups of 4 points.")
        west_center = np.mean(west[1]); east_center = np.mean((east[1] + 180) % 360 - 180)
        return east if abs(east_center) > abs(west_center) else west

    def fast_inclusion_marker(self, target_geodetic, r_eci, v_eci, t_datetime, t_ahead,
                              el_min_deg=0.0, confirm=True):
        """Very fast yes/no: could the target be visible within [time, time+t_ahead]?"""
        if self.is_in_sight(target_geodetic, r_eci, v_eci, t_datetime, el_min_deg):
            return True

        r_tgt, v_tgt = self._target_state_eci(target_geodetic, t_datetime)
        dr = np.asarray(r_eci, float).flatten() - r_tgt
        v_rel = np.asarray(v_eci, float).flatten() - v_tgt

        drn = float(np.linalg.norm(dr))
        if drn < 1.0:
            return False

        rr = float(np.dot(dr / drn, v_rel))  # range-rate
        if rr >= 0.0:
            return False

        vrel2 = float(np.dot(v_rel, v_rel))
        if vrel2 < 1e-12:
            return False

        t_star = -float(np.dot(dr, v_rel)) / vrel2
        if not (0.0 < t_star <= t_ahead):
            return False

        if not confirm:
            return True

        r_star = np.asarray(r_eci, float).flatten() + np.asarray(v_eci, float).flatten() * t_star
        t_future = t_datetime + timedelta(seconds=float(t_star))
        return self.is_in_sight(target_geodetic, r_star, v_eci, t_future, el_min_deg)

    @staticmethod
    def _kepler_propagate_universal(r0, v0, t_step, mu=3.986004418e14):
        """Propagate (r0,v0) by t_step [s] under 2-body using universal variables; returns (r,v) as 3x1 arrays."""
        r0 = np.asarray(r0, float).reshape(3)
        v0 = np.asarray(v0, float).reshape(3)
        if t_step == 0.0:
            return r0.reshape(3, 1), v0.reshape(3, 1)

        r0n = float(np.linalg.norm(r0))
        v0n = float(np.linalg.norm(v0))
        vr0 = float(np.dot(r0, v0) / r0n)
        alpha = 2.0 / r0n - (v0n * v0n) / mu

        def C(z):
            if z > 0:  s = np.sqrt(z);  return (1 - np.cos(s)) / z
            if z < 0:  s = np.sqrt(-z); return (np.cosh(s) - 1) / (-z)
            return 0.5

        def S(z):
            if z > 0:  s = np.sqrt(z);  return (s - np.sin(s)) / (s ** 3)
            if z < 0:  s = np.sqrt(-z); return (np.sinh(s) - s) / (s ** 3)
            return 1.0 / 6.0

        root_mu = np.sqrt(mu)
        chi = (root_mu * abs(alpha) * t_step) if abs(alpha) > 1e-12 else (root_mu * t_step / r0n)

        for _ in range(8):
            z = alpha * chi * chi
            Cz = C(z);
            Sz = S(z)
            f = (r0n * vr0 / root_mu) * (chi * chi) * Cz + (1.0 - alpha * r0n) * (chi ** 3) * Sz + r0n * chi - root_mu * t_step
            fp = (r0n * vr0 / root_mu) * chi * (1.0 - z * Sz) + (1.0 - alpha * r0n) * (chi * chi) * Cz + r0n
            dchi = f / fp
            chi -= dchi
            if abs(dchi) <= 1e-12:
                break

        z = alpha * chi * chi
        Cz = C(z);
        Sz = S(z)
        f = 1.0 - (chi * chi / r0n) * Cz
        g = t_step - (chi ** 3 / root_mu) * Sz
        r = f * r0 + g * v0
        rn = float(np.linalg.norm(r))
        fdot = (root_mu / (r0n * rn)) * (z * Sz - 1.0) * chi
        gdot = 1.0 - (chi * chi / rn) * Cz
        v = fdot * r0 + gdot * v0
        return r.reshape(3, 1), v.reshape(3, 1)

    def _target_state_eci(self, target_geodetic, t_datetime):
        """Target position and inertial velocity in ECI (from Earth rotation)."""
        lat, lon, alt = target_geodetic
        x, y, z = pm.geodetic2ecef(lat, lon, alt)
        rx, ry, rz = pm.ecef2eci(x, y, z, t_datetime)
        r_tgt_eci = np.array([rx, ry, rz], dtype=float).flatten()
        omega_earth = np.array([0.0, 0.0, 7.2921150e-5])  # rad/s
        v_tgt_eci = np.cross(omega_earth, r_tgt_eci).flatten()
        return r_tgt_eci, v_tgt_eci

    # --- eotools.py: REPLACE set_max_offnadir COMPLETELY ---

    def set_max_offnadir(self, offnadir_max, offnadir_unbound, pointing_vec_lvlh_target, r_eci, v_eci, target_geodetic, t_datetime, dt_step_coarse=1.0, dt_step_fine=0.5, dt_step_ultrafine=0.1, dt_max=600.0, mode="max"):
        """set_max_offnadir(offnadir_max,offnadir_unbound,pointing_vec_lvlh_target,r_eci,v_eci,target_geodetic,t_datetime,dt_step_coarse=1.0,dt_step_fine=0.5,dt_step_ultrafine=0.1,dt_max=600.0,mode="max") -> tuple[np.ndarray,float,float|None]: Enforce off-nadir limit in LVLH; returns (LVLH LOS vec, offnadir_deg, chosen_time_s)."""
        offnadir_max = float(offnadir_max)
        offnadir_unbound = float(offnadir_unbound)

        pv = np.asarray(pointing_vec_lvlh_target, float).reshape(3)
        n = float(np.linalg.norm(pv))
        if n <= 0.0:
            pv = np.array([0.0, 0.0, 1.0], float)
        else:
            pv /= n

        if offnadir_unbound <= offnadir_max + 1e-3:
            return pv, offnadir_unbound, 0.0

        mode = str(mode).lower()

        if mode == "cap":
            boresight_lvlh = np.array([0.0, 0.0, 1.0], float)

            rot_axis = np.cross(boresight_lvlh, pv)
            axn = float(np.linalg.norm(rot_axis))
            if axn < 1e-12:
                return boresight_lvlh.copy(), offnadir_max, 0.0
            rot_axis /= axn

            ang = np.deg2rad(offnadir_max)
            K = np.array([
                [0.0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0.0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0.0]
            ], float)
            R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)

            new_vec = R @ boresight_lvlh
            new_vec /= float(np.linalg.norm(new_vec))
            return new_vec, offnadir_max, 0.0

        if mode == "max":
            dt_step_coarse = float(dt_step_coarse)
            dt_step_fine = min(float(dt_step_fine), dt_step_coarse)
            dt_step_ultrafine = min(float(dt_step_ultrafine), dt_step_coarse)
            dt_max = float(dt_max)

            dt = 0.0
            r0 = np.asarray(r_eci, float).reshape(3)
            v0 = np.asarray(v_eci, float).reshape(3)

            while dt <= dt_max + 1e-9:
                t_future = t_datetime + timedelta(seconds=float(dt))
                r_future, v_future = self._kepler_propagate_universal(r0, v0, float(dt))

                pv_future, off_future = self.point_to_target_unbounded(
                    r_future, v_future, target_geodetic, t_future, frame="LVLH"
                )

                if float(off_future) <= offnadir_max + 1e-9:
                    pv_future = np.asarray(pv_future, float).reshape(3)
                    pv_future /= float(np.linalg.norm(pv_future))
                    return pv_future, float(off_future), float(dt)

                # adaptive stepping based on how far outside the bound we are
                if float(off_future) > offnadir_max + 2.0:
                    step = dt_step_coarse
                elif float(off_future) > offnadir_max + 0.5:
                    step = dt_step_fine
                else:
                    step = dt_step_ultrafine

                dt += float(step)

            return None, None, None

        raise ValueError("mode must be 'cap' or 'max'")


