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

from custom_paseos.utils.reference_frame_transformation import (
    LVLH2IRF,
    IRF2LVLH,
    LVLH2BRF_eul,
    BRF2LVLH_eul,
    IRF2BRF_eul,
    BRF2IRF_eul,
)
from custom_paseos.utils.point_transformation import (
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

        self.t_eul_commanded = None
        self.delay_slew_stab = None

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

    def _find_intersection_in_Geodetic(self, ray_dirs_brf, time, r_eci, v_eci):
        if self.att_model is None:
            raise RuntimeError("EOTools._find_intersection_in_Geodetic: att_model not attached.")

        r = np.asarray(r_eci, float).reshape(3)
        v = np.asarray(v_eci, float).reshape(3)

        a, b = 6378137.0, 6356752.314245

        x_ecef, y_ecef, z_ecef = pm.eci2ecef(r[0], r[1], r[2], time)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef], float)

        eul_deg = self.att_model._actor_attitude_deg

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
    def point_to_target_unbounded(self, r_eci, v_eci, tgt_geodetic, t_datetime):
        P_tgt_eci = Point_Geodetic2ECI(*tgt_geodetic, t_datetime)
        vec_eci = P_tgt_eci.flatten() - np.asarray(r_eci, float).reshape(3)
        d_brf = IRF2BRF_eul(vec_eci, r_eci, v_eci, [0.0, 0.0, 0.0])
        d_brf /= np.linalg.norm(d_brf)
        offnadir = np.degrees(np.arccos(np.clip(np.dot([0, 0, 1], d_brf), -1, 1)))
        return d_brf, float(offnadir)


    def point_to_target_bounded(self, r_eci, v_eci, target_geodetic, t_datetime,
                                offnadir_max=None, mode='cap'):
        pointing_vec_brf_target, offnadir_unbound = self.point_to_target_unbounded(
            r_eci, v_eci, target_geodetic, t_datetime
        )

        if offnadir_max is not None and offnadir_unbound > offnadir_max:
            pointing_vec_brf_target, offnadir_deg_target, time_to_sight = self.set_max_offnadir(
                offnadir_max, offnadir_unbound, pointing_vec_brf_target,
                r_eci, v_eci, target_geodetic, t_datetime, mode=mode
            )
        else:
            offnadir_deg_target = offnadir_unbound
            time_to_sight = 0.0

        return pointing_vec_brf_target, offnadir_deg_target, offnadir_unbound, time_to_sight

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
            t += h
            t_step = t_datetime + timedelta(seconds=float(t))

            if self.is_in_sight(target_geodetic, r, v, t_step, el_min_deg):
                return True, t  # visible in t seconds

        return False, None

    # -------------------------------------------------------------------------
    # Footprint helpers
    # -------------------------------------------------------------------------
    def get_center_vector_in_BRF(self):
        return np.array([[0.0, 0.0, 1.0]], float)

    def get_FovPoints(self, r_vec, v_vec, t_datetime):
        rays_brf = self.get_fov_vectors_in_BRF()
        P = self._find_intersection_in_Geodetic(rays_brf, t_datetime, r_vec, v_vec)
        return P[:2, :].T

    def get_CenterRay_Intersection(self, r_vec, v_vec, t_datetime):
        sat_lat, sat_lon, _ = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
        a, b = 6378137.0, 6356752.314245
        x0, y0, z0 = pm.geodetic2ecef(sat_lat, sat_lon, 0.0)
        n_ecef = np.array([x0/(a*a), y0/(a*a), z0/(b*b)], float)
        n_ecef /= np.linalg.norm(n_ecef)
        nx, ny, nz = pm.ecef2eci(-n_ecef[0], -n_ecef[1], -n_ecef[2], t_datetime)
        ray_brf = IRF2LVLH([nx, ny, nz], r_vec, v_vec)
        return self._find_intersection_in_Geodetic([ray_brf], t_datetime, r_vec, v_vec)

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



    def set_max_offnadir(self,
                         offnadir_max: float,
                         offnadir_deg_target: float,
                         pointing_vec_brf_target,
                         r_eci, v_eci, target_geodetic, t_datetime,
                         dt_step_coarse: float = 10.0,
                         dt_step_fine: float = 2.0,
                         dt_step_ultrafine: float = 0.2,
                         dt_max: float = 600.0,
                         mode: str = 'cap'):
        """Predictive off-nadir limiter with adaptive step size."""
        if offnadir_deg_target <= offnadir_max:
            return pointing_vec_brf_target, offnadir_deg_target, None

        if mode == 'cap':
            boresight_brf = np.array([0.0, 0.0, 1.0])
            rot_axis = np.cross(boresight_brf, pointing_vec_brf_target)
            n = np.linalg.norm(rot_axis)
            rot_axis = np.array([1.0, 0.0, 0.0]) if n < 1e-8 else rot_axis / n

            ang = np.deg2rad(offnadir_max)
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                          [rot_axis[2], 0, -rot_axis[0]],
                          [-rot_axis[1], rot_axis[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
            new_vec = R @ boresight_brf
            new_vec /= np.linalg.norm(new_vec)
            if np.dot(new_vec, pointing_vec_brf_target) < 0:
                new_vec = -new_vec
            return new_vec, offnadir_max, None

        if mode == 'max':
            dt = 0.0
            while dt <= dt_max:
                t_future = t_datetime + timedelta(seconds=dt)
                r_future, v_future = self._kepler_propagate_universal(r_eci, v_eci, dt)
                pointing_vec_brf_future, offnadir_future = self.point_to_target_unbounded(
                    r_future, v_future, target_geodetic, t_future
                )
                if offnadir_future <= offnadir_max:
                    return pointing_vec_brf_future, offnadir_future, dt
                if offnadir_future > offnadir_max + 2.0:
                    step = dt_step_coarse
                elif offnadir_future > offnadir_max + 0.5:
                    step = dt_step_fine
                else:
                    step = dt_step_ultrafine
                dt += step

            return pointing_vec_brf_future, offnadir_future, dt

