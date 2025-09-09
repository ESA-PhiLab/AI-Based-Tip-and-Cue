# Libraries Importing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.time import Time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import contextily as ctx
import time
import pdb
import pandas as pd
import pymap3d as pm
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
from loguru import logger
from paseos.actors.spacecraft_actor import SpacecraftActor
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import pyproj
from pyproj import Geod
import warnings
from custom_paseos.utils.constants import R_p, R_e
from datetime import datetime
from PIL import Image
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, Point
from pyproj import Geod
import pandas as pd
import pyproj
import warnings
import pdb
import os

from custom_paseos.utils.reference_frame_transformation import (
    LVLH2IRF,
    RotMat_by_quat,
    IRF2LVLH,
    LVLH2BRF_eul,
    BRF2LVLH_eul,
    BRF2IRF_eul,
    IRF2BRF_eul,
    RotMat_LVLH_to_BRF_by_eul,
    rotation_matrix_to_ypr
)

from custom_paseos.utils.point_transformation import (
    Point_ECI2Geodetic, Point_Geodetic2ECI
)


class EOTools:
    # Spacecraft_actor.
    _actor = None
    # Actor attitude in deg
    eul_ang_deg = None

    fov_angles = None

    phi_deg = None      # Rotation angle about own FOV axis (pointing_vec_brf)

    busy = False

    """
    This class is provided with all the functions needed to perform the dedicated Earth-Observation activities
    """

    def __init__(
            self,
            local_actor,
            initial_eul_ang_deg: list[float] = [0.0, 0.0, 0.0],
            fov_act_deg: list[float] = [1.0],
            fov_alt_deg: list[float] = [1.0],
    ):

        assert isinstance(local_actor, SpacecraftActor), (
            "local_actor must be a " "SpacecraftActor" "."
        )

        logger.trace("Initializing EOTools Function")
        self._actor = local_actor

        # Convert attitude in np.ndarray
        self.eul_ang_deg = np.array(initial_eul_ang_deg)
        # Convert pointing vector in np.ndarray
        # Creation of the FOV
        self.fov_angles = [fov_act_deg[0], fov_alt_deg[0]]

        self.phi_deg = 0.0

    # Module to create a Piramidal 3D FOV in the BRF. Thi module allows to create the piramidal shape of a rectangular footprint in the BRF.
    # Note that this module allows to have the directions of the 3D prismatic FOV lines in the BRF, to get the intersection with a geoidetic reference model.
    def get_fov_vectors_in_BRF(self):
        theta_x = np.deg2rad(self.fov_angles[1])
        theta_y = np.deg2rad(self.fov_angles[0])
        V1 = [-np.tan(theta_x / 2), -np.tan(theta_y / 2), 1]  # Top-right
        d1 = V1 / np.linalg.norm(V1)
        V2 = [-np.tan(theta_x / 2), np.tan(theta_y / 2), 1]  # Top-left
        d2 = V2 / np.linalg.norm(V2)
        V3 = [np.tan(theta_x / 2), np.tan(theta_y / 2), 1]  # Bottom-left
        d3 = V3 / np.linalg.norm(V3)
        V4 = [np.tan(theta_x / 2), -np.tan(theta_y / 2), 1]  # Bottom-right
        d4 = V4 / np.linalg.norm(V4)

        return np.array([d1, d2, d3, d4])

    # Module to find the edge points of the footprint created by a piramidal 3D FOV with a Geodetic Reference Model (WGS84). The Geoid Model is easily represented by:
    # R_e (Equatorial Radius) = 6378137 m;
    # R_p (Polar Radius) = 6356752 m;

    def _find_intersection_in_Geodetic(self, ray_direction, time, r, v):
        """
        Intersect BRF rays with the WGS-84 ellipsoid.
        - Transform BRF rays -> ECI (via attitude) -> ECEF (at 'time').
        - Solve quadratic in ECEF against (x^2/a^2 + y^2/a^2 + z^2/b^2 = 1).
        - Convert intersection ECEF -> geodetic (lat, lon, alt).
        """

        # Ensure 1D vectors
        r = np.asarray(r, dtype=float).flatten()
        v = np.asarray(v, dtype=float).flatten()

        # WGS-84 semi-axes (exact)
        a = 6378137.0
        b = 6356752.314245

        # Satellite position in ECEF at this epoch
        x_ecef, y_ecef, z_ecef = pm.eci2ecef(r[0], r[1], r[2], time)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef], dtype=float)

        # Build ray directions in ECEF
        d_ecef_list = []
        for ray in ray_direction:
            # BRF -> IRF (ECI)
            d_eci = BRF2IRF_eul(np.asarray(ray, dtype=float).flatten(), r, v, self.eul_ang_deg)
            d_eci = d_eci / np.linalg.norm(d_eci)

            # ECI -> ECEF (pure rotation at 'time')
            dx_ecef, dy_ecef, dz_ecef = pm.eci2ecef(d_eci[0], d_eci[1], d_eci[2], time)
            d_ecef = np.array([dx_ecef, dy_ecef, dz_ecef], dtype=float)
            d_ecef /= np.linalg.norm(d_ecef)
            d_ecef_list.append(d_ecef)

        intersections = []
        for d_ecef in d_ecef_list:
            dx, dy, dz = d_ecef

            # Quadratic against ellipsoid in ECEF
            A = (dx * dx) / (a * a) + (dy * dy) / (a * a) + (dz * dz) / (b * b)
            B = 2.0 * ((r_ecef[0] * dx) / (a * a) + (r_ecef[1] * dy) / (a * a) + (r_ecef[2] * dz) / (b * b))
            C = (r_ecef[0] * r_ecef[0]) / (a * a) + (r_ecef[1] * r_ecef[1]) / (a * a) + (r_ecef[2] * r_ecef[2]) / (
                        b * b) - 1.0

            delta = B * B - 4 * A * C
            if delta < 0:
                continue

            sqrt_delta = np.sqrt(delta)
            t1 = (-B + sqrt_delta) / (2 * A)
            t2 = (-B - sqrt_delta) / (2 * A)

            # Nearest positive root
            t_candidates = [t for t in (t1, t2) if t > 0]
            if not t_candidates:
                continue
            t = min(t_candidates)

            # Intersection point in ECEF
            p_ecef = r_ecef + t * d_ecef

            # ECEF -> geodetic (WGS-84)
            lat, lon, alt = pm.ecef2geodetic(p_ecef[0], p_ecef[1], p_ecef[2])

            intersections.append([lat, lon, alt])

        if not (len(intersections) == 4 or len(intersections) == 1):
            raise ValueError("Not enough intersections point")

        # Stack to 3xN
        intersections_matrix = np.array(intersections, dtype=float).T  # [lat; lon; alt]

        # Handle longitude wrap-around if FOV spans the dateline
        lats = intersections_matrix[0, :]
        lons = intersections_matrix[1, :]
        alts = intersections_matrix[2, :]

        max_lon_diff = np.max([abs(l1 - l2) for l1 in lons for l2 in lons])
        if max_lon_diff > 300:
            lons_fov1 = np.array([lon - 360 if lon > 0 else lon for lon in lons])
            fov1 = np.vstack((lats, lons_fov1, alts))
            lons_fov2 = np.array([lon + 360 if lon < 0 else lon for lon in lons])
            fov2 = np.vstack((lats, lons_fov2, alts))
            intersections_matrix = np.hstack((fov1, fov2))
            # print("\t Crossed longitude border, picked largest FOV")
            intersections_matrix = self.pick_west_or_east(intersections_matrix)

        return intersections_matrix

    def check_point_in_footprint(self, point, footprint):
        # Check if the point is in the footprint

        lat, lon, _ = point

        lats = footprint[:, 0]
        lons = footprint[:, 1]

        # =================== ADDED =============================

        # Cheap reject via bbox (optionally expanded by `margin`)
        lat_min, lat_max = np.min(lats), np.max(lats)
        lon_min, lon_max = np.min(lons), np.max(lons)

        # margin = 0.1 * ( (lat_max - lat_min) + (lon_max - lon_min))
        margin = 1.0 # deg

        if lon < lon_min - margin or lon > lon_max + margin or lat < lat_min - margin or lat > lat_max + margin:

            return 0

        # ==================== ADDED ================================

        # if not, do more expensive reject
        polygon_coords = [(lon_i, lat_i) for lat_i, lon_i in zip(lats, lons)]

        # Create the polygon
        polygon = Polygon(polygon_coords)
        pt = Point(lon, lat)

        return 1 if polygon.contains(pt) else 0 ## Changed 2 to 0

    def check_point_in_footprint_constrained(self, point, footprint, delta_time_sec, time_limit_sec):

        lat, lon, _ = point

        # extract the lats and lons from the matrix
        lats = footprint[:, 0]
        lons = footprint[:, 1]
        polygon_coords = [(lon_i, lat_i) for lat_i, lon_i in zip(lats, lons)]

        # Create the polygon
        polygon = Polygon(polygon_coords)
        pt = Point(lon, lat)

        in_footprint = polygon.contains(pt)
        within_time_limit = delta_time_sec <= time_limit_sec

        return True if in_footprint and within_time_limit else False

    def load_kml(self, file_path):

        tree = ET.parse(file_path)
        root = tree.getroot()

        # Namespace KML
        ns = {'kml': 'http://www.opengis.net/kml/2.2', 'ns0': 'http://www.opengis.net/kml/2.2'}

        placemarks = []
        for placemark in root.findall(".//kml:Placemark", ns):
            name = placemark.find("kml:name", ns).text if placemark.find("kml:name", ns) is not None else None

            # Estracts Begin Time and End Time
            timespan = placemark.find(".//kml:TimeSpan", ns)
            begin_time = timespan.find("kml:begin", ns).text if timespan is not None and timespan.find("kml:begin",
                                                                                                       ns) is not None else None
            end_time = timespan.find("kml:end", ns).text if timespan is not None and timespan.find("kml:end",
                                                                                                   ns) is not None else None

            polygon = placemark.find(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)

            if polygon is not None:
                coords_text = polygon.text.strip()
                coords_list = [tuple(map(float, coord.split(","))) for coord in coords_text.split()]
            else:
                coords_list = None

            # Let's Add the dataset
            placemarks.append({
                "name": name,
                "begin_time": begin_time,
                "end_time": end_time,
                "polygon": coords_list
            })

        # Dataframe conversion
        df = pd.DataFrame(placemarks)

        # Convertion of the timestamp
        df['begin_time'] = pd.to_datetime(df['begin_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        return df

    def off_nadir_pointing_angle(self, r_eci, v_eci, target_geodetic, time):
        # geodetic target → ECI

        eul_ang_ref = [0.0, 0.0, 0.0]
        P_target_eci = Point_Geodetic2ECI(*target_geodetic, time)
        # Distance Computation
        vec_eci = P_target_eci - r_eci
        # ECI → BRF
        pointing_vec_brf_target = IRF2BRF_eul(vec_eci, r_eci, v_eci, eul_ang_ref)

        # Compute the angle
        boresight_brf = np.array([0.0, 0.0, 1.0])
        boresight_brf /= np.linalg.norm(boresight_brf)

        pointing_vec_brf_target = pointing_vec_brf_target / np.linalg.norm(pointing_vec_brf_target)

        dot_product = np.clip(boresight_brf @ pointing_vec_brf_target, -1.0, 1.0)
        offnadir_angle_rad = np.arccos(dot_product)
        offnadir_angle_deg = np.rad2deg(offnadir_angle_rad)

        return float(offnadir_angle_deg), pointing_vec_brf_target

    def is_in_sight(self, target_geodetic, r_eci, v_eci, time, el_min):
        lat, lon, alt = target_geodetic
        Ecef_satellite = pm.eci2ecef(r_eci[0], r_eci[1], r_eci[2], time)
        e, n, u = pm.ecef2enu(Ecef_satellite[0], Ecef_satellite[1], Ecef_satellite[2], target_geodetic[0],
                              target_geodetic[1], target_geodetic[2])
        az, el, _ = pm.enu2aer(e, n, u)
        in_sight = el > el_min
        return in_sight

    def pointing_attitude_brf(self, pointing_vec_brf_target, is_in_view):
        if not is_in_view:
            return [0.0, 0.0, 0.0]

        eul_ang_ref = [0.0, 0.0, 0.0]
        phi_deg_ref = 0.0
        l1 = np.array([0.0, 0.0, 1.0])  # boresight reference (down)
        l2 = np.asarray(pointing_vec_brf_target).flatten()
        l1 /= np.linalg.norm(l1)
        l2 /= np.linalg.norm(l2)

        dot = np.clip(np.dot(l1, l2), -1.0, 1.0)
        cross = np.cross(l2, l1)  # <<< FIX HERE

        cos_phi_2 = np.cos(phi_deg_ref * np.pi / 180 / 2)
        sin_phi_2 = np.sin(phi_deg_ref * np.pi / 180 / 2)

        numerator_vec = cross * cos_phi_2 + (l1 + l2) * sin_phi_2
        numerator_scalar = (1 + dot) * cos_phi_2
        denominator = np.sqrt(2 * (1 + dot))

        q_vec = numerator_vec / denominator
        q_scalar = numerator_scalar / denominator
        quat = np.concatenate((q_vec, [q_scalar]))

        Rot_SRFa_SRFp = RotMat_by_quat(quat)
        Rot_LVLH2SRFp = RotMat_LVLH_to_BRF_by_eul(eul_ang_ref) @ Rot_SRFa_SRFp

        roll, pitch, yaw = rotation_matrix_to_ypr(Rot_LVLH2SRFp)
        return [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]


    @staticmethod
    def check_fov_in_polygon(df, simulation_time, fov_vertices):

        fov_vertices = [(lon, lat) for lat, lon in fov_vertices]

        # Check datetime
        df['begin_time'] = pd.to_datetime(df['begin_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        # Search the polygon in the KML which has the closest begin_time with respect to the observation epoch
        df['time_diff'] = abs(df['begin_time'] - simulation_time)
        closest_row = df.loc[df['time_diff'].idxmin()]

        # Extract the Polygon
        polygon_coords = closest_row['polygon']
        polygon = Polygon(polygon_coords) if polygon_coords else None
        geod = Geod(ellps="WGS84")

        # Set the results to a standard value
        fov_inside = False
        coverage_ratio = 0
        inside_count = 0
        intersection_area_km2 = 0
        temporal_distance_sec = (simulation_time - closest_row['begin_time']).total_seconds()

        if polygon:
            inside_count = sum(Point(pt).within(polygon) for pt in fov_vertices) - 1
            if inside_count >= 2:
                # Create the polygon
                fov_polygon = Polygon(fov_vertices)
                project = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    try:
                        intersection_polygon = fov_polygon.intersection(polygon)
                        if not intersection_polygon.is_empty:
                            coords_fov = list(fov_polygon.exterior.coords)
                            lon_fov = [pt[0] for pt in coords_fov]
                            lat_fov = [pt[1] for pt in coords_fov]
                            area_fov = abs(geod.polygon_area_perimeter(lon_fov, lat_fov)[0]) / 1e6
                            coords_int = list(intersection_polygon.exterior.coords)
                            lon_int = [pt[0] for pt in coords_int]
                            lat_int = [pt[1] for pt in coords_int]
                            area_int = abs(geod.polygon_area_perimeter(lon_int, lat_int)[0]) / 1e6
                            coverage_ratio = (area_int / area_fov) * 100 if area_fov > 0 else 0
                            intersection_area_km2 = area_int
                        fov_inside = inside_count >= 2 and coverage_ratio > 90
                    except RuntimeWarning:
                        print("Intersection Check Failed")
                        pdb.set_trace()

        return {
            'selected_polygon': polygon,
            'fov_inside': fov_inside,
            'coverage_ratio': coverage_ratio if polygon else 0,
            'inside_count': inside_count if polygon else 0,
            'intersection_area_km2': intersection_area_km2 if polygon else 0,
            'closest_time': closest_row['begin_time'],
            'area_id': closest_row['name'],
            'time_offset_sec': temporal_distance_sec
        }

    @staticmethod
    def check_fov_in_polygon_constrained(df, simulation_time, fov_vertices, max_time_diff_seconds):

        fov_vertices = [(lon, lat) for lat, lon in fov_vertices]
        df['begin_time'] = pd.to_datetime(df['begin_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        df['time_diff'] = abs(df['begin_time'] - simulation_time)
        closest_row = df.loc[df['time_diff'].idxmin()]
        temporal_distance_sec = abs((simulation_time - closest_row['begin_time']).total_seconds())

        polygon_coords = closest_row['polygon']
        polygon = Polygon(polygon_coords) if polygon_coords and temporal_distance_sec < max_time_diff_seconds else None
        geod = Geod(ellps="WGS84")

        fov_inside = False
        coverage_ratio = 0
        inside_count = 0
        intersection_area_km2 = 0
        containment_flag = None
        who_is_smaller = None

        if polygon:
            fov_polygon = Polygon(fov_vertices)

            # Areas computation
            coords_fov = list(fov_polygon.exterior.coords)
            lon_fov = [pt[0] for pt in coords_fov]
            lat_fov = [pt[1] for pt in coords_fov]
            area_fov = abs(geod.polygon_area_perimeter(lon_fov, lat_fov)[0]) / 1e6

            coords_poly = list(polygon.exterior.coords)
            lon_poly = [pt[0] for pt in coords_poly]
            lat_poly = [pt[1] for pt in coords_poly]
            area_poly = abs(geod.polygon_area_perimeter(lon_poly, lat_poly)[0]) / 1e6

            # Determine which is the smaller polygon
            if area_fov < area_poly:
                who_is_smaller = "fov"
                containment_flag = fov_polygon.within(polygon)
                larger_polygon = polygon
                smaller_polygon = fov_polygon
                area_smaller = area_fov
            else:
                who_is_smaller = "polygon"
                containment_flag = polygon.within(fov_polygon)
                larger_polygon = fov_polygon
                smaller_polygon = polygon
                area_smaller = area_poly

            inside_count = sum(Point(pt).within(larger_polygon) for pt in list(smaller_polygon.exterior.coords)) - 1

            if inside_count >= 2:
                project = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    try:
                        intersection_polygon = smaller_polygon.intersection(larger_polygon)
                        if not intersection_polygon.is_empty:
                            coords_int = list(intersection_polygon.exterior.coords)
                            lon_int = [pt[0] for pt in coords_int]
                            lat_int = [pt[1] for pt in coords_int]
                            area_int = abs(geod.polygon_area_perimeter(lon_int, lat_int)[0]) / 1e6
                            coverage_ratio = (area_int / area_smaller) * 100 if area_fov > 0 else 0
                            intersection_area_km2 = area_int

                            fov_inside = containment_flag
                    except RuntimeWarning:
                        print("Intersection Check Failed")
                        pdb.set_trace()

        return {
            'selected_polygon': polygon,
            'fov_inside': fov_inside,
            'coverage_ratio': coverage_ratio if polygon else 0,
            'inside_count': inside_count if polygon else 0,
            'intersection_area_km2': intersection_area_km2 if polygon else 0,
            'closest_time': closest_row['begin_time'],
            'area_id': closest_row['name'],
            'time_offset_sec': temporal_distance_sec,
            'who_is_smaller': who_is_smaller,
            'contains_other': containment_flag
        }

    @staticmethod
    def check_fov_and_coverage_constrained(df, simulation_time, fov_vertices, max_time_diff_seconds, center_lat,
                                           center_lon, coverage_threshold_percent):

        fov_vertices = [(lon, lat) for lat, lon in fov_vertices]
        df['begin_time'] = pd.to_datetime(df['begin_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        df['time_diff'] = abs(df['begin_time'] - simulation_time)
        closest_row = df.loc[df['time_diff'].idxmin()]
        temporal_distance_sec = abs((simulation_time - closest_row['begin_time']).total_seconds())

        polygon_coords = closest_row['polygon']
        polygon = Polygon(polygon_coords) if polygon_coords and temporal_distance_sec < max_time_diff_seconds else None
        geod = Geod(ellps="WGS84")

        fov_inside = False
        coverage_ratio = 0
        inside_count = 0
        intersection_area_km2 = 0
        is_inside_flag = None
        who_is_smaller = None
        is_valid = None  # Variabile logica finale da restituire

        center_point = Point(center_lat, center_lon)
        fov_polygon = Polygon(fov_vertices)

        # Step 1: check se il centro è nel FOV
        if not center_point.within(fov_polygon):
            return None, None

        if polygon:
            # Aree dei poligoni
            lon_fov, lat_fov = zip(*fov_polygon.exterior.coords)
            area_fov = abs(geod.polygon_area_perimeter(lon_fov, lat_fov)[0]) / 1e6

            lon_poly, lat_poly = zip(*polygon.exterior.coords)
            area_poly = abs(geod.polygon_area_perimeter(lon_poly, lat_poly)[0]) / 1e6

            # Chi è più piccolo
            if area_fov < area_poly:
                who_is_smaller = "fov"
                containment_flag = fov_polygon.within(polygon)
                larger_polygon = polygon
                smaller_polygon = fov_polygon
                area_smaller = area_fov
            else:
                who_is_smaller = "polygon"
                containment_flag = polygon.within(fov_polygon)
                larger_polygon = fov_polygon
                smaller_polygon = polygon
                area_smaller = area_poly

            # Intersezione e copertura
            inside_count = sum(Point(pt).within(larger_polygon) for pt in list(smaller_polygon.exterior.coords)) - 1

            if inside_count >= 2:
                try:
                    intersection_polygon = smaller_polygon.intersection(larger_polygon)
                    if not intersection_polygon.is_empty:
                        lon_int, lat_int = zip(*intersection_polygon.exterior.coords)
                        area_int = abs(geod.polygon_area_perimeter(lon_int, lat_int)[0]) / 1e6
                        coverage_ratio = (area_int / area_smaller) * 100 if area_smaller > 0 else 0
                        intersection_area_km2 = area_int

                        if coverage_ratio >= coverage_threshold_percent:
                            fov_inside = containment_flag
                            is_valid = 1
                        else:
                            is_valid = None
                except RuntimeWarning:
                    print("Intersection Check Failed")
                    pdb.set_trace()

        result = {
            'selected_polygon': polygon,
            'fov_inside': fov_inside,
            'coverage_ratio': coverage_ratio if polygon else 0,
            'inside_count': inside_count if polygon else 0,
            'intersection_area_km2': intersection_area_km2 if polygon else 0,
            'closest_time': closest_row['begin_time'],
            'area_id': closest_row['name'],
            'time_offset_sec': temporal_distance_sec,
            'who_is_smaller': who_is_smaller,
            'contains_other': containment_flag
        }

        return int(bool(is_valid)), result if is_valid else None

    @staticmethod
    def vec3d_to_list(v):
        return np.array([v.getX(), v.getY(), v.getZ()])

    # ========================================================================================
    # ADDED
    # ========================================================================================

    def get_center_vector_in_BRF(self):

        d5 = np.array([0.0, 0.0, 1.0])  # Add ray pointing to the center (boresight ray)

        # Normalize
        rays = [d5]
        rays = [v / np.linalg.norm(v) for v in rays]

        return np.array(rays)

    # taken from TestNum10.py
    def get_FovPoints(self, r_vec, v_vec, t_datetime):

        ray_directions = self.get_fov_vectors_in_BRF()
        intersections_matrix = self._find_intersection_in_Geodetic(ray_directions, t_datetime, r_vec, v_vec)
        FovPoints = intersections_matrix[:2, :].T

        return FovPoints

    def get_CenterRay_Intersection(self, r_vec, v_vec, t_datetime):
        """
        Intersect the center ray defined as the *geodetic nadir* (ellipsoid normal),
        so its hit matches the satellite's geodetic lat/lon and alt=0 (to numerical precision).
        """
        # Satellite geodetic (lat, lon, h) at epoch
        sat_lat, sat_lon, _ = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()

        # WGS-84 a,b
        a = 6378137.0
        b = 6356752.314245

        # Surface point (alt=0) in ECEF at sub-satellite geodetic lat/lon
        x0, y0, z0 = pm.geodetic2ecef(sat_lat, sat_lon, 0.0)

        # Ellipsoid normal at that surface point (outward); geodetic nadir is inward
        n_ecef = np.array([x0 / (a * a), y0 / (a * a), z0 / (b * b)], dtype=float)
        n_ecef /= np.linalg.norm(n_ecef)
        nadir_ecef = -n_ecef  # point toward Earth

        # Express this ray in BRF coordinates so the generic pipeline can be reused
        # ECEF -> ECI (direction) at the same epoch
        nx_eci, ny_eci, nz_eci = pm.ecef2eci(nadir_ecef[0], nadir_ecef[1], nadir_ecef[2], t_datetime)
        n_eci = np.array([nx_eci, ny_eci, nz_eci], dtype=float)
        n_eci /= np.linalg.norm(n_eci)

        # ECI direction -> LVLH (components); with eul_ang=[0,0,0], BRF==LVLH
        ray_brf = IRF2LVLH(n_eci, r_vec, v_vec)

        # Use the general intersection routine
        boresight_hit = self._find_intersection_in_Geodetic(
            np.array([ray_brf]),
            self.eul_ang_deg,
            t_datetime,
            r_vec,
            v_vec
        )
        return boresight_hit

    def get_CenterRay_Intersection_Attitude(self, r_vec, v_vec, t_datetime):
        """
        Intersect the boresight ray defined by the spacecraft BRF z-axis
        after applying the current Euler attitude (roll, pitch, yaw).
        This is needed for off-nadir pointing tests.
        """
        # BRF boresight (z-axis)
        boresight_brf = np.array([0.0, 0.0, 1.0])

        # Use the general intersection routine with this ray
        boresight_hit = self._find_intersection_in_Geodetic(
            np.array([boresight_brf]),
            self.eul_ang_deg,
            t_datetime,
            r_vec,
            v_vec
        )

        return boresight_hit

    def pick_west_or_east(self, points3xN):
        lats, lons, alts = points3xN

        # Split
        west_mask = lons < 0
        east_mask = lons > 0

        west_pts = points3xN[:, west_mask]
        east_pts = points3xN[:, east_mask]

        if west_pts.shape[1] != 4 or east_pts.shape[1] != 4:
            raise ValueError("Expected two groups of 4 points each.")

        # Mean longitude (wrap east > 180 back into [-180,180] for comparison)
        west_center = np.mean(west_pts[1])
        east_center = np.mean((east_pts[1] + 180) % 360 - 180)

        # Decide: which one is "more shifted"?
        # If absolute east_center > absolute west_center → east, else west
        if abs(east_center) > abs(west_center):
            return east_pts  # more to the east
        else:
            return west_pts  # more to the west


    def _wrap_deg180(self, a):
        """Wrap angles (deg) elementwise to [-180, 180]."""
        a = (a + 180.0) % 360.0 - 180.0
        a[a == 180.0] = -180.0
        return a

    def compute_delta_euler_step(self,
        eul_ang_cue_target,
        rot_rate_max,
        sim_step_seconds,
        deadband_deg=1e-6):

        """
        Compute per-timestep delta [d_roll, d_pitch, d_yaw] in degrees, moving from
        eul_ang_deg toward eul_ang_cue_target, limited by a maximum TOTAL rotation rate
        (deg/s) across all axes combined. Allows 'diagonal' rotation by scaling the full vector.

        Parameters
        ----------
        eul_ang_deg : array-like, shape (3,)
            Current Euler angles [roll, pitch, yaw] in degrees.
        eul_ang_cue_target : array-like, shape (3,)
            Target Euler angles [roll, pitch, yaw] in degrees.
        max_total_sensor_rot_deg_per_sec : float
            Maximum combined rotation rate (deg/s) across all axes.
        sim_step_seconds : float
            Simulation time step in seconds.
        deadband_deg : float, optional
            If the remaining difference norm is below this, return zeros.

        Returns
        -------
        np.ndarray, shape (3,)
            Delta to apply this timestep in degrees [d_roll, d_pitch, d_yaw].
        """

        current = np.asarray(self.eul_ang_deg, dtype=float).reshape(3)
        target  = np.asarray(eul_ang_cue_target, dtype=float).reshape(3)

        if rot_rate_max <= 0.0 or sim_step_seconds <= 0.0:
            return np.zeros(3, dtype=float)

        # Shortest-path difference
        diff = self._wrap_deg180(target - current)
        diff_mag = float(np.linalg.norm(diff))

        if diff_mag <= deadband_deg:
            return np.zeros(3, dtype=float)

        max_step_mag = rot_rate_max * sim_step_seconds

        if diff_mag <= max_step_mag:
            return diff

        return diff * (max_step_mag / diff_mag)























