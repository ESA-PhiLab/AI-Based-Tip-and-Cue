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
    _actor_attitude_in_deg = None
    # Actor pointing vector expressed in inertial frame.
    _actor_pointing_vector_eci = None
    # Actor pointing vector expressed in body reference frame
    _actor_pointing_vector_body = None
    # Earth Polar Radius

    """
    This class is provided with all the functions needed to perform the dedicated Earth-Observation activities
    """

    def __init__(
            self,
            local_actor,
            actor_initial_attitude_in_deg: list[float] = [0.0, 0.0, 0.0],
            actor_FOV_ACT_in_deg: list[float] = [1.0],
            actor_FOV_ALT_in_deg: list[float] = [1.0],
            actor_pointing_vector_body: list[float] = [0.0, 0.0, 1.0],
    ):

        assert isinstance(local_actor, SpacecraftActor), (
            "local_actor must be a " "SpacecraftActor" "."
        )

        logger.trace("Initializing EOTools Function")
        self._actor = local_actor

        # Convert attitude in np.ndarray
        self._actor_attitude_in_deg = np.array(actor_initial_attitude_in_deg)
        # Convert pointing vector in np.ndarray
        self._actor_pointing_vector_body = np.array(actor_pointing_vector_body) / np.linalg.norm(
            np.array(actor_pointing_vector_body))
        # Creation of the FOV
        self.fov_angles = [actor_FOV_ACT_in_deg[0], actor_FOV_ALT_in_deg[0]]

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

    def _find_intersection_in_Geodetic(self, ray_direction, eul_ang, time, r, v):
        r = np.asarray(r).flatten()
        v = np.asarray(v).flatten()
        # Allocation of the spacraft ECI coordinates
        x_ECI, y_ECI, z_ECI = r[0], r[1], r[2]
        # Cration of an empty collctor of FOV lines directions in ECI
        d_in_ECI = []


        # Transformation of the FOV direction from Body Reference Frame to the ECI referenc frame
        for ray in ray_direction:
            new_vector = BRF2IRF_eul(ray, r, v, eul_ang)
            d_in_ECI.append(new_vector)

        # Procedure to get the footprint edge-points with the geodetic referenc model WGS84
        intersections = []
        for d in d_in_ECI:
            dx, dy, dz = d
            # Coefficients of the solutions obtained from the intersection of the FOV directions with the quadratic ellipsoid mathematical function
            A = (dx ** 2 / R_e ** 2) + (dy ** 2 / R_e ** 2) + (dz ** 2 / R_p ** 2)
            B = 2 * ((x_ECI * dx / R_e ** 2) + (y_ECI * dy / R_e ** 2) + (z_ECI * dz / R_p ** 2))
            C = (x_ECI ** 2 / R_e ** 2) + (y_ECI ** 2 / R_e ** 2) + (z_ECI ** 2 / R_p ** 2) - 1

            # Computation of the Delta for the solution
            delta = B ** 2 - 4 * A * C

            if delta < 0:
                continue  # No sol obtained
            # Solutions obtained
            t1 = (-B + np.sqrt(delta)) / (2 * A)
            t2 = (-B - np.sqrt(delta)) / (2 * A)

            # We select the min t in [t1;t2] value since related to the smallest-distance intersection point with respect the spacecraft
            if min(t1, t2) > 0:
                t = min(t1, t2)
            else:
                t = max(t1, t2)

            if t < 0:
                continue

            # Get and append the intersection points
            intersection_ECI = r + t * d
            intersection_Geod = Point_ECI2Geodetic(intersection_ECI[0], intersection_ECI[1], intersection_ECI[2], time)
            intersections.append(intersection_Geod)

        # print("\t\t Intersections: ", intersections)

        # Check on the number of intersections! If the intersection points are lower than 4, this means that the FOV does not fully intersect the Earth! (added != 1, so it works for boresight ray)
        if not (len(intersections) == 4 or len(intersections) == 1):
            raise ValueError("Not enough intersections point")

        # Return a matrix with the intersection points
        intersections_matrix = np.column_stack(intersections)
        lats = intersections_matrix[0, :]
        lons = intersections_matrix[1, :]
        alts = intersections_matrix[2, :]

        # (wrap-around check)
        max_lon_diff = np.max([abs(l1 - l2) for l1 in lons for l2 in lons])

        if max_lon_diff > 300:
            # Shifting Point FOV1
            lons_fov1 = np.array([lon - 360 if lon > 0 else lon for lon in lons])
            fov1 = np.vstack((lats, lons_fov1, alts))  # shape (3, 4)

            # Shifting Point FOV2
            lons_fov2 = np.array([lon + 360 if lon < 0 else lon for lon in lons])
            fov2 = np.vstack((lats, lons_fov2, alts))  # shape (3, 4)

            # Vertical Concatenation
            intersections_matrix = np.hstack((fov1, fov2))

            # ====================== ADDED ============================================
            print("\t Crossed longitude border, picked largest FOV")
            intersections_matrix = self.pick_west_or_east(intersections_matrix)

            # ==================== ADDED ======================================================

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

    def off_nadir_pointing_angle(self, z_brf, r_eci, v_eci, target_geodetic, eul_angles_deg, time):
        # geodetic target → ECI
        P_target_eci = Point_Geodetic2ECI(*target_geodetic, time)
        # Distance Computation
        vec_eci = P_target_eci - r_eci
        # ECI → BRF
        vec_brf = IRF2BRF_eul(vec_eci, r_eci, v_eci, eul_angles_deg)
        # Compute the angle
        z_brf = z_brf / np.linalg.norm(z_brf)
        vec_brf = vec_brf / np.linalg.norm(vec_brf)

        dot_product = np.clip(z_brf.T @ vec_brf, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.rad2deg(angle_rad)

        return float(angle_deg), vec_brf

    def is_in_sight(self, target_geodetic, r_eci, v_eci, time, el_min):
        lat, lon, alt = target_geodetic
        Ecef_satellite = pm.eci2ecef(r_eci[0], r_eci[1], r_eci[2], time)
        e, n, u = pm.ecef2enu(Ecef_satellite[0], Ecef_satellite[1], Ecef_satellite[2], target_geodetic[0],
                              target_geodetic[1], target_geodetic[2])
        az, el, _ = pm.enu2aer(e, n, u)
        in_sight = el > el_min
        return in_sight

    def pointing_attitude(self, l1, l2, phi_rad, attitude, is_in_view):

        if not is_in_view:
            return np.nan, np.nan, np.nan

        l1 = np.asarray(l1).flatten()
        l2 = np.asarray(l2).flatten()
        l1 = l1 / np.linalg.norm(l1)
        l2 = l2 / np.linalg.norm(l2)
        dot = np.clip(np.dot(l1, l2), -1.0, 1.0)
        cross = np.cross(l1, l2)
        cos_phi_2 = np.cos(phi_rad / 2)
        sin_phi_2 = np.sin(phi_rad / 2)
        numerator_vec = cross * cos_phi_2 + (l1 + l2) * sin_phi_2
        numerator_scalar = (1 + dot) * cos_phi_2
        denominator = np.sqrt(2 * (1 + dot))
        q_vec = numerator_vec / denominator
        q_scalar = numerator_scalar / denominator
        vec_fin = np.concatenate((q_vec, [q_scalar]))
        Rot_SRFa_SRFp = RotMat_by_quat(vec_fin)
        Rot_LVLH2SRFp = Rot_SRFa_SRFp @ RotMat_LVLH_to_BRF_by_eul(attitude)
        yaw, pitch, roll = rotation_matrix_to_ypr(Rot_LVLH2SRFp)
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        return yaw, pitch, roll

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
    def get_FovPoints(self, r_vec, v_vec, eul_ang, t_datetime):

        ray_directions = self.get_fov_vectors_in_BRF()
        intersections_matrix = self._find_intersection_in_Geodetic(ray_directions, eul_ang, t_datetime, r_vec, v_vec)
        FovPoints = intersections_matrix[:2, :].T

        return FovPoints

    def get_CenterRay_Intersection(self, r_vec, v_vec, eul_ang, t_datetime):
        boresight_ray_BRF = np.array([0.0, 0.0, 1.0])  # Ray pointing to the middle of the image

        boresight_hit = self._find_intersection_in_Geodetic(
            np.array([boresight_ray_BRF]),
            eul_ang,
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
























