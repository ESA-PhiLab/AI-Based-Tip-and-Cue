# plot_pyvista.py
# Add Sun lighting in ECI: init_sun_light(), update_sun_light_eci(), sun_eci_vector()
# Now uses direct ECI/ECEF with Y-flip applied before PyVista rendering

import math
from datetime import datetime, timezone
from typing import List


import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from pyvista import examples
import pymap3d as pm
import gc
import math

from paseos.custom_paseos.utils.constants import R_earth
from paseos.custom_paseos.utils.point_transformation import Point_Geodetic2ECI, Point_ECI2Geodetic
from paseos.custom_paseos.utils.help_functions import compute_orbital_period
import pyvista as pv

# --- Orekit (used for precise Sun position in EME2000/ECI) ---
import orekit  # VM must be initialized by caller before using these functions
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory

from ..constants import R_earth


# --------------------------- Axis fix ---------------------------

def _eci_to_pv(coords: np.ndarray) -> np.ndarray:
    """Flip Y axis to map ECI/ECEF coords into PyVista's rendering frame."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        return np.array([-coords[0], -coords[1], coords[2]], dtype=float)
    return np.column_stack([-coords[:, 0], -coords[:, 1], coords[:, 2]])


# --------------------------- Basic Map Helpers (optional) ---------------------------

def plot_mask(mask: np.ndarray, res_deg: float) -> None:
    lon = np.linspace(-180.0, 180.0, mask.shape[1])
    lat = np.linspace(90.0, -90.0, mask.shape[0])
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(lon, lat, mask, cmap="Greys", shading="auto")
    plt.title("Land Mask (land=1, water=0)")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.colorbar(label="Mask value")
    plt.tight_layout()
    plt.show()



def camera_position_xy(dist_factor: float, angle_deg: float) -> tuple[float, float, float]:

    r = dist_factor * R_earth
    theta = math.radians(angle_deg)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = 0.0
    return (x, y, z)



# --------------------------- Static ECEF Plotter ---------------------------


def whales_to_points(whales: dict) -> np.ndarray:
    if not whales:
        return np.zeros((0, 3), dtype=float)
    pts = np.array(
        [Point_Geodetic2ECI(w["lat"], w["lon"], float(w.get("alt", 0.0))).flatten()
         for _, w in sorted(whales.items())],
        dtype=float,
    )
    return pts


def sats_to_points(sat_positions_ecef: List[np.ndarray]) -> np.ndarray:
    if len(sat_positions_ecef) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(sat_positions_ecef, dtype=float)


def _latlon_list_to_ecef(latlon: np.ndarray) -> np.ndarray:
    latlon = np.asarray(latlon, dtype=float)
    pts = np.array([Point_Geodetic2ECI(lat, lon, 0.0).flatten() for lat, lon in latlon], dtype=float)
    return pts


def _faces_from_fan(n_vertices: int) -> np.ndarray:
    if n_vertices < 3:
        raise ValueError("FoV polygon needs at least 3 vertices.")
    faces = []
    for i in range(1, n_vertices - 1):
        faces.extend([3, 0, i, i + 1])
    return np.asarray(faces, dtype=np.int32)


def fovpoly_from_latlon(latlon: np.ndarray) -> pv.PolyData:
    pts = _latlon_list_to_ecef(np.asarray(latlon))
    faces = _faces_from_fan(len(pts))
    return pv.PolyData(pts, faces)


def fovline_from_latlon(latlon: np.ndarray, close: bool = True) -> pv.PolyData:
    pts = _latlon_list_to_ecef(np.asarray(latlon))
    if close and len(pts) > 0 and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pv.lines_from_points(pts, close=False)


def init_fov_layers(
        pl: pv.Plotter,
        n_tip: int,
        n_cue: int,
        tip_fill_color: str = "orange",
        cue_fill_color: str = "cyan",
        tip_edge_color: str = "white",
        cue_edge_color: str = "white",
        opacity: float = 0.35,
        line_width: float = 5.0,
):
    placeholder_latlon = np.array([[0.0, 0.0], [0.0, 1e-6], [1e-6, 0.0]], dtype=float)

    tip_fill, tip_edge = [], []
    for _ in range(n_tip):
        poly = fovpoly_from_latlon(placeholder_latlon)
        line = fovline_from_latlon(placeholder_latlon, close=True)
        pl.add_mesh(poly, color=tip_fill_color, opacity=opacity, smooth_shading=True)
        pl.add_mesh(line, color=tip_edge_color, line_width=line_width)
        tip_fill.append(poly)
        tip_edge.append(line)

    cue_fill, cue_edge = [], []
    for _ in range(n_cue):
        poly = fovpoly_from_latlon(placeholder_latlon)
        line = fovline_from_latlon(placeholder_latlon, close=True)
        pl.add_mesh(poly, color=cue_fill_color, opacity=opacity, smooth_shading=True)
        pl.add_mesh(line, color=cue_edge_color, line_width=line_width)
        cue_fill.append(poly)
        cue_edge.append(line)

    return tip_fill, tip_edge, cue_fill, cue_edge


def update_fov_layers(
        tip_fill,
        tip_edge,
        cue_fill,
        cue_edge,
        fovpoints_tip_list: List[np.ndarray],
        fovpoints_cue_list: List[np.ndarray],
):
    for pd_fill, pd_edge, latlon in zip(tip_fill, tip_edge, fovpoints_tip_list):
        latlon = np.asarray(latlon)
        if latlon.size < 6:
            continue
        pts = _latlon_list_to_ecef(latlon)
        faces = _faces_from_fan(len(pts))
        pd_fill.points = pts
        pd_fill.faces = faces

        line_pts = _latlon_list_to_ecef(latlon)
        if not np.allclose(line_pts[0], line_pts[-1]):
            line_pts = np.vstack([line_pts, line_pts[0]])
        n_line_pts = len(line_pts)
        lines = np.c_[np.full(n_line_pts - 1, 2), np.arange(0, n_line_pts - 1), np.arange(1, n_line_pts)].ravel()
        pd_edge.points = line_pts
        pd_edge.lines = lines

    for pd_fill, pd_edge, latlon in zip(cue_fill, cue_edge, fovpoints_cue_list):
        latlon = np.asarray(latlon)
        if latlon.size < 6:
            continue
        pts = _latlon_list_to_ecef(latlon)
        faces = _faces_from_fan(len(pts))
        pd_fill.points = pts
        pd_fill.faces = faces

        line_pts = _latlon_list_to_ecef(latlon)
        if not np.allclose(line_pts[0], line_pts[-1]):
            line_pts = np.vstack([line_pts, line_pts[0]])
        n_line_pts = len(line_pts)
        lines = np.c_[np.full(n_line_pts - 1, 2), np.arange(0, n_line_pts - 1), np.arange(1, n_line_pts)].ravel()
        pd_edge.points = line_pts
        pd_edge.lines = lines


# --------------------------- ECI Plotter (rotating Earth) ---------------------------

def _eci_to_greenwich_angle_rad(t: datetime) -> float:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    xe, ye, _ = pm.eci2ecef(1.0, 0.0, 0.0, t)
    return math.atan2(ye, xe)


def make_plotter_eci(uhd):
    window_size = (3840, 2160) if uhd else (1920, 1072)
    off_screen = True if uhd else False

    pl = pv.Plotter(lighting="none", window_size=window_size, off_screen=off_screen)
    cubemap = examples.download_cubemap_space_4k()
    pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, is_srgb=True)

    earth_mesh = examples.planets.load_earth(radius=R_earth)
    earth_tex = examples.load_globe_texture()
    earth_actor = pl.add_mesh(earth_mesh, texture=earth_tex, smooth_shading=True)

    pl.show_axes()
    pl.view_isometric()
    state = {"last_theta": None}
    return pl, earth_actor, state


def update_earth_rotation_eci(earth_actor: pv.Actor, t: datetime, state: dict) -> None:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    theta = _eci_to_greenwich_angle_rad(t)
    if state.get("last_theta") is None:
        earth_actor.SetOrientation(0.0, 0.0, -math.degrees(theta))
    else:
        dtheta = theta - state["last_theta"]
        earth_actor.rotate_z(-math.degrees(dtheta))
    state["last_theta"] = theta


# --------------------------- ECI Conversions and Updaters ---------------------------

def geodetic_to_eci(lat_deg: float, lon_deg: float, alt_m: float, t: datetime) -> np.ndarray:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    x, y, z = pm.geodetic2ecef(lat_deg, lon_deg, alt_m)
    X, Y, Z = pm.ecef2eci(x, y, z, t)
    return np.array([X, Y, Z], dtype=float)


def sats_to_points_eci(sat_positions_eci: List[np.ndarray]) -> np.ndarray:
    if len(sat_positions_eci) == 0:
        return np.zeros((0, 3), dtype=float)
    return _eci_to_pv(np.asarray(sat_positions_eci, dtype=float))


def fovpoly_from_latlon_eci(latlon: np.ndarray, t: datetime) -> pv.PolyData:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    latlon = np.asarray(latlon, dtype=float)
    if latlon.ndim != 2 or latlon.shape[1] != 2 or len(latlon) < 3:
        raise ValueError("FoV polygon needs at least 3 rows of [lat, lon].")
    eci = np.array([geodetic_to_eci(lat, lon, 0.0, t) for lat, lon in latlon], dtype=float)
    eci = _eci_to_pv(eci)
    faces = _faces_from_fan(len(eci))
    return pv.PolyData(eci, faces)


def fovline_from_latlon_eci(latlon: np.ndarray, t: datetime, close: bool = True) -> pv.PolyData:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    latlon = np.asarray(latlon, dtype=float)
    eci = np.array([geodetic_to_eci(lat, lon, 0.0, t) for lat, lon in latlon], dtype=float)
    eci = _eci_to_pv(eci)
    if close and len(eci) > 0 and not np.allclose(eci[0], eci[-1]):
        eci = np.vstack([eci, eci[0]])
    return pv.lines_from_points(eci, close=False)


def init_fov_layers_eci(
        pl: pv.Plotter,
        n_tip: int,
        n_cue: int,
        tip_fill_color: str = "orange",
        cue_fill_color: str = "cyan",
        tip_edge_color: str = "white",
        cue_edge_color: str = "white",
        opacity: float = 0.35,
        line_width: float = 5.0,
):
    placeholder = np.array([[0.0, 0.0], [0.0, 1e-6], [1e-6, 0.0]], dtype=float)
    t0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    tip_fill, tip_edge = [], []
    for _ in range(n_tip):
        poly = fovpoly_from_latlon_eci(placeholder, t=t0)
        line = fovline_from_latlon_eci(placeholder, t=t0, close=True)
        pl.add_mesh(poly, color=tip_fill_color, opacity=opacity, smooth_shading=True)
        pl.add_mesh(line, color=tip_edge_color, line_width=line_width)
        tip_fill.append(poly)
        tip_edge.append(line)

    cue_fill, cue_edge = [], []
    for _ in range(n_cue):
        poly = fovpoly_from_latlon_eci(placeholder, t=t0)
        line = fovline_from_latlon_eci(placeholder, t=t0, close=True)
        pl.add_mesh(poly, color=cue_fill_color, opacity=opacity, smooth_shading=True)
        pl.add_mesh(line, color=cue_edge_color, line_width=line_width)
        cue_fill.append(poly)
        cue_edge.append(line)

    return tip_fill, tip_edge, cue_fill, cue_edge


def update_fov_layers_eci(
        tip_fill,
        tip_edge,
        cue_fill,
        cue_edge,
        fovpoints_tip_list: List[np.ndarray],
        fovpoints_cue_list: List[np.ndarray],
        t: datetime,
):
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    for pd_fill, pd_edge, latlon in zip(tip_fill, tip_edge, fovpoints_tip_list):
        if latlon is None or len(latlon) < 3:
            continue
        latlon = np.asarray(latlon, dtype=float)
        eci = np.array([geodetic_to_eci(lat, lon, 0.0, t) for lat, lon in latlon], dtype=float)
        eci = _eci_to_pv(eci)
        faces = _faces_from_fan(len(eci))
        pd_fill.points = eci
        pd_fill.faces = np.asarray(faces, dtype=np.int32)

        line_pts = eci
        if not np.allclose(line_pts[0], line_pts[-1]):
            line_pts = np.vstack([line_pts, line_pts[0]])
        n_line = len(line_pts)
        lines = np.c_[np.full(n_line - 1, 2), np.arange(0, n_line - 1), np.arange(1, n_line)].ravel()
        pd_edge.points = line_pts
        pd_edge.lines = lines

    for pd_fill, pd_edge, latlon in zip(cue_fill, cue_edge, fovpoints_cue_list):
        if latlon is None or len(latlon) < 3:
            continue
        latlon = np.asarray(latlon, dtype=float)
        eci = np.array([geodetic_to_eci(lat, lon, 0.0, t) for lat, lon in latlon], dtype=float)
        eci = _eci_to_pv(eci)
        faces = _faces_from_fan(len(eci))
        pd_fill.points = eci
        pd_fill.faces = np.asarray(faces, dtype=np.int32)

        line_pts = eci
        if not np.allclose(line_pts[0], line_pts[-1]):
            line_pts = np.vstack([line_pts, line_pts[0]])
        n_line = len(line_pts)
        lines = np.c_[np.full(n_line - 1, 2), np.arange(0, n_line - 1), np.arange(1, n_line)].ravel()
        pd_edge.points = line_pts
        pd_edge.lines = lines


# --------------------------- Utility: robust point-clouds ---------------------------

def make_points_polydata(n: int) -> pv.PolyData:
    pts = np.zeros((n, 3), dtype=float)
    poly = pv.PolyData(pts)
    verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).ravel()
    poly.verts = verts
    return poly


def update_points_polydata(poly: pv.PolyData, new_pts: np.ndarray) -> None:
    new_pts = np.asarray(new_pts, dtype=float)
    if new_pts.ndim != 2 or new_pts.shape[1] != 3:
        raise ValueError("new_pts must be (N,3)")
    if poly.n_points != new_pts.shape[0]:
        poly.points = new_pts
        n = new_pts.shape[0]
        verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).ravel()
        poly.verts = verts
    else:
        poly.points = new_pts


# --------------------------- Ground targets (whales) in ECI ---------------------------

def whales_to_points_eci(whales, t: datetime) -> np.ndarray:
    """
    Convert a dict of Whale objects to ECI coordinates at time t.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    if not whales:
        return np.zeros((0, 3), dtype=float)

    rows = []
    for _, w in sorted(whales.items()):
        lat, lon, alt = w.lat, w.lon, w.alt
        x, y, z = pm.geodetic2ecef(lat, lon, alt)
        X, Y, Z = pm.ecef2eci(x, y, z, t)
        rows.append([X, Y, Z])
    return _eci_to_pv(np.asarray(rows, dtype=float))


# --------------------------- Sun light in ECI ---------------------------

def sun_eci_vector(t: datetime) -> np.ndarray:
    """
    Sun position in EME2000 (ECI) in meters at UTC time t using Orekit.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    utc = TimeScalesFactory.getUTC()
    ad = AbsoluteDate(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6, utc)
    eme2000 = FramesFactory.getEME2000()
    sun = CelestialBodyFactory.getSun()
    pos = np.array(sun.getPVCoordinates(ad, eme2000).getPosition().toArray(), dtype=float)
    return pos  # ECI meters


def init_sun_light(pl: pv.Plotter) -> pv.Light:
    """
    Create a directional light representing the Sun. Must be updated each frame with update_sun_light_eci().
    """
    light = pv.Light(light_type='scene light', color='white')
    light.intensity = 1.0
    light.diffuse_color = 'white'
    light.focal_point = (0.0, 0.0, 0.0)
    pl.add_light(light)
    return light


def update_sun_light_eci(light: pv.Light, t: datetime, distance_scale: float = 1e8) -> None:
    """
    Point the light from the Sun toward Earth in ECI.
    distance_scale controls how far away the directional light position is placed.
    """
    s_eci = sun_eci_vector(t)
    s_eci = _eci_to_pv(s_eci)
    norm = float(np.linalg.norm(s_eci))
    if norm == 0.0:
        return
    light.position = tuple((s_eci / norm) * distance_scale)


def update_points_from_targets(points_array, targets_dict, t_datetime):
    """
    Update a (N,3) numpy array with Whale positions in ECI.
    NaN for whales not in the dict.
    """
    points_array[:] = np.nan
    for whale_idx, whale in targets_dict.items():
        pos = whales_to_points_eci({whale_idx: whale}, t_datetime)[0]
        points_array[whale_idx] = pos
    return points_array

def compute_movie_framerate(a, sim_step_seconds, plot_interval, movie_orbit_sec):
    T_orbit = compute_orbital_period(a)
    steps_per_orbit = T_orbit / sim_step_seconds
    frames_per_orbit = steps_per_orbit / plot_interval
    framerate = frames_per_orbit / movie_orbit_sec
    framerate = int(framerate)
    if framerate < 1:
        framerate = 1
    return framerate, frames_per_orbit

def reset_plotter(pl, all_targets, n_whales, tip_actors, cue_actors, last_theta=None, uhd=True):
    """Reset PyVista scene and add Earth, satellites, whales, FoVs, and Sun light."""
    pl.clear()

    res = 2 if uhd else 1

    # Background
    cubemap = examples.download_cubemap_space_4k()
    pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, is_srgb=True)

    # Earth
    earth_mesh = examples.planets.load_earth(radius=R_earth)
    earth_tex = examples.load_globe_texture()
    earth_actor = pl.add_mesh(earth_mesh, texture=earth_tex, smooth_shading=True)
    earth_state = {"last_theta": last_theta}

    # --- All whales (baseline size 8) ---
    whale_points = np.zeros((n_whales, 3))
    whales_poly = pv.PolyData(whale_points)
    whales_poly["state"] = np.zeros(n_whales, dtype=int)

    state_colors = ["navy", "orange", "yellow", "cyan", "green", "red"]
    cmap = mcolors.ListedColormap(state_colors)

    pl.add_points(
        whales_poly,
        scalars="state",
        render_points_as_spheres=True,
        point_size=8*res,
        cmap=cmap,
        clim=(0, 5),          # lock LUT range so 0..5 map correctly
        nan_color="gray",
        show_scalar_bar=False,
    )

    # --- Tasked whales (overlay, size 12, yellow) ---
    # IMPORTANT: build with verts and keep them updated later
    tasked_poly = make_points_polydata(0)
    pl.add_points(
        tasked_poly,
        color="orange",
        render_points_as_spheres=True,
        point_size=12*res,
    )

    # Satellites
    cloud_tip_sats = pv.PolyData(np.zeros((len(tip_actors), 3)))
    pl.add_points(cloud_tip_sats, color="yellowgreen", point_size=20*res, render_points_as_spheres=True)

    cloud_cue_sats = pv.PolyData(np.zeros((len(cue_actors), 3)))
    pl.add_points(cloud_cue_sats, color="lightseagreen", point_size=15*res, render_points_as_spheres=True)

    # FoVs
    tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes = init_fov_layers_eci(
        pl, n_tip=len(tip_actors), n_cue=len(cue_actors),
        tip_fill_color="orange", cue_fill_color="cyan",
        tip_edge_color="white", cue_edge_color="white",
        opacity=0.35, line_width=5.0
    )

    # Sun
    sun_light = init_sun_light(pl)

    # Step label
    step_text = pl.add_text("Step: 0", font_size=10, position="lower_right", color="slategrey")

    return (earth_actor, earth_state, sun_light,
            whales_poly, tasked_poly, cloud_tip_sats, cloud_cue_sats,
            tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
            step_text)


def update_plotter(pl,
                   earth_actor, earth_state,
                   sun_light, whales_poly, tasked_poly,
                   cloud_tip_sats, cloud_cue_sats,
                   tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
                   t_datetime, tip_positions, cue_positions,
                   all_targets, observed_targets_tip, tasked_targets, observed_targets_cue,
                   confirmed_targets_pos, confirmed_targets_neg,
                   FovPoints_tip, FovPoints_cue, step_text, n_steps):

    # Earth + Sun
    update_earth_rotation_eci(earth_actor, t_datetime, earth_state)
    update_sun_light_eci(sun_light, t_datetime)

    # Satellites
    cloud_tip_sats.points = sats_to_points_eci(tip_positions)
    cloud_cue_sats.points = sats_to_points_eci(cue_positions)

    # ---------------------- Whales (vectorized) ----------------------
    whales = list(all_targets.values())
    n_whales = len(whales)

    if n_whales > 0:
        lats = np.array([w.lat for w in whales])
        lons = np.array([w.lon for w in whales])
        alts = np.array([w.alt for w in whales])

        x, y, z = pm.geodetic2ecef(lats, lons, alts)
        X, Y, Z = pm.ecef2eci(x, y, z, t_datetime)
        whales_poly.points = _eci_to_pv(np.column_stack([X, Y, Z]))

        ids = np.array([w.id for w in whales], dtype=object)
        state = np.zeros(n_whales, dtype=int)

        # priority via order of assignment (later wins)
        if observed_targets_cue:
            state[np.isin(ids, list(observed_targets_cue.keys()))] = 3
        if observed_targets_tip:
            state[np.isin(ids, list(observed_targets_tip.keys()))] = 1
        if confirmed_targets_neg:
            state[np.isin(ids, list(confirmed_targets_neg.keys()))] = 5
        if confirmed_targets_pos:
            state[np.isin(ids, list(confirmed_targets_pos.keys()))] = 4

        whales_poly["state"] = state

        # --- Tasked whales overlay (bigger yellow) ---
        if tasked_targets:
            mask_tasked = np.isin(ids, list(tasked_targets.keys()))
            new_pts = whales_poly.points[mask_tasked]
            update_points_polydata(tasked_poly, new_pts)     # <-- ensures verts are correct
        else:
            update_points_polydata(tasked_poly, np.zeros((0, 3)))  # clear overlay cleanly

    # ---------------------- FoVs ----------------------
    update_fov_layers_eci(
        tip_fill_meshes, tip_edge_meshes,
        cue_fill_meshes, cue_edge_meshes,
        FovPoints_tip, FovPoints_cue, t_datetime
    )

    # Step label
    # Step label with timestamp
    step_text.SetText(1, t_datetime.strftime("%d-%m-%y %H:%M:%S"))

    pl.update()



def _remove_light(pl, L):
    if L is None:
        return
    try:
        try:
            pl.remove_light(L)      # detach from renderer if present
        except Exception:
            pass
        try:
            L.off()                 # be extra safe
        except Exception:
            pass
    except Exception:
        pass

# --- add once, anywhere near your close helper ---
def _mute_pyvista_light_destructor():
    try:
        from pv.plotting.lights import Light
        # Make destructor a no-op to avoid shutdown tracebacks
        Light.__del__ = lambda self: None
    except Exception:
        pass


def close_plotter_safely(pl, *, sun_light=None, extra_lights=None):

    if pl is None:
        _mute_pyvista_light_destructor()
        return

    # stop movie
    try:
        if getattr(pl, "_movie_open", False):
            pl._close_movie()
    except Exception:
        pass

    # remove lights
    try:
        if sun_light is not None:
            try: pl.remove_light(sun_light)
            except Exception: pass
    except Exception:
        pass
    if extra_lights:
        for L in list(extra_lights):
            try: pl.remove_light(L)
            except Exception: pass

    # clear scene and close
    for f in (pl.disable_picking, pl.clear, pl.deep_clean, pl.close, pv.close_all):
        try: f()
        except Exception: pass

    # drop strong refs BEFORE interpreter teardown
    try:
        del sun_light
        del extra_lights
    except Exception:
        pass

    # silence fragile destructor just in case
    _mute_pyvista_light_destructor()

    gc.collect()


    # 2) remove any custom lights (prevents Light.__del__ traceback)
    try:
        _remove_light(pl, sun_light)
        if extra_lights:
            for L in extra_lights:
                _remove_light(pl, L)
    except Exception:
        pass

    # 3) disable picking and clear actors/widgets/text
    try:
        pl.disable_picking()
    except Exception:
        pass
    try:
        pl.clear()                  # remove actors from the scene
    except Exception:
        pass

    # 4) free renderer resources and close window
    try:
        pl.deep_clean()
    except Exception:
        pass
    try:
        pl.close()                  # closes the render window + interactor
    except Exception:
        pass

    # 5) close any other open plotters (paranoia)
    try:
        pv.close_all()
    except Exception:
        pass

    # 6) make sure ref cycles are collected
    gc.collect()
# ----------------------------------------------------------------------------



