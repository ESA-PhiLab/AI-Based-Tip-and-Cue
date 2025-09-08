import numpy as np
import pykep as pk
import pyvista as pv
from pyvista import examples
from datetime import datetime, timezone
import seaborn as sns
import re
from collections import defaultdict

from settings import R_earth
from .plot_pyvista import _eci_to_pv, init_sun_light, update_sun_light_eci


def analyze_keplerian_constellation(planets):
    """
    Extract plane/satellite structure from keplerian planets.
    """
    pattern = r"plane(\d+)_sat(\d+)"
    plane_sat_map = defaultdict(list)
    max_plane_id, max_sat_id = -1, -1
    max_semi_major_axis = float("-inf")

    for planet in planets:
        name = planet.name
        if name is None:
            continue
        try:
            sma = planet.orbital_elements[0]
            if sma >= max_semi_major_axis:
                max_semi_major_axis = sma
        except Exception:
            pass
        match = re.search(pattern, name)
        if match:
            plane_id = int(match.group(1))
            sat_id = int(match.group(2))
            plane_sat_map[plane_id].append(sat_id)
            max_plane_id = max(max_plane_id, plane_id)
            max_sat_id = max(max_sat_id, sat_id)

    num_planes = max_plane_id + 1 if max_plane_id >= 0 else 0
    sats_per_plane = max_sat_id + 1 if max_sat_id >= 0 else 0
    return num_planes, sats_per_plane, max_semi_major_axis


def orbit_points(planet, n_points=200):
    a, e, i, RAAN, w, M = planet.orbital_elements
    nus = np.linspace(0, 2 * np.pi, n_points)
    positions = []
    for nu in nus:
        elems = [a, e, i, RAAN, w, nu]
        r, _ = pk.par2ic(elems, pk.MU_EARTH)
        positions.append(np.array(r))
    return np.array(positions)


def satellite_position(planet):
    elems = planet.orbital_elements
    r, _ = pk.par2ic(elems, pk.MU_EARTH)
    return np.array(r)


def plot_constellation_pyvista(planet_lst_tip, planet_lst_cue, t_datetime=None):
    """
    Show Earth, orbital lines, and satellite spheres.
    Per-plane coloring using seaborn Paired palette.
    """
    if t_datetime is None:
        t_datetime = datetime.now(timezone.utc)

    pl = pv.Plotter(lighting="none")

    # Background
    cubemap = examples.download_cubemap_space_4k()
    pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, True)

    # Earth in meters
    earth_mesh = examples.planets.load_earth(radius=R_earth / 1000.0)
    earth_mesh.points *= 1000.0
    earth_tex = examples.load_globe_texture()
    pl.add_mesh(earth_mesh, texture=earth_tex, smooth_shading=True)

    # --- TIP constellation ---
    nPlanes_tip, nSats_tip, _ = analyze_keplerian_constellation(planet_lst_tip)
    colors_tip = sns.color_palette("Paired", nPlanes_tip)
    for planet in planet_lst_tip:
        match = re.search(r"plane(\d+)_sat(\d+)", planet.name)
        if not match:
            continue
        plane_id = int(match.group(1))
        color = colors_tip[plane_id]

        pts = _eci_to_pv(orbit_points(planet))
        line = pv.Spline(pts, pts.shape[0])
        pl.add_mesh(line, color=color, line_width=1.5)

        sat_pos = _eci_to_pv(satellite_position(planet))
        pl.add_mesh(pv.Sphere(radius=400e3, center=sat_pos[0]),
                    color=color, smooth_shading=True)

    # --- CUE constellation ---
    nPlanes_cue, nSats_cue, _ = analyze_keplerian_constellation(planet_lst_cue)
    colors_cue = sns.color_palette("Paired", nPlanes_cue)
    for planet in planet_lst_cue:
        match = re.search(r"plane(\d+)_sat(\d+)", planet.name)
        if not match:
            continue
        plane_id = int(match.group(1))
        color = colors_cue[plane_id]

        pts = _eci_to_pv(orbit_points(planet))
        line = pv.Spline(pts, pts.shape[0])
        pl.add_mesh(line, color=color, line_width=1.0)

        sat_pos = _eci_to_pv(satellite_position(planet))
        pl.add_mesh(pv.Sphere(radius=250e3, center=sat_pos[0]),
                    color=color, smooth_shading=True)

    # Sun light
    sun_light = init_sun_light(pl)
    update_sun_light_eci(sun_light, t_datetime, distance_scale=1e11)

    # Text
    pl.add_text("Constellation Orbits with Satellites", font_size=12)

    pl.show()
