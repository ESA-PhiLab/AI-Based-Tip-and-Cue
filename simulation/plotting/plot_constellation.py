import numpy as np
import pykep as pk
import pyvista as pv
from pyvista import examples
from datetime import datetime, timezone
import seaborn as sns
import re

from settings import R_earth
from .plot_pyvista import init_sun_light, update_sun_light_eci


def _eci_to_pv(coords: np.ndarray) -> np.ndarray:
    """Flip X and Y axes to map ECI/ECEF coords into PyVista's rendering frame."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        return np.array([-coords[0], -coords[1], coords[2]], dtype=float)
    return np.column_stack([-coords[:, 0], -coords[:, 1], coords[:, 2]])


def _get_plane_ids(planets):
    """Return a color-group id per planet. Walker names use plane id; independent names get unique ids."""
    plane_ids = []
    pattern = r"plane(\d+)_sat(\d+)"

    for idx, planet in enumerate(planets):
        name = getattr(planet, "name", "")
        match = re.search(pattern, name)
        if match:
            plane_ids.append(int(match.group(1)))
        else:
            plane_ids.append(idx)

    return plane_ids


def orbit_points(planet, n_points=200):
    """Sample orbit points from keplerian elements."""
    a, e, i, RAAN, w, M = planet.orbital_elements
    nus = np.linspace(0, 2 * np.pi, n_points)
    positions = []
    for nu in nus:
        elems = [a, e, i, RAAN, w, nu]
        r, _ = pk.par2ic(elems, pk.MU_EARTH)
        positions.append(np.array(r))
    return np.array(positions)


def satellite_position(planet):
    """Return current position from keplerian elements."""
    elems = planet.orbital_elements
    r, _ = pk.par2ic(elems, pk.MU_EARTH)
    return np.array(r)


def _add_group(pl, planets, sat_radius_m, line_width, palette_name):
    """Add one satellite group to the PyVista plotter with its own palette."""
    if not planets:
        return

    plane_ids = _get_plane_ids(planets)
    unique_plane_ids = sorted(set(plane_ids))
    plane_to_color_idx = {pid: i for i, pid in enumerate(unique_plane_ids)}
    colors = sns.color_palette(palette_name, max(1, len(unique_plane_ids)))

    for planet, plane_id in zip(planets, plane_ids):
        color = colors[plane_to_color_idx[plane_id]]

        pts = _eci_to_pv(orbit_points(planet))
        line = pv.Spline(pts, pts.shape[0])
        pl.add_mesh(line, color=color, line_width=line_width)

        sat_pos = _eci_to_pv(satellite_position(planet))
        pl.add_mesh(
            pv.Sphere(radius=sat_radius_m, center=sat_pos),
            color=color,
            smooth_shading=True
        )


def plot_constellation_pyvista(planet_lst_tip, planet_lst_cue, t_datetime=None):
    """Show Earth, orbital lines, and satellite spheres."""
    if t_datetime is None:
        t_datetime = datetime.now(timezone.utc)

    pl = pv.Plotter(lighting="none")

    cubemap = examples.download_cubemap_space_4k()
    pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, True)

    earth_mesh = examples.planets.load_earth(radius=R_earth / 1000.0)
    earth_mesh.points *= 1000.0
    earth_tex = examples.load_globe_texture()
    pl.add_mesh(earth_mesh, texture=earth_tex, smooth_shading=True)

    _add_group(pl, planet_lst_tip, sat_radius_m=400e3, line_width=1.5, palette_name="Blues")
    _add_group(pl, planet_lst_cue, sat_radius_m=250e3, line_width=1.0, palette_name="Reds")

    sun_light = init_sun_light(pl)
    update_sun_light_eci(sun_light, t_datetime, distance_scale=1e11)

    pl.add_text("Constellation Orbits with Satellites", font_size=12)
    pl.show()


def plot_constellation_pyvista_plain(planet_lst_tip, planet_lst_cue, t_datetime=None):
    """Show transparent Earth sphere, orbital lines, and satellite spheres."""
    if t_datetime is None:
        t_datetime = datetime.now(timezone.utc)

    pl = pv.Plotter(lighting="none")
    pl.set_background("white")

    earth_sphere = pv.Sphere(radius=R_earth, theta_resolution=60, phi_resolution=60)
    pl.add_mesh(earth_sphere, color="grey", opacity=0.5, smooth_shading=True)

    _add_group(pl, planet_lst_tip, sat_radius_m=400e3, line_width=3.0, palette_name="Blues")
    _add_group(pl, planet_lst_cue, sat_radius_m=250e3, line_width=3.0, palette_name="Reds")

    pl.show_grid(color="black", font_size=10, bold=False)
    pl.add_text("Constellation Orbits with Satellites", font_size=12, color="black")
    pl.show()


def plot_constellation_pyvista_transparent_earth(planet_lst_tip, planet_lst_cue, t_datetime=None):
    """Show transparent Earth, orbital lines, and satellite spheres."""
    if t_datetime is None:
        t_datetime = datetime.now(timezone.utc)

    pl = pv.Plotter(lighting="none")
    pl.set_background("white")

    earth_mesh = examples.planets.load_earth(radius=R_earth / 1000.0)
    earth_mesh.points *= 1000.0
    earth_tex = examples.load_globe_texture()
    pl.add_mesh(
        earth_mesh,
        texture=earth_tex,
        smooth_shading=True,
        opacity=0.6,
        backface_culling=True
    )

    _add_group(pl, planet_lst_tip, sat_radius_m=400e3, line_width=1.5, palette_name="Blues")
    _add_group(pl, planet_lst_cue, sat_radius_m=250e3, line_width=1.0, palette_name="Reds")

    sun_light = init_sun_light(pl)
    update_sun_light_eci(sun_light, t_datetime, distance_scale=1e11)

    pl.add_text("Constellation Orbits with Satellites", font_size=12, color="black")
    pl.show()