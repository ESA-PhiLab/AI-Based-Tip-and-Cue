#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orekit
import paseos
import pyvista as pv
import pykep as pk
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.frames import FramesFactory
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import IERSConventions

from settings import *  # noqa: F403

from paseos import ActorBuilder, SpacecraftActor
from paseos.custom_paseos.propagation.orekit_propagator import OrekitPropagator

from simulation.constants import R_earth
from simulation.constellation import build_constellation
from simulation.targets.whales import init_whales
from simulation.targets.water_target_utils import load_land_mask, generate_random_water_targets, build_land_mask
from simulation.sim_utils import convert_M_to_lv, satellite_in_shadow, daylight_mask
from simulation.plotting.plot_pyvista import (
    make_plotter_eci,
    camera_position_xy,
    update_earth_rotation_eci,
    init_sun_light,
    update_sun_light_eci,
    sats_to_points_eci,
    whales_to_points_eci,
    _pump_pyvista_events,
)


def _poly_points(n: int) -> pv.PolyData:
    """_poly_points(n) -> pv.PolyData: Create point cloud PolyData with correct verts."""
    pts = np.zeros((n, 3), dtype=float)
    poly = pv.PolyData(pts)
    verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).ravel()
    poly.verts = verts
    return poly


def _set_poly_points(poly: pv.PolyData, pts: np.ndarray) -> None:
    """_set_poly_points(poly, pts) -> None: Replace points and keep verts consistent."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be (N,3)")
    poly.points = pts
    n = pts.shape[0]
    verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).ravel()
    poly.verts = verts


def _eci_to_pv(v: np.ndarray) -> np.ndarray:
    """_eci_to_pv(v) -> np.ndarray: Apply plotting axis flip ([-x,-y,+z])."""
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return np.array([-v[0], -v[1], v[2]], dtype=float)
    return np.column_stack([-v[:, 0], -v[:, 1], v[:, 2]])


def _make_line_poly() -> pv.PolyData:
    """_make_line_poly() -> pv.PolyData: Create a 2-point line PolyData that can be updated."""
    pts = np.zeros((2, 3), dtype=float)
    lines = np.array([2, 0, 1], dtype=np.int64)
    return pv.PolyData(pts, lines)


def _set_line(line_poly: pv.PolyData, p0: np.ndarray, p1: np.ndarray) -> None:
    """_set_line(line_poly, p0, p1) -> None: Update endpoints of a 2-point line."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    line_poly.points = np.vstack([p0, p1])


def main() -> int:
    """main() -> int: Render a single PyVista frame at t0 with daylight coloring and Sun-to-Earth ray."""
    uhd = False

    # -----------------------------
    # Sun ray toggle (Sun -> Earth center)
    # -----------------------------
    show_sun_ray = True
    sun_ray_length = 3 * R_earth  # make it clearly visible beyond Earth
    sun_ray_color = "peachpuff"
    sun_ray_width = 4

    main_path = Path(__file__).resolve().parent
    os.chdir(main_path)

    pv.global_theme.allow_empty_mesh = True

    # -----------------------------
    # Orekit init
    # -----------------------------
    orekit.initVM()
    setup_orekit_curdir(from_pip_library=True)

    utc = TimeScalesFactory.getUTC()
    iers2010 = IERSConventions.valueOf("IERS_2010")
    earth = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(iers2010, True))
    sun = CelestialBodyFactory.getSun()

    # -----------------------------
    # Start epoch
    # -----------------------------
    t0_local = datetime(2025, 9, 21, 12, 00, 00, tzinfo=timezone.utc)
    n_targets_local = 500

    t0_dt = t0_local
    t0_orekit = AbsoluteDate(
        t0_dt.year, t0_dt.month, t0_dt.day,
        t0_dt.hour, t0_dt.minute, t0_dt.second + t0_dt.microsecond / 1e6,
        utc,
    )
    t0_pykep = pk.epoch_from_string(t0_dt.strftime("%Y-%m-%d %H:%M:%S"))

    # -----------------------------
    # Build constellation (actors)
    # -----------------------------
    planet_lst_tip, _, _ = build_constellation(params_tip, "Tip", t0_pykep)  # noqa: F405
    planet_lst_cue, _, _ = build_constellation(params_cue, "Cue", t0_pykep)  # noqa: F405
    all_planets = planet_lst_tip + planet_lst_cue

    tip_actors, cue_actors = [], []
    for planet in all_planets:
        orbital_elements_true = convert_M_to_lv(planet.orbital_elements, t0_orekit)

        propagator = OrekitPropagator(
            orbital_elements=orbital_elements_true,
            epoch=t0_orekit,
            satellite_mass=sat_mass,  # noqa: F405
            area_s=area_s, cr_s=cr_s, area_d=area_d, cd=cd,  # noqa: F405
        )

        actor = ActorBuilder.get_actor_scaffold(name=planet.name, actor_type=SpacecraftActor, epoch=t0_pykep)
        ActorBuilder.set_custom_orbit(
            actor,
            lambda t, p=propagator: (
                list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getPosition().toArray()),
                list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getVelocity().toArray()),
            ),
            t0_pykep,
        )
        ActorBuilder.set_geometric_model(actor, sat_mass)  # noqa: F405
        ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)

        (tip_actors if "Tip" in planet.name else cue_actors).append(actor)

    if len(tip_actors) != 0:
        sim = paseos.init_sim(local_actor=tip_actors[0])
        for a in tip_actors[1:] + cue_actors:
            sim.add_known_actor(a)
    else:
        sim = paseos.init_sim(local_actor=cue_actors[0])
        for a in cue_actors[1:]:
            sim.add_known_actor(a)

    for a in tip_actors + cue_actors:
        a.set_time(t0_pykep)

    # -----------------------------
    # Land mask + targets (no motion)
    # -----------------------------
    os.makedirs(worldmap_dir, exist_ok=True)  # noqa: F405
    npy_path_full = os.path.join(worldmap_dir, mask_npy)  # noqa: F405

    if not os.path.exists(npy_path_full):
        mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)  # noqa: F405
    else:
        mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)  # noqa: F405

    known_targets = generate_random_water_targets(
        n_targets_local,
        mask,
        res_deg,  # noqa: F405
        seed_val=whale_seed,  # noqa: F405
        max_abs_lat_val=max_abs_lat,  # noqa: F405
    )
    all_targets = init_whales(known_targets, seed_val=whale_seed, pos_fraction=pos_fraction)  # noqa: F405

    # -----------------------------
    # Sun vectors at t0
    # -----------------------------
    sun_pos_eci = sun.getPVCoordinates(t0_orekit, FramesFactory.getEME2000()).getPosition()
    sun_vec_eci = np.array([sun_pos_eci.getX(), sun_pos_eci.getY(), sun_pos_eci.getZ()], dtype=float)

    sun_pos_ecef = sun.getPVCoordinates(t0_orekit, FramesFactory.getITRF(iers2010, True)).getPosition()
    sun_vec_ecef = np.array([sun_pos_ecef.getX(), sun_pos_ecef.getY(), sun_pos_ecef.getZ()], dtype=float)

    illuminated_ids = daylight_mask(all_targets, sun_vec_ecef)

    # Unit direction from Earth center to Sun (ECI)
    s_hat_eci = sun_vec_eci / max(float(np.linalg.norm(sun_vec_eci)), 1e-12)

    # -----------------------------
    # PyVista scene (single render)
    # -----------------------------
    pl, earth_actor, earth_state = make_plotter_eci(uhd=uhd)
    pl.clear()

    cubemap = pv.examples.download_cubemap_space_4k()
    pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, is_srgb=True)

    earth_mesh = pv.examples.planets.load_earth(radius=R_earth)
    earth_tex = pv.examples.load_globe_texture()
    earth_actor = pl.add_mesh(earth_mesh, texture=earth_tex, smooth_shading=True)
    earth_state = {"last_theta": None}

    pl.show_axes()

    # --- Camera facing Europe ---
    # Europe approx: lat ~50°N, lon ~10°E

    lat_deg = 50.0
    lon_deg = 10.0
    dist_factor = 6.25 * R_earth

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    x = dist_factor * np.cos(lat) * np.cos(lon)
    y = dist_factor * np.cos(lat) * np.sin(lon)
    z = dist_factor * np.sin(lat)

    cam_pos_eci = np.array([x, y, z])
    cam_pos_pv = _eci_to_pv(cam_pos_eci)

    pl.camera.position = cam_pos_pv
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)

    sun_light = init_sun_light(pl)

    whales_day = _poly_points(0)
    whales_night = _poly_points(0)
    sats_lit = _poly_points(0)
    sats_unlit = _poly_points(0)

    pl.add_points(whales_day, color="yellow", render_points_as_spheres=True, point_size=8)
    pl.add_points(whales_night, color="slategray", render_points_as_spheres=True, point_size=8)
    pl.add_points(sats_lit, color="lawngreen", render_points_as_spheres=True, point_size=18)
    pl.add_points(sats_unlit, color="red", render_points_as_spheres=True, point_size=18)

    # --- Sun ray (outside Earth, always visible on lit side) ---
    # --- Sun -> Earth center ray using add_lines ---
    if show_sun_ray:
        p_center = _eci_to_pv(np.array([0.0, 0.0, 0.0]))
        p_sun = _eci_to_pv(s_hat_eci * sun_ray_length)

        pl.add_lines(
            np.array([p_sun, p_center]),
            color=sun_ray_color,
            width=sun_ray_width,
        )


    pl.add_text("Single timestep (t0)", position="lower_left", font_size=10, color="white")

    if not uhd:
        pl.show(cpos="yz", interactive_update=True, auto_close=False)

    # Earth rotation + sunlight at t0
    update_earth_rotation_eci(earth_actor, t0_dt, earth_state)
    update_sun_light_eci(sun_light, t0_dt)

    # Targets -> day/night
    pts_all = whales_to_points_eci(all_targets, t=t0_dt)
    ids_sorted = [w.id for _, w in sorted(all_targets.items())]
    mask_day = np.array([tid in illuminated_ids for tid in ids_sorted], dtype=bool)

    if pts_all.shape[0] == 0:
        _set_poly_points(whales_day, np.zeros((0, 3)))
        _set_poly_points(whales_night, np.zeros((0, 3)))
    else:
        _set_poly_points(whales_day, pts_all[mask_day])
        _set_poly_points(whales_night, pts_all[~mask_day])

    sat_pos_lit, sat_pos_unlit = [], []
    for a in tip_actors + cue_actors:
        try:
            r_list, _v_list = a.get_position_velocity(t0_pykep)
            r = np.asarray(r_list, dtype=float).reshape(3)
        except Exception:
            continue

        is_lit = not satellite_in_shadow(r, sun_vec_eci, earth.getEquatorialRadius())
        (sat_pos_lit if is_lit else sat_pos_unlit).append(r)

    _set_poly_points(sats_lit, sats_to_points_eci(sat_pos_lit))
    _set_poly_points(sats_unlit, sats_to_points_eci(sat_pos_unlit))

    print("n sats lit/unlit:", len(sat_pos_lit), len(sat_pos_unlit))
    if sat_pos_lit:
        print("example sat norm (km):", np.linalg.norm(sat_pos_lit[0]) / 1000.0)

    pl.render()
    if not uhd:
        _pump_pyvista_events(pl)

    if not uhd:
        while pl.ren_win is not None:
            _pump_pyvista_events(pl)
            time.sleep(0.02)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())