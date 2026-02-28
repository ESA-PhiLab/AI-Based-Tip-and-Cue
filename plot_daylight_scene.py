#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
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
from simulation.targets.whales import update_whales, init_whales
from simulation.targets.water_target_utils import load_land_mask, generate_random_water_targets, build_land_mask
from simulation.sim_utils import convert_M_to_lv, propagate_actor, satellite_in_shadow, daylight_mask
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


def main() -> int:
    """main() -> int: Display and record a daylight-colored PyVista scene."""
    # -----------------------------
    # User toggles
    # -----------------------------
    movie_name = "daylight_scene.mp4"
    movie_framerate = 30
    uhd = False  # keep False on Windows for interactive + recording like your run_simulation.py

    # write a frame every N sim steps (1 = every step)
    write_every_n_steps = 1

    # -----------------------------
    # Run from script directory
    # -----------------------------
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
    t0_dt = t0  # noqa: F405
    if t0_dt.tzinfo is None:
        t0_dt = t0_dt.replace(tzinfo=timezone.utc)

    t0_orekit = AbsoluteDate(
        t0_dt.year, t0_dt.month, t0_dt.day,
        t0_dt.hour, t0_dt.minute, t0_dt.second + t0_dt.microsecond / 1e6,
        utc,
    )
    t0_pykep = pk.epoch_from_string(t0_dt.strftime("%Y-%m-%d %H:%M:%S"))

    # -----------------------------
    # Build constellation
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

    # -----------------------------
    # Land mask + targets
    # -----------------------------
    os.makedirs(worldmap_dir, exist_ok=True)  # noqa: F405
    npy_path_full = os.path.join(worldmap_dir, mask_npy)  # noqa: F405

    if not os.path.exists(npy_path_full):
        mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)  # noqa: F405
    else:
        mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)  # noqa: F405

    known_targets = generate_random_water_targets(
        n_targets,  # noqa: F405
        mask,
        res_deg,  # noqa: F405
        seed_val=whale_seed,  # noqa: F405
        max_abs_lat_val=max_abs_lat,  # noqa: F405
    )
    all_targets = init_whales(known_targets, seed_val=whale_seed, pos_fraction=pos_fraction)  # noqa: F405

    # -----------------------------
    # PyVista scene (EXACT PATTERN: show -> render -> open_movie)
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

    # camera like your sim
    dist_factor = 6.25
    angle_deg = -45.0
    pl.camera.position = camera_position_xy(dist_factor, angle_deg)
    pl.camera.focal_point = (0, 0, 0)

    sun_light = init_sun_light(pl)

    whales_day = _poly_points(0)
    whales_night = _poly_points(0)
    sats_lit = _poly_points(0)
    sats_unlit = _poly_points(0)

    pl.add_points(whales_day, color="yellow", render_points_as_spheres=True, point_size=10)
    pl.add_points(whales_night, color="slategray", render_points_as_spheres=True, point_size=10)
    pl.add_points(sats_lit, color="lawngreen", render_points_as_spheres=True, point_size=18)
    pl.add_points(sats_unlit, color="red", render_points_as_spheres=True, point_size=18)

    txt = pl.add_text("Daylight view", position="lower_left", font_size=10, color="white")

    # 1) SHOW FIRST (like your sim)
    if not uhd:
        pl.show(cpos="yz", interactive_update=True, auto_close=False)

    # 2) FORCE RENDER + PUMP EVENTS (like your sim)
    pl.render()
    if not uhd:
        _pump_pyvista_events(pl)

    # 3) FIX MACROBLOCK HEIGHT BASED ON REAL FRAMEBUFFER (like your sim)
    width, height = pl.window_size
    macro = 16
    height_fixed = (height + macro - 1) // macro * macro
    if height_fixed != height:
        pl.window_size = (width, height_fixed)
        pl.render()
        if not uhd:
            _pump_pyvista_events(pl)

    # 4) OPEN MOVIE AFTER WINDOW IS REALIZED (critical)
    pl.open_movie(
        movie_name,
        framerate=movie_framerate,
        format="FFMPEG",
        codec="libx264",
        quality=8,
    )

    # One more render after opening movie helps on Windows
    pl.render()
    if not uhd:
        _pump_pyvista_events(pl)

    # -----------------------------
    # Simulation loop
    # -----------------------------
    sim_seconds = float(sim_duration_hours) * 3600.0  # noqa: F405
    dt = float(sim_step_seconds)  # noqa: F405
    n_steps_total = int(sim_seconds / dt) + 1

    for k in range(n_steps_total):
        t_pykep = sim.local_time
        t_dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(days=t_pykep.mjd2000)
        t_abs = AbsoluteDate(
            t_dt.year, t_dt.month, t_dt.day,
            t_dt.hour, t_dt.minute, t_dt.second + t_dt.microsecond / 1e6,
            utc,
        )

        for a in tip_actors + cue_actors:
            a.set_time(t_pykep)

        update_whales(all_targets, mask, res_deg, dt, whale_propagation)  # noqa: F405

        # Sun vectors
        sun_pos_eci = sun.getPVCoordinates(t_abs, FramesFactory.getEME2000()).getPosition()
        sun_vec_eci = np.array([sun_pos_eci.getX(), sun_pos_eci.getY(), sun_pos_eci.getZ()], dtype=float)

        sun_pos_ecef = sun.getPVCoordinates(t_abs, FramesFactory.getITRF(iers2010, True)).getPosition()
        sun_vec_ecef = np.array([sun_pos_ecef.getX(), sun_pos_ecef.getY(), sun_pos_ecef.getZ()], dtype=float)

        illuminated_ids = daylight_mask(all_targets, sun_vec_ecef)

        # Earth rotation + sunlight
        update_earth_rotation_eci(earth_actor, t_dt, earth_state)
        update_sun_light_eci(sun_light, t_dt)

        # Targets -> day/night split
        pts_all = whales_to_points_eci(all_targets, t=t_dt)
        ids_sorted = [w.id for _, w in sorted(all_targets.items())]
        mask_day = np.array([tid in illuminated_ids for tid in ids_sorted], dtype=bool)

        if pts_all.shape[0] == 0:
            _set_poly_points(whales_day, np.zeros((0, 3)))
            _set_poly_points(whales_night, np.zeros((0, 3)))
        else:
            _set_poly_points(whales_day, pts_all[mask_day])
            _set_poly_points(whales_night, pts_all[~mask_day])

        # Satellites -> lit/shadow split
        sat_pos_lit, sat_pos_unlit = [], []
        for a in tip_actors + cue_actors:
            r_vec, v_vec, r, v = propagate_actor(a, t_pykep, trajectories=None, n_steps=0, show_orbits=False)
            is_lit = not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius())
            (sat_pos_lit if is_lit else sat_pos_unlit).append(np.asarray(r, dtype=float))

        _set_poly_points(sats_lit, sats_to_points_eci(sat_pos_lit))
        _set_poly_points(sats_unlit, sats_to_points_eci(sat_pos_unlit))

        txt.SetText(0, f"{t_dt.strftime('%d-%m-%y %H:%M:%S')} | day={int(mask_day.sum())} night={int((~mask_day).sum())}")

        pl.render()
        if not uhd:
            _pump_pyvista_events(pl)

        if k % max(1, write_every_n_steps) == 0:
            pl.write_frame()

        sim.advance_time(time_to_advance=dt, current_power_consumption_in_W=0.0)

    # Close movie cleanly
    try:
        pl._close_movie()
    except Exception:
        pass

    print(f"Saved movie: {movie_name}")

    # Keep window open
    if not uhd:
        while pl.ren_win is not None:
            _pump_pyvista_events(pl)
            time.sleep(0.02)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())