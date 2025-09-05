from settings import *

import orekit
import pyvista as pv
import numpy as np
import pykep as pk
from datetime import datetime, timedelta
import csv, atexit
import time
import gc
import pandas as pd
from openpyxl import Workbook, load_workbook
import os

from orekit.pyhelpers import setup_orekit_curdir

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from simulation.constellation import build_constellation
from custom_paseos.utils.point_transformation import Point_ECI2Geodetic

from simulation.plotting.plot_functions import plot_constallation, plot_orbits, plot_all_fov_footprints
from simulation.plotting.plot_pyvista import make_plotter_eci, reset_plotter, update_plotter
from simulation.propagate_whales import update_whales, load_land_mask, generate_random_water_targets,  init_whales, build_land_mask
from simulation.simulation_functions import init_eo_tools, cleanup_tasked_targets, propagate_actor, log_tip_detection, log_cue_evaluation, satellite_in_shadow, daylight_mask, convert_M_to_lv
from simulation.logging import init_excel_log, log_tip_detection, log_cue_evaluation, gsd_offnadir, at_exit

show_constellation = False
plot_propagation = False
plot_footprints = False
show_orbits = False
generate_image = False
logging = True
verbose = False

# Initialize Orekit
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import IERSConventions
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory

from astropy.utils.iers import conf
conf.auto_max_age = None  # allow predictive values older than 30 days

pv.global_theme.allow_empty_mesh = True
paseos.set_log_level("WARNING")

# Time setup
print(f"Initiate simulation {sim_name}")
utc = TimeScalesFactory.getUTC()
t0_orekit = AbsoluteDate(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond / 1e6, utc)
t0_pykep = pk.epoch_from_string(t0.strftime("%Y-%m-%d %H:%M:%S"))

iers2010 = IERSConventions.valueOf("IERS_2010")
earth = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(iers2010, True))
sun = CelestialBodyFactory.getSun()

# Get constellations
planet_lst_tip, sats_tip, _ = build_constellation(params_tip, "Tip", t0_pykep)
planet_lst_cue, sats_cue, _ = build_constellation(params_cue, "Cue", t0_pykep)

# Combine planets
all_planets = planet_lst_tip + planet_lst_cue

if show_constellation:
    plot_constallation(planet_lst_tip, planet_lst_cue, R_earth=R_earth, plot_margin=500e3)

# Create actors
tip_actors, cue_actors = [], []
for planet in all_planets:

    orbital_elements_true = convert_M_to_lv(planet.orbital_elements, t0_orekit)

    propagator = OrekitPropagator(
        orbital_elements=orbital_elements_true,
        epoch=t0_orekit,
        satellite_mass=satellite_mass,
        area_s=area_s, cr_s=cr_s, area_d=area_d, cd=cd
    )

    actor = ActorBuilder.get_actor_scaffold(name=planet.name, actor_type=SpacecraftActor, epoch=t0_pykep)
    ActorBuilder.set_custom_orbit(actor, lambda t, p=propagator: (
        list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getPosition().toArray()),
        list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getVelocity().toArray())
    ), t0_pykep)

    ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)

    (tip_actors if "Tip" in planet.name else cue_actors).append(actor)

z_brf = np.array([0.0, 0.0, 1.0])
eul_ang_tip = [0.0, 0.0, 0.0]
eul_ang_cue = [0.0, 0.0, 0.0]
offnadir_cue_deg = 0.0
offnadir_tip_deg = 0.0
yaw, pitch, roll = 0.0, 0.0, 0.0
phi_rad = 0.0
all_fov_polygons = []

# EO Tools
eo_tools_dict = {}
eo_tools_dict.update(init_eo_tools(tip_actors, fov_tip, [0.0, 0.0, 0.0]))
eo_tools_dict.update(init_eo_tools(cue_actors, fov_cue, [0.0, 0.0, 0.0]))

if len(tip_actors) != 0:
    sim = paseos.init_sim(local_actor=tip_actors[0])
    for actor in tip_actors[1:] + cue_actors:
        sim.add_known_actor(actor)

else:
    sim = paseos.init_sim(local_actor=cue_actors[0])
    for actor in cue_actors[1:]:
        sim.add_known_actor(actor)

n_steps_total = int(sim_duration_seconds / sim_step_seconds) + 1
print("Total number of simulation steps:", n_steps_total)

if show_orbits:
    trajectories = {
        actor.name: {
            "r": np.zeros((n_steps_total, 3), dtype=float),
            "v": np.zeros((n_steps_total, 3), dtype=float)
        }
        for actor in tip_actors + cue_actors
    }
else:
    trajectories = None  # no storage


if logging:
    header_tip = ["target_id", "date", "actor", "target_lat", "target_lon", "target_alt",
                  "x", "y", "z", "vx", "vy", "vz", "offnadir_deg", "GSD_m", "in_footprint"]

    header_cue = ["target_id", "date", "actor", "target_lat", "target_lon", "target_alt",
                  "x", "y", "z", "vx", "vy", "vz",
                  "offnadir_deg", "GSD_m", "in_view", "in_footprint", "yaw", "pitch", "roll"]

    writer_tip = init_excel_log("sim_output_tip.xlsx", header_tip, sheet_name="TipLog")
    writer_cue = init_excel_log("sim_output_cue.xlsx", header_cue, sheet_name="CueLog")

    atexit.register(at_exit, save_name = sim_name)

os.makedirs(worldmap_dir, exist_ok=True)
npy_path_full = os.path.join(worldmap_dir, mask_npy)

if not os.path.exists(npy_path_full):
    mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)
else:
    mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)

known_targets = generate_random_water_targets(n_whales, mask, res_deg, seed_val=whale_seed, max_abs_lat_val=max_abs_lat)
all_targets = init_whales(known_targets, seed_val=whale_seed)       # live updated
tasked_targets, detected_targets, evaluated_targets = {}, {}, {}

if plot_propagation:
    pl, earth_actor, earth_state = make_plotter_eci()

    (earth_actor, earth_state,
     whales_plot_all, whales_plot_evaluated, whales_plot_tasked,
     cloud_tip_sats, cloud_cue_sats,
     tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
     sun_light,
     eval_pts, task_pts) = reset_plotter(pl, all_targets, n_whales, tip_actors, cue_actors, last_theta=None)

    pl.show(cpos="xy", interactive_update=True, auto_close=False)


elapsed_time, n_steps = 0.0, 0
while elapsed_time <= sim_duration_seconds:

    t_start = time.time()

    t_pykep = sim.local_time
    t_datetime = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=t_pykep.mjd2000)
    t_abs = AbsoluteDate(t_datetime.year, t_datetime.month, t_datetime.day, t_datetime.hour, t_datetime.minute, t_datetime.second + t_datetime.microsecond / 1e6, utc)

    for actor in tip_actors + cue_actors:
        actor.set_time(t_pykep)

    tip_positions, cue_positions, FovPoints_tip, FovPoints_cue = [], [], [], []

    # Update whales + cleanup
    update_whales(all_targets, tasked_targets, mask, res_deg, sim_step_seconds, whale_propagation)
    cleanup_tasked_targets(tasked_targets, t_datetime, detection_time_limit)

    # Sun vector in ECI (for satellite shadow check)
    sun_pos_eci = sun.getPVCoordinates(t_abs, FramesFactory.getEME2000()).getPosition()
    sun_vec_eci = np.array([sun_pos_eci.getX(), sun_pos_eci.getY(), sun_pos_eci.getZ()])

    # Sun vector in ECEF (for daylight check)
    sun_pos_ecef = sun.getPVCoordinates(t_abs, FramesFactory.getITRF(iers2010, True)).getPosition()
    sun_vec_ecef = np.array([sun_pos_ecef.getX(), sun_pos_ecef.getY(), sun_pos_ecef.getZ()])
    illuminated_targets = daylight_mask(all_targets, sun_vec_ecef)

    for actor in tip_actors:

        n_detections = 0
        tip_detected = False
        eo_tools = eo_tools_dict[actor.name]

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        tip_positions.append(r)

        FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_tip, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
        FovPoints_tip.append(FovPoints)

        h_m = float(np.linalg.norm(np.array(r))) - R_earth
        gsd_tip = gsd_offnadir(GSD0_tip, h_m, offnadir_tip_deg)

        try:
            tip_illuminated = not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        except:
            tip_illuminated = True
            print("!! Tip: failed to compute illumination state, set to True preventing exclusion.")
            print(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        if tip_illuminated:
            for whale_idx, whale in all_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                target_coord = (whale["lat"], whale["lon"], whale["alt"])
                in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

                if in_footprint:
                    print("!! Tip: Target detected", whale_idx)

                    w = {"lat": target_coord[0], "lon": target_coord[1], "alt": target_coord[2], "detection_time": t_datetime, "detection_satellite": actor.name}
                    w["tasking_delay"] = tasking_delay_tip
                    tasked_targets[whale_idx], detected_targets[whale_idx] = w, w
                    all_targets[whale_idx]["detected"] = 1

                    if logging:
                        log_tip_detection(writer_tip, t_datetime, actor, whale_idx, target_coord, r, v, offnadir_tip_deg,
                                          gsd_tip, in_footprint)


                    n_detections +=1
                    tip_detected = True

        if verbose == True and n_steps % print_interval == 0:
            if tip_detected == False:
                print(
                    f"\t{actor.name} | {t_datetime.isoformat()} | "
                    f"detections={n_detections} | illuminated={tip_illuminated} | gsd={gsd_tip}")

            if tip_detected == True:
                print(
                    f"\t{actor.name} | {t_datetime.isoformat()} | "
                    f"detections={n_detections} | illuminated={tip_illuminated} | gsd={gsd_tip} | {w['lat'], w['lon'], w['alt']}")

            boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_tip, t_datetime)
            if boresight_hit is not None:

                # Print where the tip satellite is positioned
                tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime)
                print(
                    f"\t\tTip position at lat={float(tip_lat):.4f}, "
                    f"lon={float(tip_lon):.4f}, alt={float(tip_alt):.1f}"
                )

                # Print where center ray intersects Earth
                lat_b, lon_b, alt_b = boresight_hit
                print(
                    f"\t\tTip boresight at lat={float(lat_b):.4f}, "
                    f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}"
                )

        if plot_footprints:
            all_fov_polygons.append(FovPoints)

    for actor in cue_actors:
        n_evaluated = 0
        in_view = False
        cue_evaluated = False
        cue_busy = False
        eo_tools = eo_tools_dict[actor.name]

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        cue_positions.append(r)

        cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
        cue_lat, cue_lon, cue_alt = float(cue_lat), float(cue_lon), float(cue_alt)  # meters above ellipsoid

        try:
            cue_illuminated = not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius())
        except:
            cue_illuminated = True
            print("!! Cue: failed to compute illumination state, set to True preventing exclusion.")
            print(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        if cue_illuminated:
            for whale_idx, whale in tasked_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                detection_time = tasked_targets[whale_idx]["detection_time"]
                tasking_delay = tasked_targets[whale_idx]["tasking_delay"]

                if t_datetime > detection_time + timedelta(seconds=tasking_delay): # Skip if not yet transmitted

                    target_coord = (whale["lat"], whale["lon"], whale["alt"])
                    in_view = eo_tools.is_in_sight(target_coord, r_vec, v_vec, t_datetime, el_min=elevation_min)

                    if in_view:

                        offnadir_cue_deg, vec_brf = eo_tools.off_nadir_pointing_angle(z_brf=z_brf, r_eci=r_vec, v_eci=v_vec, target_geodetic=target_coord, eul_angles_deg=[0.0, 0.0, 0.0], time=t_datetime)

                        if offnadir_cue_deg <= offnadir_max:

                            yaw, pitch, roll = eo_tools.pointing_attitude(z_brf, vec_brf, phi_rad, [0.0, 0.0, 0.0], in_view)
                            eul_ang_cue = [yaw, pitch, roll]

                            print("!! Cue: Target in view", whale_idx, " | Set yaw, pitch, roll to:", eul_ang_cue)
                            cue_busy = True

                            # ----------------------------
                            h_m = float(np.linalg.norm(np.array(r))) - R_earth
                            gsd_cue = gsd_offnadir(GSD0_cue, h_m, offnadir_cue_deg)

                            print(f"\t Off nadir {offnadir_cue_deg:.2f}, GSD {gsd_cue:.3f}")

                            # -------------------------------------

            if not cue_busy:
                eul_ang_cue = [0.0, 0.0, 0.0]
                offnadir_cue_deg = 0.0

        try:
            FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_cue, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
            FovPoints_cue.append(FovPoints)

        except:
            print("!! Cue: target out of sight, continue to the next step")
            eul_ang_cue = [0.0, 0.0, 0.0]
            offnadir_cue_deg = 0.0
            continue

        h_m = float(np.linalg.norm(np.array(r))) - R_earth
        gsd_cue = gsd_offnadir(GSD0_cue, h_m, offnadir_cue_deg)

        if cue_illuminated:

            for whale_idx, whale in all_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                target_coord = (whale["lat"], whale["lon"], whale["alt"])
                in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

                if in_footprint:

                    print("!! Cue: Target in footprint", whale_idx, " | Evaluate target")
                    w = {"lat": target_coord[0], "lon": target_coord[1], "alt": target_coord[2], "detection_time": t_datetime, "detection_satellite": actor.name}
                    w["tasking_delay"] = tasking_delay_cue
                    detected_targets[whale_idx], tasked_targets[whale_idx] = w, w  # to keep track of history
                    all_targets[whale_idx]["detected"] = 2  # detected by cue

                    if logging:
                        log_cue_evaluation(writer_cue, t_datetime, actor, whale_idx, target_coord, r, v,
                                           offnadir_cue_deg, gsd_cue, in_view, in_footprint, yaw, pitch, roll)

                    n_evaluated += 1
                    cue_evaluated = True

                    if generate_image:
                        print("Generate image")
                        DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(
                            img_path, satellite, cue_lat, cue_lon, cue_alt, target_coord[0], target_coord[1],
                            target_coord[2], t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                    evaluated_targets[whale_idx] = w
                    del tasked_targets[whale_idx]

                    if plot_footprints:
                        all_fov_polygons.append(FovPoints)
        
        if verbose == True and n_steps % print_interval == 0:
            if cue_evaluated == True:
                print(
                    f"\t{actor.name} | {t_datetime.isoformat()} | "
                    f"target={target_coord[:2]} | "
                    f"off nadir angle={offnadir_cue_deg:.2f}, in_view={in_view}, in_footprint={in_footprint} | gsd={gsd_cue}"
                )

            if cue_evaluated == False:
                print(
                    f"\t{actor.name} | {t_datetime.isoformat()} | "
                    f"detections={n_evaluated} | illuminated={cue_illuminated} | gsd={gsd_cue}")

            boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_cue, t_datetime)
            if boresight_hit is not None:
                # Print where the tip satellite is positioned

                print(
                    f"\t\tCue position  at lat={float(cue_lat):.4f}, "
                    f"lon={float(cue_lon):.4f}, alt={float(cue_alt):.1f}"
                )
                lat_b, lon_b, alt_b = boresight_hit
                print(
                    f"\t\tCue boresight at lat={float(lat_b):.4f}, "
                    f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}")

                if cue_evaluated == True:
                    print(
                        f"\t\tTarget location at lat={float(target_coord[0]):.4f}, "
                        f"lon={float(target_coord[1]):.4f}, alt={float(target_coord[2]):.1f}"
                    )

    t_mid  = time.time()

    if n_steps % 10 == 0:
        gc.collect()

    if plot_propagation and n_steps % plot_interval == 0:
        if n_steps % reset_plot_interval == 0 and n_steps > 0:
            last_theta = earth_state.get("last_theta", None)
            (earth_actor, earth_state,
             whales_plot_all, whales_plot_evaluated, whales_plot_tasked,
             cloud_tip_sats, cloud_cue_sats,
             tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
             sun_light, eval_pts, task_pts) = reset_plotter(pl, all_targets, n_whales, tip_actors, cue_actors, last_theta=last_theta)

        eval_pts, task_pts = update_plotter(
            pl,
            earth_actor, earth_state,
            sun_light,
            cloud_tip_sats, cloud_cue_sats,
            whales_plot_all, whales_plot_evaluated, whales_plot_tasked,
            tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
            t_datetime,
            tip_positions, cue_positions,
            all_targets, evaluated_targets, tasked_targets,
            eval_pts, task_pts,
            FovPoints_tip, FovPoints_cue
        )

    t_end = time.time()

    if n_steps % print_interval == 0:
        print(f" {n_steps} Time iteration: {t_mid - t_start:.2f} | Time plot: {t_end - t_mid:.2f}")

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)
    elapsed_time += sim_step_seconds
    n_steps += 1

if show_orbits:
    plot_orbits(trajectories)

if plot_footprints:
    plot_all_fov_footprints(all_fov_polygons, known_targets)





