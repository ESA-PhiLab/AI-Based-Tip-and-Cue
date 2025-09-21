from settings import *

import orekit
import pyvista as pv
import numpy as np
import pykep as pk
from datetime import datetime, timedelta
import atexit
import time
import gc
import os, sys
import pandas as pd
import openpyxl
import shutil
import random
import uuid

from orekit.pyhelpers import setup_orekit_curdir

from paseos import ActorBuilder, SpacecraftActor
import paseos

from paseos.custom_paseos.propagation.orekit_propagator import OrekitPropagator
from paseos.custom_paseos.utils.point_transformation import Point_ECI2Geodetic, Point_Geodetic2ECI

from simulation.targets.whales import update_whales, init_whales
from simulation.targets.water_target_utils import load_land_mask, generate_random_water_targets, build_land_mask
from simulation.sim_utils import init_eo_tools, init_attitude_models, link_eo_attitude, cleanup_timeout_targets, propagate_actor, satellite_in_shadow, daylight_mask, convert_M_to_lv, pointing_cost, count_orbits_completed, compute_coverage_fraction, _clear_actor_task
from simulation.plotting.plot_functions import plot_orbits, plot_all_fov_footprints_plotly, plot_offnadir_distribution, plot_latency_distribution, plot_viewing_time_distribution, plot_gsd_distribution
from simulation.plotting.plot_pyvista import make_plotter_eci, reset_plotter, update_plotter, compute_movie_framerate, camera_position_xy, close_plotter_safely
from simulation.plotting.plot_constellation import plot_constellation_pyvista_plain
from simulation.sim_logging import init_excel_log, log_tip, log_cue, log_combined, log_img, gsd_offnadir, at_exit, Logger, compute_stats, format_hms, merge_tip_cue_combined

from onboard_ai.onboard_ai_tip import tip_ai_decision
from onboard_ai.onboard_ai_cue import cue_ai_decision

from offnadir_imaging.rendering import generate_image

show_constellation = False
show_orbits = False
plot_propagation, uhd = True, True
plot_footprints = True
plot_whale_trajectories = False

create_image = False
onboard_ai_tip = True
onboard_ai_cue = True
model_attitude_control = True

logging = True
verbose = False

if real_run:
    show_constellation = False
    show_orbits = False
    plot_propagation, uhd = True, True
    plot_footprints = True
    plot_whale_trajectories = False

    create_image = False
    onboard_ai_tip = True
    onboard_ai_cue = True
    model_attitude_control = True

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

from simulation.constellation import build_constellation

from astropy.utils.iers import conf
conf.auto_max_age = None  # allow predictive values older than 30 days

pv.global_theme.allow_empty_mesh = True
paseos.set_log_level("WARNING")

old_files = ["sim_output.xlsx", "simulation.mp4", "output.log"]
for old_file in old_files:
    if os.path.exists(old_file):
      os.remove(old_file)

# Time setup
# Redirect both stdout and stderr
if logging:
    sys.stdout = Logger("output.log")
    sys.stderr = sys.stdout

current_time_str = time.strftime("%H:%M:%S", time.localtime(time.time()))
print(f"Initiate {sim_name} | Real Run {real_run} | Runtime start {current_time_str}")

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
    #plot_constellation_pyvista(planet_lst_tip, planet_lst_cue, t0)
    plot_constellation_pyvista_plain(planet_lst_tip, planet_lst_cue, t0)

# Create actors
tip_actors, cue_actors = [], []
for planet in all_planets:
    orbital_elements_true = convert_M_to_lv(planet.orbital_elements, t0_orekit)

    propagator = OrekitPropagator(
        orbital_elements=orbital_elements_true,
        epoch=t0_orekit,
        satellite_mass=sat_mass,
        area_s=area_s, cr_s=cr_s, area_d=area_d, cd=cd
    )

    actor = ActorBuilder.get_actor_scaffold(name=planet.name, actor_type=SpacecraftActor, epoch=t0_pykep)
    actor.running_ai = False

    ActorBuilder.set_custom_orbit(actor, lambda t, p=propagator: (
        list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getPosition().toArray()),
        list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getVelocity().toArray())
    ), t0_pykep)

    ActorBuilder.set_geometric_model(actor, sat_mass)
    ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)

    (tip_actors if "Tip" in planet.name else cue_actors).append(actor)

eul_ang_tip_default = [0.0, 0.0, 0.0]
eul_ang_cue_default = [0.0, 0.0, 0.0]
offnadir_tip_deg = 0.0
offnadir_cue_deg = 0.0
offnadir_unbound = 0.0

rng_ai_tip = random.Random(seed_ai_tip)   # seed for Tip AI
rng_ai_cue = random.Random(seed_ai_cue)   # seed for Cue AI
rng_dem = random.Random(seed_dem)

n_targets_pos = int(round(n_targets * pos_fraction))
n_targets_neg = n_targets - n_targets_pos

# EO Tools
eo_tools_dict = init_eo_tools(tip_actors, cue_actors, fov_tip, fov_cue, offnadir_limit)
att_models_dict = init_attitude_models(tip_actors, cue_actors, eul_ang_tip_default, eul_ang_cue_default, omega_max_rad, alpha_max_rad, zeta, wn_rad, offnadir_limit, offnadir_margin)
link_eo_attitude(eo_tools_dict, att_models_dict)

if len(tip_actors) != 0:
    sim = paseos.init_sim(local_actor=tip_actors[0])
    for actor in tip_actors[1:] + cue_actors:
        sim.add_known_actor(actor)

else:
    sim = paseos.init_sim(local_actor=cue_actors[0])
    for actor in cue_actors[1:]:
        sim.add_known_actor(actor)

sim_duration_seconds = sim_duration_hours * 3600
n_steps_total = int(sim_duration_seconds / sim_step_seconds) + 1
n_snapshots = n_steps_total // plot_fov_interval + 1

if nPlanes_tip != 0 and nSats_tip != 0:
    fov_polygons_tip = [None] * n_snapshots * nPlanes_tip * nSats_tip  # fixed length list
else:
    fov_polygons_tip = []

if nPlanes_cue != 0 and nSats_cue != 0:
    fov_polygons_cue = [None] * n_snapshots * nPlanes_cue * nSats_cue  # fixed length list
else:
    fov_polygons_cue = []

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

os.makedirs(worldmap_dir, exist_ok=True)
npy_path_full = os.path.join(worldmap_dir, mask_npy)

if not os.path.exists(npy_path_full):
    mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)
else:
    mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)

n_targets_positive = int(n_targets * pos_fraction)
n_targets_negative = n_targets - n_targets_positive

known_targets = generate_random_water_targets(n_targets, mask, res_deg, seed_val=whale_seed, max_abs_lat_val=max_abs_lat)
all_targets = init_whales(known_targets, seed_val=whale_seed, pos_fraction=pos_fraction)

tasked_targets, observed_targets_tip, observed_targets_cue, confirmed_targets_pos, confirmed_targets_neg = {}, {}, {}, {}, {}
whale_trajectories = {idx: np.full((n_steps_total, 2), np.nan, dtype=float) for idx in all_targets.keys()}

if plot_propagation:
    pl, earth_actor, earth_state = make_plotter_eci(uhd)

    (earth_actor, earth_state, sun_light,
     whales_poly, tasked_poly, cloud_tip_sats, cloud_cue_sats,
     tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
     step_text) = reset_plotter(pl, all_targets, n_targets, tip_actors, cue_actors, last_theta=None, uhd=uhd)


    pl.show(cpos="yz", interactive_update=True, auto_close=False)

    dist_factor = 6.25
    angle_deg = -45.0

    pl.camera.position = camera_position_xy(dist_factor, angle_deg)  # look from -Y
    pl.camera.focal_point = (0, 0, 0)  # look at Earth center

    pv_framerate, frames_per_orbit = compute_movie_framerate(a_cue, sim_step_seconds, plot_pyvista_interval, movie_orbit_sec)
    pl.open_movie( "simulation.mp4",  framerate=pv_framerate)

if logging:
    header_tip = ["detection_id", "target_id","tip_actor", "tip_observation_time", "tip_confirmation_time", "tip_ai_decision", "true_label", "correct", "offnadir_deg", "gsd_m", "target_lat", "target_lon", "target_alt",
                  "tip_lat", "tip_lon", "tip_alt", "x", "y", "z", "vx", "vy", "vz"]

    header_cue = ["detection_id", "target_id", "cue_actor", "cue_observation_time", "cue_confirmation_time", "cue_ai_decision","true_label", "correct", "offnadir_deg", "gsd_m", "viewing_time","latency_observation", "latency_confirmation", "slew_stab_time",
                  "target_lat", "target_lon", "target_alt", "cue_lat", "cue_lon", "cue_alt", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw"]

    header_combined = ["detection_id", "target_id", "tip_actor", "cue_actor", "tip_observation_time", "tip_confirmation_time", "cue_observation_time", "cue_confirmation_time", "tip_ai_decision", "cue_ai_decision", "true_label", "correct", "offnadir_deg", "gsd_m", "viewing_time",
                       "latency_observation", "latency_confirmation", "target_lat", "target_lon", "target_alt", "cue_lat", "cue_lon", "cue_alt"]

    header_img_gen = ["detection_id", "cue_lat", "cue_lon", "cue_alt", "tgt_lat", "tgt_lon",
                            "tgt_alt", "t_datetime", "dem_seed"]

    header_overview = ["Metric", "Value", "Comment"]

    writer_overview = init_excel_log("sim_output.xlsx", header_overview, sheet_name="Overview")
    writer_combined = init_excel_log("sim_output.xlsx", header_combined, sheet_name="Combined")
    writer_tip = init_excel_log("sim_output.xlsx", header_tip, sheet_name="Tip")
    writer_cue = init_excel_log("sim_output.xlsx", header_cue, sheet_name="Cue")
    writer_img_gen = init_excel_log("sim_output.xlsx", header_img_gen, sheet_name="Img")

    results_dir = os.path.join("0_results", sim_name)
    os.makedirs(results_dir, exist_ok=True)

    copy_file = "settings.py"
    if os.path.exists(copy_file):
        dst = os.path.join(results_dir, copy_file)
        shutil.copy(copy_file, dst)
        if verbose:
            print(f"Copied {copy_file} to {dst.replace(os.sep, '/')}")
    else:
        if verbose:
            print(f"Warning: {copy_file} not found, skipping.")

    atexit.register(at_exit, save_name=sim_name, pl=pl, sun_light=sun_light, verbose_def=False, verbose_error=False)
    print("Initiated logging files")

observed_idx_tip = None
observed_idx_cue = None
footprint_idx_tip = 0
footprint_idx_cue = 0
cleanup_idx = []

n_observed_tip = 0
n_confirmed_tip = 0
n_tasked_tip = 0
n_tasked_cue = 0
n_observed_cue = 0
n_confirmed_cue = 0
n_confirmed_pos = 0
n_confirmed_neg = 0

elapsed_seconds, elapsed_hours, n_steps = 0.0, 0.0, 0

print("Total number of simulation steps:", n_steps_total)

t_sim_start = time.time()
while elapsed_seconds <= sim_duration_seconds:

    t_start = time.time()

    t_pykep = sim.local_time
    t_datetime = datetime(2000, 1, 1, 0, 0, 0) + timedelta(days=t_pykep.mjd2000)
    t_abs = AbsoluteDate(t_datetime.year, t_datetime.month, t_datetime.day, t_datetime.hour, t_datetime.minute, t_datetime.second + t_datetime.microsecond / 1e6, utc)

    for actor in tip_actors + cue_actors:
        actor.set_time(t_pykep)

    tip_positions, cue_positions, FovPoints_tip, FovPoints_cue = [], [], [], []

    # Update whales + cleanup
    update_whales(all_targets, mask, res_deg, sim_step_seconds, whale_propagation)
    if plot_whale_trajectories:
        for idx, whale in all_targets.items():
            whale_trajectories[idx][n_steps, :] = (whale.lat, whale.lon)

    all_cleanup_idx = cleanup_timeout_targets(all_targets, tasked_targets, t_datetime, observation_time_limit, cleanup_idx, eo_tools_dict, eul_ang_cue_default)

    cleanup_idx = []

    for idx in all_cleanup_idx:
        if verbose:
            print(f"!! Reset Target {idx} observation history")

    # Sun vector in ECI (for satellite shadow check)
    sun_pos_eci = sun.getPVCoordinates(t_abs, FramesFactory.getEME2000()).getPosition()
    sun_vec_eci = np.array([sun_pos_eci.getX(), sun_pos_eci.getY(), sun_pos_eci.getZ()])

    # Sun vector in ECEF (for daylight check)
    sun_pos_ecef = sun.getPVCoordinates(t_abs, FramesFactory.getITRF(iers2010, True)).getPosition()
    sun_vec_ecef = np.array([sun_pos_ecef.getX(), sun_pos_ecef.getY(), sun_pos_ecef.getZ()])
    illuminated_targets = daylight_mask(all_targets, sun_vec_ecef)

    for actor in tip_actors:

        tip_observed = False

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        tip_positions.append(r)

        FovPoints = eo_tools_dict[actor.name].get_FovPoints(r_vec, v_vec, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
        FovPoints_tip.append(FovPoints)

        if plot_footprints and n_steps % plot_fov_interval == 0:
            fov_polygons_tip[footprint_idx_tip] = FovPoints
            footprint_idx_tip += 1

        try:
            tip_illuminated = not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        except:
            tip_illuminated = True
            print(f"!! {actor.name}: failed to compute illumination state, set to True preventing exclusion.")
            print(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        if tip_illuminated:
            for whale_idx, whale in all_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                target_coord = whale.position()
                in_footprint = eo_tools_dict[actor.name].check_point_in_footprint(target_coord, FovPoints)

                # TIP OBSERVATION
                if in_footprint and whale.state_observing != 1:

                    print(f"!! {actor.name}: Observed Target {whale_idx}")

                    whale.tip_actor = actor.name
                    whale.t_observed_tip = t_datetime
                    whale.delay_confirmation_tip = delay_confirmation_tip
                    whale.state_observing = 1
                    whale.coord_observed = whale.position()

                    if whale.detection_id is None:
                        whale.detection_id = str(uuid.uuid4())


                    observed_targets_tip[whale_idx] = whale
                    observed_idx_tip = whale_idx
                    tip_observed = True
                    n_observed_tip += 1

                    h_m = float(np.linalg.norm(np.array(r))) - R_earth
                    gsd_tip = gsd_offnadir(gsd0_tip, h_m, offnadir_tip_deg)

                    if logging:
                        tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r[0], r[1], r[2], t_datetime).flatten()

                        log_tip(writer_tip,
                                detection_id=whale.detection_id,
                                target_id=whale_idx, tip_actor=actor.name,
                                tip_observation_date=t_datetime, tip_confirmation_date=None,
                                tip_ai_decision=None, true_label=whale.ai_class_true, correct=None,
                                offnadir_deg=offnadir_tip_deg, gsd_m=gsd_tip,
                                target_lat=whale.lat, target_lon=whale.lon, target_alt=whale.alt,
                                tip_lat=tip_lat, tip_lon=tip_lon, tip_alt=tip_alt,
                                x=r_vec[0], y=r_vec[1], z=r_vec[2],
                                vx=v_vec[0], vy=v_vec[1], vz=v_vec[2])

                        log_combined(writer_combined,
                                     detection_id=whale.detection_id,
                                     target_id=whale_idx, tip_actor=actor.name, cue_actor=None,
                                     tip_observation_date=t_datetime, tip_confirmation_date=None,
                                     cue_observation_date=None, cue_confirmation_date=None,
                                     tip_ai_decision=None, cue_ai_decision=None,
                                     true_label=whale.ai_class_true, correct=None,
                                     offnadir_deg=offnadir_tip_deg, gsd_m=gsd_tip, viewing_time=None,
                                     latency_observation=None, latency_confirmation=None,
                                     target_lat=whale.lat, target_lon=whale.lon, target_alt=whale.alt,
                                     cue_lat=None, cue_lon=None, cue_alt=None)

                # TIP CONFIRMATION
                if whale.t_observed_tip != None and whale.state_confirming < 1 and t_datetime > (whale.t_observed_tip + timedelta(seconds=delay_confirmation_tip)):

                        if onboard_ai_tip:
                            whale.confirmed_tip, label_tip = tip_ai_decision(whale, tip_tpr, tip_tnr, rng_ai_tip)

                        else:
                            whale.confirmed_tip, label_tip = True, "whale-tipped"

                        task_coord = whale.coord_observed

                        whale.t_confirmed_tip = t_datetime
                        whale.state_confirming=1

                        n_confirmed_tip +=1

                        if whale.confirmed_tip:

                            best_cue, best_dist = None, float("inf")

                            for cue_actor in cue_actors:
                                # Propagate cue satellite forward by avg_time_delay
                                t_future = pk.epoch(t_pykep.mjd2000 + avg_time_delay / pk.DAY2SEC)
                                r_future, _, _, _ = propagate_actor(cue_actor, t_future, None, n_steps, show_orbits=False)

                                # Target position in ECI at the same future time
                                tgt_lat, tgt_lon, tgt_alt = task_coord
                                tgt_vec = np.array(Point_Geodetic2ECI(tgt_lat, tgt_lon, tgt_alt, t_datetime + timedelta(seconds=avg_time_delay)))

                                # Distance between satellite and target in ECI
                                dist = np.linalg.norm(r_future - tgt_vec)

                                if dist < best_dist:
                                    best_dist = dist
                                    best_cue = cue_actor.name

                            whale.assigned_cue = best_cue
                            whale.t_tasked_tip = t_datetime
                            tasked_targets[whale_idx] = whale
                            n_tasked_tip += 1

                            eo_tools_dict[best_cue].task_queue.append({
                                "target_id": whale_idx,
                                "coord": task_coord,
                                "assign_time": t_datetime
                            })

                            whale.ai_class_predicted="whale-tipped"

                            print(f"!! {actor.name}: Confirmed Target {whale_idx}={whale.ai_class_predicted}, assigned to {best_cue} (actual={whale.ai_class_true})")

                        elif not whale.confirmed_tip:
                            whale.ai_class_predicted="not-whale"
                            confirmed_targets_neg[whale_idx] = whale
                            n_confirmed_neg +=1
                            cleanup_idx.append(whale_idx)

                            print(f"!! {actor.name}: Confirmed Target {whale_idx}={whale.ai_class_predicted}, no Cue assigned (actual={whale.ai_class_true})")

                        if logging:
                            log_tip(writer_tip,
                                    detection_id=whale.detection_id,
                                    target_id=whale_idx, tip_actor=actor.name,
                                    tip_observation_date=None, tip_confirmation_date=t_datetime,
                                    tip_ai_decision=label_tip, true_label=whale.ai_class_true,
                                    correct=((label_tip == "whale-tipped" and whale.ai_class_true == 'whale') or (label_tip == whale.ai_class_true)),
                                    offnadir_deg=None, gsd_m=None,
                                    target_lat=None, target_lon=None, target_alt=None,
                                    tip_lat=None, tip_lon=None, tip_alt=None,
                                    x=None, y=None, z=None, vx=None, vy=None, vz=None)

                            log_combined(writer_combined,
                                         detection_id=whale.detection_id,
                                         target_id=whale_idx, tip_actor=actor.name, cue_actor=whale.assigned_cue,
                                         tip_observation_date=None, tip_confirmation_date=t_datetime,
                                         cue_observation_date=None, cue_confirmation_date=None,
                                         tip_ai_decision=label_tip, cue_ai_decision=None,
                                         true_label=whale.ai_class_true,
                                         correct=(label_tip == whale.ai_class_true),
                                         offnadir_deg=None, gsd_m=None, viewing_time=None,
                                         latency_observation=None, latency_confirmation=None,
                                         target_lat=None, target_lon=None, target_alt=None,
                                         cue_lat=None, cue_lon=None, cue_alt=None)

        # Do not change attitude for tip
        att_models_dict[actor.name].set_target_euler(eul_ang_tip_default)
        # att_models_dict[actor.name]._actor_attitude_deg = att_models_dict[actor.name]._target_attitude_deg

        if verbose == True and n_steps % print_interval == 0:
            tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
            tip_lat, tip_lon, tip_alt = float(tip_lat), float(tip_lon), float(tip_alt)  # meters above ellipsoid

            print(f"\t\t{actor.name} | {t_datetime.isoformat()} | lat={tip_lat:.1f}, lon={tip_lon:.1f}, alt={tip_alt:.1f} | illuminated={tip_illuminated}")
            if tip_observed and observed_idx_tip is not None:
                whale_dbg = all_targets[observed_idx_tip]
                print(f"\t\tTarget: idx={observed_idx_tip} | gsd={gsd_tip:.2f} | lat={whale_dbg.lat:.1f}, lon={whale_dbg.lon:.1f}, alt={whale_dbg.alt:.1f}")

    for actor in cue_actors:
        cue_observed = False
        cue_confirmed = False

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        cue_positions.append(r)

        eo_tools_dict[actor.name]._actor = actor

        try:
            cue_illuminated = not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius())
        except:
            cue_illuminated = True
            print(f"!! {actor.name}: failed to compute illumination state, set to True preventing exclusion.")
            print(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        # CUE TASKING
        if cue_illuminated:
            for whale_idx, whale in tasked_targets.items():
                if whale.assigned_cue != actor.name and whale.assigned_cue is not None:
                    continue

                tip_time = whale.t_confirmed_tip

                if t_datetime > tip_time + timedelta(seconds=delay_transmission_TC):

                    if any(task.get("target_id") == whale_idx for task in eo_tools_dict[actor.name].task_queue) and whale.state_tasked == 0:
                        print(f"!! {actor.name}: Received task for Target {whale_idx}")
                        whale.state_tasked = 1


                    # If no current task, check the queue
                    if eo_tools_dict[actor.name].current_task is None and eo_tools_dict[actor.name].task_queue:

                        # Filter tasks that will actually become visible soon
                        ready_tasks = []
                        for task in eo_tools_dict[actor.name].task_queue:
                            will_be_visible, _ = eo_tools_dict[actor.name].will_be_visible_within(
                                task["coord"], r_vec, v_vec, t_datetime,
                                2.5* 60, el_min_deg=elevation_min, step=60.0
                            )

                            if will_be_visible:
                                ready_tasks.append(task)

                        if ready_tasks:
                            # Pick the ready task with least pointing cost
                            eo_tools_dict[actor.name].current_task = min(
                                ready_tasks,
                                key=lambda task: pointing_cost(task, eo_tools_dict[actor.name], r_vec, v_vec, t_datetime)
                            )
                            eo_tools_dict[actor.name].task_queue.remove(eo_tools_dict[actor.name].current_task)
                            print(f"!! {actor.name}: Starting task for Target {eo_tools_dict[actor.name].current_task['target_id']}")
                            whale.t_tasked_cue = t_datetime
                            n_tasked_cue += 1

                    # If still no current task, assign this whale now
                    if eo_tools_dict[actor.name].current_task is None and whale.assigned_cue == actor.name:

                        eo_tools_dict[actor.name].current_task = {
                            "target_id": whale_idx,
                            "coord": whale.coord_observed
                        }
                        print(f"!! {actor.name}: New task assigned: Target {whale_idx}")

                    # Work on the current task if one exists
                    if eo_tools_dict[actor.name].current_task and eo_tools_dict[actor.name].current_task["target_id"] == whale_idx:

                        task_id = eo_tools_dict[actor.name].current_task["target_id"]
                        task_coord = eo_tools_dict[actor.name].current_task["coord"]

                        in_view = eo_tools_dict[actor.name].is_in_sight(task_coord, r_vec, v_vec, t_datetime, el_min_deg=elevation_min)
                        will_be_in_view_soon, t_until = eo_tools_dict[actor.name].will_be_visible_within(task_coord, r_vec, v_vec, t_datetime, att_models_dict[actor.name].slew_stab_time_max, el_min_deg=elevation_min, step=30.0)  # check visibility within 2.5 min, as that is more than enough to prepare slewing and settle
                        will_be_in_view_later, _ = eo_tools_dict[actor.name].will_be_visible_within(task_coord, r_vec, v_vec, t_datetime, delta_t_tipcue, el_min_deg=elevation_min,  step=60.0)  # check visibility within 2.5 min, as that is more than enough to prepare slewing and settle
                        moving_towards, _ = eo_tools_dict[actor.name].is_moving_towards_target(r_vec, v_vec, task_coord, t_datetime, dt_check=sim_step_seconds )

                        if (will_be_in_view_soon or in_view) and not (eo_tools_dict[actor.name].move_set and moving_towards):
                            pointing_vec_brf_target, _, offnadir_unbound, time_to_sight = eo_tools_dict[actor.name].point_to_target_bounded(r_eci=r_vec, v_eci=v_vec, target_geodetic=task_coord, t_datetime=t_datetime, offnadir_max=offnadir_limit, mode='max', dt_step_coarse=sim_step_seconds)

                            eo_tools_dict[actor.name].move_set = True
                            eo_tools_dict[actor.name].offnadir_unbound_target = offnadir_unbound

                        if (in_view or will_be_in_view_soon) and not (offnadir_unbound >= (offnadir_limit + offnadir_margin) and not moving_towards):
                            att_models_dict[actor.name]._new_target_attitude_deg = att_models_dict[actor.name].pointing_attitude_brf(pointing_vec_brf_target)


                        if not (in_view or will_be_in_view_later):
                            # Task finished → reset and pick next later
                            print(f"!! {actor.name}: Target {task_id} out of view, delete task")

                            # reset local EO tool state
                            att_models_dict[actor.name]._new_target_attitude_deg = eul_ang_cue_default
                            _clear_actor_task(actor.name, task_id, eo_tools_dict, att_models_dict, eul_ang_cue_default)

                            # Global removal happens at the next loop via cleanup_timeout_targets:
                            if task_id in tasked_targets:
                                cleanup_idx.append(task_id)

            if eo_tools_dict[actor.name].current_task is None:
                if not np.allclose(att_models_dict[actor.name]._target_attitude_deg, eul_ang_cue_default, atol=0.1):
                    att_models_dict[actor.name]._new_target_attitude_deg = eul_ang_cue_default
                    print(f"!! {actor.name}: Set roll, pitch, yaw target back to default {eul_ang_cue_default[0]:.1f}, {eul_ang_cue_default[1]:.1f}, {eul_ang_cue_default[2]:.1f} deg")


                    # eo_tools_dict[actor.name].offnadir_unbound_target = None

            # CUE ATTITUDE CONTROL
            if model_attitude_control:

                # --- Handle new target attitude ---
                if not np.allclose(att_models_dict[actor.name]._new_target_attitude_deg, att_models_dict[actor.name]._target_attitude_deg, atol=0.1):

                    if not att_models_dict[actor.name].slew_active or not np.allclose(att_models_dict[actor.name]._new_target_attitude_deg, att_models_dict[actor.name]._target_attitude_deg, atol=10.0):

                            att_models_dict[actor.name]._planned_start_eul = att_models_dict[actor.name]._actor_attitude_deg.copy()
                            att_models_dict[actor.name]._planned_start_time = elapsed_seconds

                            task_update_mode = "new"

                    else:
                        task_update_mode = "taken"

                    att_models_dict[actor.name].plan_slew( start_eul_deg=att_models_dict[actor.name]._planned_start_eul, target_eul_deg=att_models_dict[actor.name]._new_target_attitude_deg,
                        omega_max_rad=omega_max_rad, alpha_max_rad=alpha_max_rad, zeta=zeta, wn_rad=wn_rad, dt=sim_step_seconds / 10, mode="per_axis",
                        t_start=att_models_dict[actor.name]._planned_start_time, w_stab_res=omega_stab_res, a_stab_res=alpha_stab_res )



                    if task_update_mode == 'new' and eo_tools_dict[actor.name].offnadir_unbound_target != None:
                        offnadir_unbound = eo_tools_dict[actor.name].offnadir_unbound_target
                        offnadir_cue_current, _ = att_models_dict[actor.name].offnadir_from_euler(att_models_dict[actor.name]._actor_attitude_deg)
                        offnadir_cue_target, _ = att_models_dict[actor.name].offnadir_from_euler(att_models_dict[actor.name]._new_target_attitude_deg)
                        current_target_eul = att_models_dict[actor.name]._actor_attitude_deg
                        new_target_eul = att_models_dict[actor.name]._new_target_attitude_deg


                        print(f"!! {actor.name}: Target {task_id} in reach at {offnadir_unbound:.1f} deg"
                            f" (current={offnadir_cue_current:.1f} deg, target={offnadir_cue_target:.1f} deg, duration={att_models_dict[actor.name].delay_slew_stab:.1f} s)")
                          #  f"| Roll, pitch, yaw current=[{current_target_eul[0]:.1f}, {current_target_eul[1]:.1f}, {current_target_eul[2]:.2f}] deg "
                          #  f", set to target=[{new_target_eul[0]:.1f}, {new_target_eul[1]:.1f}, {new_target_eul[2]:.1f}] deg"

                if att_models_dict[actor.name].slew_active:
                    att_models_dict[actor.name].follow_planned_slew(elapsed_seconds)

                    if np.allclose(att_models_dict[actor.name]._actor_attitude_deg,  att_models_dict[actor.name]._target_attitude_deg,  atol=0.1) and np.any(np.abs(att_models_dict[actor.name]._actor_angular_velocity)) <= 0.01 and np.any(np.abs(att_models_dict[actor.name]._actor_angular_acceleration) <= 0.01):
                        att_models_dict[actor.name].slew_active = False
                        current_eul = att_models_dict[actor.name]._actor_attitude_deg
                        print(f"!! {actor.name}: Completed move to roll, pitch, yaw {current_eul[0]:.1f}, {current_eul[1]:.1f}, {current_eul[2]:.1f} deg")


                        eo_tools_dict[actor.name].slew_stab_time = att_models_dict[actor.name].delay_slew_stab
                        att_models_dict[actor.name].delay_slew_stab = None

            #   if verbose and n_steps % print_interval == 0:
              #       delta_move_eul = eul_new_deg - prev_eul
              #       print(f"\t\t {actor.name} | Delta roll, pitch, yaw={delta_move_eul[0]:.1f}, {delta_move_eul[1]:.1f}, {delta_move_eul[2]:.1f} deg")

        try:
            # check off-nadir angle, and where the center ray intersects the Earth

            FovPoints = eo_tools_dict[actor.name].get_FovPoints(r_vec, v_vec, t_datetime)
            FovPoints_cue.append(FovPoints)

            if plot_footprints and n_steps % plot_fov_interval == 0:
                fov_polygons_cue[footprint_idx_cue] = FovPoints
                footprint_idx_cue += 1

        except:
            print(f"!! {actor.name}: no FOV intersection, continue to the next step")
            att_models_dict[actor.name].set_target_euler(eul_ang_cue_default)
            eo_tools_dict[actor.name].offnadir_unbound_target = None

            continue

        if cue_illuminated and not att_models_dict[actor.name].slew_active:
        # only observe if stable

            for whale_idx, whale in all_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                target_coord = whale.position()
                in_footprint = eo_tools_dict[actor.name].check_point_in_footprint(target_coord, FovPoints)
                offnadir_cue_deg, _ = att_models_dict[actor.name].offnadir_from_euler(att_models_dict[actor.name]._actor_attitude_deg)


                # CUE OBSERVATION
                if in_footprint and whale.state_observing != 2 and offnadir_cue_deg <= (offnadir_limit + offnadir_margin):

                    print(f"!! {actor.name}: Observed Target {whale_idx} at off-nadir {offnadir_cue_deg:.1f} deg")

                    whale.cue_actor = actor.name
                    whale.t_observed_cue = t_datetime
                    whale.coord_observed = target_coord

                    if whale.detection_id is None:
                        whale.detection_id = str(uuid.uuid4())

                    whale.delay_confirmation_cue = delay_confirmation_cue
                    whale.state_observing = 2

                    observed_targets_cue[whale_idx] = whale
                    observed_idx_cue = whale_idx

                    n_observed_cue += 1
                    cue_observed = True

                    # Remove globally
                    if whale_idx in tasked_targets:
                        del tasked_targets[whale_idx]
                        whale.state_tasked = 0

                    h_m = float(np.linalg.norm(np.array(r))) - R_earth
                    gsd_cue = gsd_offnadir(gsd0_cue, h_m, offnadir_cue_deg)

                    cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r[0], r[1], r[2], t_datetime).flatten()
                    cue_lat, cue_lon, cue_alt = float(cue_lat), float(cue_lon), float(cue_alt)
                    dem_seed = rng_dem.randint(0, 1000)

                    if create_image:
                        print("Generate image")
                        DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(img_path, satellite, cue_lat, cue_lon, cue_alt, target_coord[0], target_coord[1], target_coord[2], t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                    if whale.tip_actor != None and whale.t_observed_tip != None:
                        if t_datetime <= whale.t_observed_tip + timedelta(seconds=observation_time_limit):
                            latency_observation = (whale.t_observed_cue - whale.t_observed_tip).total_seconds()
                    else:
                        latency_observation = 0.0

                    viewing_time_left = eo_tools_dict[actor.name].compute_viewing_time(
                        r_vec, v_vec, task_coord, t_datetime, offnadir_limit, offnadir_margin=offnadir_margin, dt_max=observation_time_limit )

                    if logging:
                        log_cue(writer_cue,
                                detection_id=whale.detection_id,
                                target_id=whale_idx, cue_actor=actor.name,
                                cue_observation_date=t_datetime, cue_confirmation_date=None,
                                cue_ai_decision=None, true_label=whale.ai_class_true, correct=None,
                                offnadir_deg=offnadir_cue_deg, gsd_m=gsd_cue, viewing_time=viewing_time_left,
                                latency_observation=latency_observation, latency_confirmation=None,
                                slew_stab_time=eo_tools_dict[actor.name].slew_stab_time,
                                target_lat=whale.lat, target_lon=whale.lon, target_alt=whale.alt,
                                cue_lat=cue_lat, cue_lon=cue_lon, cue_alt=cue_alt,
                                x=r_vec[0], y=r_vec[1], z=r_vec[2],
                                vx=v_vec[0], vy=v_vec[1], vz=v_vec[2],
                                roll=att_models_dict[actor.name]._actor_attitude_deg[0],
                                pitch=att_models_dict[actor.name]._actor_attitude_deg[1],
                                yaw=att_models_dict[actor.name]._actor_attitude_deg[2])

                        log_combined(writer_combined,
                                     detection_id=whale.detection_id,
                                     target_id=whale_idx, tip_actor=whale.tip_actor, cue_actor=actor.name,
                                     tip_observation_date=whale.t_observed_tip, tip_confirmation_date=whale.t_confirmed_tip,
                                     cue_observation_date=t_datetime, cue_confirmation_date=None,
                                     tip_ai_decision=None, cue_ai_decision=None,
                                     true_label=whale.ai_class_true, correct=None,
                                     offnadir_deg=offnadir_cue_deg, gsd_m=gsd_cue, viewing_time=viewing_time_left,
                                     latency_observation=latency_observation, latency_confirmation=None,
                                     target_lat=whale.lat, target_lon=whale.lon, target_alt=whale.alt,
                                     cue_lat=cue_lat, cue_lon=cue_lon, cue_alt=cue_alt)

                        log_img(writer_img_gen, whale.detection_id, cue_lat, cue_lon, cue_alt, target_coord[0], target_coord[1],target_coord[2], t_datetime, dem_seed)

                    _clear_actor_task(actor.name, task_id, eo_tools_dict, att_models_dict)

                # CUE CONFIRMATION
                if whale.t_observed_cue != None and whale.state_confirming < 2 and t_datetime > (whale.t_observed_cue + timedelta(seconds=delay_confirmation_cue)):
                        if onboard_ai_cue:
                            whale.confirmed_cue, label_cue = cue_ai_decision(whale, cue_tpr, cue_tnr, rng_ai_cue)

                        else:
                            whale.confirmed_cue, label_cue = True, "whale"

                        whale.t_confirmed_cue = t_datetime
                        whale.state_confirming= 2

                        n_confirmed_cue += 1
                        # cleanup_idx.append(whale_idx)

                        if whale.confirmed_cue:
                            whale.ai_class_predicted="whale"
                            confirmed_targets_pos[whale_idx] = whale
                            n_confirmed_pos += 1

                        elif not whale.confirmed_cue:
                            whale.ai_class_predicted = "not-whale"
                            confirmed_targets_neg[whale_idx] = whale
                            n_confirmed_neg += 1

                        if whale.tip_actor != None and whale.t_observed_tip != None:
                            if t_datetime <= whale.t_observed_tip + timedelta(seconds=observation_time_limit):
                                latency_confirmation = (whale.t_confirmed_cue - whale.t_observed_tip).total_seconds()

                        elif whale.cue_actor != None and whale.t_observed_cue != None:
                            if t_datetime <= whale.t_observed_cue + timedelta(seconds=observation_time_limit):
                                latency_confirmation = (whale.t_confirmed_cue - whale.t_observed_cue).total_seconds()

                        else:
                            latency_confirmation = None



                        print(f"!! {actor.name}: Confirmed Target {whale_idx}={whale.ai_class_predicted} (actual={whale.ai_class_true})")

                        if logging:
                            log_cue(writer_cue,
                                    detection_id=whale.detection_id,
                                    target_id=whale_idx, cue_actor=actor.name,
                                    cue_observation_date=None, cue_confirmation_date=t_datetime,
                                    cue_ai_decision=label_cue, true_label=whale.ai_class_true,
                                    correct=(label_cue == whale.ai_class_true),
                                    offnadir_deg=None, gsd_m=None, viewing_time=None,
                                    latency_observation=None, latency_confirmation=latency_confirmation,
                                    slew_stab_time=None,
                                    target_lat=None, target_lon=None, target_alt=None,
                                    cue_lat=None, cue_lon=None, cue_alt=None,
                                    x=None, y=None, z=None, vx=None, vy=None, vz=None,
                                    roll=None, pitch=None, yaw=None)

                            log_combined(writer_combined,
                                         detection_id=whale.detection_id,
                                         target_id=whale_idx, tip_actor=whale.tip_actor, cue_actor=actor.name,
                                         tip_observation_date=None, tip_confirmation_date=None,
                                         cue_observation_date=None, cue_confirmation_date=t_datetime,
                                         tip_ai_decision=whale.ai_class_predicted if whale.confirmed_tip else None,
                                         cue_ai_decision=label_cue,
                                         true_label=whale.ai_class_true,
                                         correct=(label_cue == whale.ai_class_true),
                                         offnadir_deg=None, gsd_m=None, viewing_time=None,
                                         latency_observation=None, latency_confirmation=latency_confirmation,
                                         target_lat=None, target_lon=None, target_alt=None,
                                         cue_lat=None, cue_lon=None, cue_alt=None)

        if verbose == True and n_steps % print_interval == 0:
            cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
            cue_lat, cue_lon, cue_alt = float(cue_lat), float(cue_lon), float(cue_alt)

            print(f"\t\t{actor.name} | {t_datetime.isoformat()} | lat={cue_lat:.1f}, lon={cue_lon:.1f}, alt={cue_alt:.1f} | illuminated={cue_illuminated}")
            if cue_observed and observed_idx_cue is not None:
                whale_dbg = all_targets[observed_idx_cue]
                print(f"\t\tTarget: idx={observed_idx_cue} | off nadir angle={offnadir_cue_deg:.1f} | gsd={gsd_cue:.2f} | lat={whale_dbg.lat:.1f}, lon={whale_dbg.lon:.1f}, alt={whale_dbg.alt:.1f}")


    if n_steps % 100 == 0:
        gc.collect()  # Empty garbage

    t_mid = time.time()

    if plot_propagation and n_steps % plot_pyvista_interval == 0:

        update_plotter(pl,
                       earth_actor, earth_state,
                       sun_light, whales_poly, tasked_poly,
                       cloud_tip_sats, cloud_cue_sats,
                       tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
                       t_datetime, tip_positions, cue_positions,
                       all_targets, observed_targets_tip, tasked_targets, observed_targets_cue,
                       confirmed_targets_pos, confirmed_targets_neg,
                       FovPoints_tip, FovPoints_cue, step_text, n_steps)

        try:
            pl.write_frame()

        except:
            print(f"PyVista warning: skipped writing frame at {n_steps}")

    t_end = time.time()

    if n_steps % print_interval == 0:
        print(f"\t\t{n_steps} Time iteration: {t_mid - t_start:.2f} | Time plot: {t_end - t_mid:.2f} "
              f"| Simulation time: {int(elapsed_hours)}h {(int(elapsed_seconds) - int(elapsed_hours) * 3600) // 60}m {(int(elapsed_seconds) - int(elapsed_hours) * 3600) % 60}s")

    for whale in all_targets.values():
        whale.update_detection_id()

    # for actor in tip_actors + cue_actors:
    #     att_models_dict[actor.name].update_attitude(sim_step_seconds)

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)

    elapsed_seconds += sim_step_seconds
    elapsed_hours = elapsed_seconds / 3600
    n_steps += 1

t_sim_end = time.time()
runtime = t_sim_end - t_sim_start
hours_sim, rem = divmod(runtime, 3600)
minutes_sim, seconds_sim = divmod(rem, 60)

print(f"\nTotal simulation time: {int(hours_sim)}h {int(minutes_sim)}m {seconds_sim:.0f}s | "
      f"Time per iteration: {runtime/n_steps:.2f}s | plot_propagation {plot_propagation}")

runtime_per_hour = runtime / sim_duration_hours if sim_duration_hours > 0 else 0
runtime_per_day  = runtime / (sim_duration_hours / 24) if sim_duration_hours > 0 else 0

merge_tip_cue_combined("sim_output.xlsx")
if plot_propagation:
    close_plotter_safely(pl, sun_light=sun_light)
    pl = None
    sun_light = None

# =========================
# --- Report (ordered) ---
# =========================
print(f"\n\n---------------------- Mission Summary {sim_name} -----------------------------------\n")
# --- System setup ---
n_sats_total = nSats_tip + nSats_cue
print(f"Number of satellites:                       {n_sats_total} (Tip={nSats_tip}, Cue={nSats_cue})")
print(f"Total targets:                              {n_targets} (positive={n_targets_pos}, negative={n_targets_neg})")
print(f"Simulation time:                            {sim_duration_hours} h\n")

# --- Coverage ---
n_float_tip, n_full_tip, residual_tip, T_tip = count_orbits_completed(a_tip, sim_duration_seconds)
n_float_cue, n_full_cue, residual_cue, T_cue = count_orbits_completed(a_cue, sim_duration_seconds)

print(f"Tip: orbits completed                       {n_float_tip:.3f} "
      f"(full={n_full_tip}, residual={residual_tip:.2f}s, period={T_tip:.1f}s)")
print(f"Cue: orbits completed                       {n_float_cue:.3f} "
      f"(full={n_full_cue}, residual={residual_cue:.2f}s, period={T_cue:.1f}s)\n")

if plot_footprints:
    (area_total_km2, area_mission_km2, area_covered_km2, area_covered_per_orbit_km2,
     area_mission_fraction_total, area_covered_fraction_total, area_covered_fraction_mission,
     area_covered_per_orbit_fraction_total, area_covered_per_orbit_fraction_mission) = compute_coverage_fraction(
        fov_polygons_tip, fov_polygons_cue, R_earth, i_tip_deg, a_tip, sim_duration_seconds)

    print(f"Total coverage:                             {area_covered_km2:,.0f} km² "
          f"({area_covered_fraction_total * 100:.2f}% Earth, "
          f"{area_covered_fraction_mission * 100:.2f}% mission)")
    print(f"Per orbit coverage:                         {area_covered_per_orbit_km2:,.0f} km² "
          f"({area_covered_per_orbit_fraction_total * 100:.2f}% Earth, "
          f"{area_covered_per_orbit_fraction_mission * 100:.2f}% mission)\n")

# --- TIP/CUE process flow ---
print(f"Tip observed:                               {n_observed_tip} (unique: {len(observed_targets_tip)})")
print(f"Tip confirmed:                              {n_confirmed_tip}")
print(f"Tip tasks sent:                             {n_tasked_tip}")
print(f"Cue tasks started:                          {n_tasked_cue}")
print(f"Cue observed:                               {n_observed_cue} (unique: {len(observed_targets_cue)})")
print(f"Cue confirmed:                              {n_confirmed_cue}\n")

# --- Latency / off-nadir ---
if logging:
    df_cue = pd.read_excel("sim_output.xlsx", sheet_name="Cue")

    avg_offnadir_deg,  min_offnadir_deg,  max_offnadir_deg,  std_offnadir_deg  = compute_stats(df_cue["offnadir_deg"])
    avg_gsd,  min_gsd,  max_gsd,  std_gsd  = compute_stats(df_cue["gsd_m"])

    avg_latency_obs_s, min_latency_obs_s, max_latency_obs_s, std_latency_obs_s = compute_stats(df_cue["latency_observation"])
    avg_latency_conf_s, min_latency_conf_s, max_latency_conf_s, std_latency_conf_s = compute_stats(df_cue["latency_confirmation"])
    avg_viewing_time, min_viewing_time, max_viewing_time, std_viewing_time = compute_stats(df_cue["viewing_time"])

    print(f"Average off-nadir angle:                    {avg_offnadir_deg:.2f}° "
          f"(min {min_offnadir_deg:.2f}°, max {max_offnadir_deg:.2f}°, std {std_offnadir_deg:.2f}°)")
    print(f"Average GSD:                    {avg_gsd:.3f} m "
          f"(min {min_gsd:.3f}, max {max_gsd:.3f}, std {std_gsd:.3f} m)\n")

    print(f"Average viewing time:                       {avg_viewing_time:.1f} s "
          f"(min {min_viewing_time:.1f}, max {max_viewing_time:.1f}, std {std_viewing_time:.1f} s)")
    print(f"Average latency (observation):              {avg_latency_obs_s:.1f} s "
          f"(min {min_latency_obs_s:.1f}, max {max_latency_obs_s:.1f}, std {std_latency_obs_s:.1f} s)")
    print(f"Average latency (confirmation):             {avg_latency_conf_s:.1f} s "
          f"(min {min_latency_conf_s:.1f}, max {max_latency_conf_s:.1f}, std {std_latency_conf_s:.1f} s)\n")


# --- Outcome summary ---
print(f"Positive confirmed:                         {n_confirmed_pos} (unique: {len(confirmed_targets_pos)}, actual={n_targets_pos})")
print(f"Negative confirmed:                         {n_confirmed_neg} (unique: {len(confirmed_targets_neg)}, actual={n_targets_neg})\n")

# --- Throughput ---
tp = fp = tn = fn = 0
for whale in all_targets.values():
    if whale.ai_class_predicted is None:
        continue
    is_true = (whale.ai_class_true == "whale")
    is_pred = (whale.ai_class_predicted == "whale")
    if  is_true and  is_pred: tp += 1
    if not is_true and  is_pred: fp += 1
    if not is_true and not is_pred: tn += 1
    if  is_true and not is_pred: fn += 1

print(f"Throughput (True Positives):                {tp}")
print(f"Throughput (Confirmations):                 {n_confirmed_cue}\n")

# --- Confusion matrix ---
print(f"True Positives (TP):                        {tp}")
print(f"False Positives (FP):                       {fp}")
print(f"True Negatives (TN):                        {tn}")
print(f"False Negatives (FN):                       {fn}\n")

# --- Derived classification metrics ---
prec_den = (tp + fp) if (tp + fp) > 0 else 1
rec_den  = (tp + fn) if (tp + fn) > 0 else 1
tn_den   = (tn + fp) if (tn + fp) > 0 else 1
fnr_den  = (tp + fn) if (tp + fn) > 0 else 1

precision = tp / prec_den
recall    = tp / rec_den
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
false_alarm_rate = fp / tn_den
miss_rate        = fn / fnr_den

print(f"Precision:                                  {precision*100:.2f}%")
print(f"Recall (sensitivity):                       {recall*100:.2f}%")
print(f"F1-score:                                   {f1*100:.2f}%")
print(f"False Alarm Rate:                           {false_alarm_rate*100:.2f}%")
print(f"Miss Rate:                                  {miss_rate*100:.2f}%\n")

# --- Process efficiencies ---
observation_efficiency_tip = n_observed_tip / n_targets if n_targets > 0 else 0
observation_efficiency_cue = n_observed_cue / n_targets if n_targets > 0 else 0
confirmation_efficiency    = n_confirmed_cue / n_targets if n_targets > 0 else 0

tip_task_util   = (n_tasked_tip / n_confirmed_tip) if n_confirmed_tip > 0 else 0
cue_success_rate = (n_confirmed_cue / n_tasked_cue) if n_tasked_cue > 0 else 0

print(f"Tip Observation efficiency:                 {observation_efficiency_tip * 100:.2f}% of whales")
print(f"Cue Observation efficiency:                 {observation_efficiency_cue * 100:.2f}% of whales")
print(f"Confirmation efficiency Cue:                {confirmation_efficiency * 100:.2f}% of whales")
print(f"Tip Task Utilization:                       {tip_task_util*100:.2f}% (Tip tasks / Tip confirmations)")
print(f"Cue Success Rate:                           {cue_success_rate*100:.2f}% (Cue confirmations / Cue tasks)\n")

# --- Mission efficiency ---
duration_days = sim_duration_seconds / 86400.0 if sim_duration_seconds > 0 else 0
tp_per_sat           = tp / n_sats_total if n_sats_total > 0 else 0
tp_per_orbit         = tp / (n_full_tip + n_full_cue) if (n_full_tip + n_full_cue) > 0 else 0
tp_per_day           = tp / duration_days if duration_days > 0 else 0
tp_per_sat_per_orbit = tp_per_orbit / n_sats_total if n_sats_total > 0 else 0
tp_per_sat_per_day   = tp_per_day / n_sats_total if n_sats_total > 0 else 0

conf_per_sat_all           = n_confirmed_cue / n_sats_total if n_sats_total > 0 else 0
conf_per_orbit_all         = n_confirmed_cue / (n_full_tip + n_full_cue) if (n_full_tip + n_full_cue) > 0 else 0
conf_per_day_all           = n_confirmed_cue / duration_days if duration_days > 0 else 0
conf_per_sat_per_orbit_all = conf_per_orbit_all / n_sats_total if n_sats_total > 0 else 0
conf_per_sat_per_day_all   = conf_per_day_all / n_sats_total if n_sats_total > 0 else 0

cue_tasks_per_sat = n_tasked_cue / nSats_cue if nSats_cue > 0 else 0
obs_per_cue_sat   = n_observed_cue / nSats_cue if nSats_cue > 0 else 0
conf_per_cue_sat  = n_confirmed_cue / nSats_cue if nSats_cue > 0 else 0

print(f"Tasks per Cue satellite:                    {cue_tasks_per_sat:.2f} tasks/sat")
print(f"Observations per Cue satellite:             {obs_per_cue_sat:.2f} targets/sat")
print(f"Confirmations per Cue satellite:            {conf_per_cue_sat:.2f} targets/sat\n")

print(f"True Positives per satellite (all):         {tp_per_sat:.2f} TP/sat")
print(f"True Positives per orbit (all sats):        {tp_per_orbit:.2f} TP/orbit")
print(f"True Positives per day (all sats):          {tp_per_day:.2f} TP/day")
print(f"True Positives per satellite per orbit:     {tp_per_sat_per_orbit:.2f} TP/sat/orbit")
print(f"True Positives per satellite per day:       {tp_per_sat_per_day:.2f} TP/sat/day\n")

print(f"Confirmations per satellite (all):          {conf_per_sat_all:.2f} conf/sat")
print(f"Confirmations per orbit (all sats):         {conf_per_orbit_all:.2f} conf/orbit")
print(f"Confirmations per day (all sats):           {conf_per_day_all:.2f} conf/day")
print(f"Confirmations per satellite per orbit:      {conf_per_sat_per_orbit_all:.2f} conf/sat/orbit")
print(f"Confirmations per satellite per day:        {conf_per_sat_per_day_all:.2f} conf/sat/day\n")

# --- Runtime ---
print(f"Total runtime (wall-clock):                 {format_hms(runtime)}")
print(f"Runtime per simulation hour:                {format_hms(runtime_per_hour)}")
print(f"Runtime per simulation day:                 {format_hms(runtime_per_day)}\n")


# --- Write Overview log row-wise with blank lines ---
if logging:
    wb = openpyxl.load_workbook("sim_output.xlsx")
    ws = wb["Overview"]

    # Clear old rows (keep header)
    ws.delete_rows(2, ws.max_row)

    overview_data = [
        # --- System setup ---
        ("Number of satellites", n_sats_total, f"(Tip={nSats_tip}, Cue={nSats_cue})"),
        ("Total targets", n_targets, f"(positive={n_targets_pos}, negative={n_targets_neg})"),
        ("Simulation time (h)", sim_duration_hours, ""),
        ("", "", ""),

        # --- Coverage ---
        ("Tip orbits completed", round(n_float_tip, 3),
         f"(full={n_full_tip}, residual={residual_tip:.2f}s, period={T_tip:.1f}s)"),
        ("Cue orbits completed", round(n_float_cue, 3),
         f"(full={n_full_cue}, residual={residual_cue:.2f}s, period={T_cue:.1f}s)"),
        ("Total coverage (km²)", round(area_covered_km2, 1) if area_covered_km2 else "",
         f"({area_covered_fraction_total * 100:.2f}% Earth, {area_covered_fraction_mission * 100:.2f}% mission)" if area_covered_fraction_total else ""),
        ("Coverage per orbit (km²)", round(area_covered_per_orbit_km2, 1) if area_covered_per_orbit_km2 else "",
         f"({area_covered_per_orbit_fraction_total * 100:.2f}% Earth, {area_covered_per_orbit_fraction_mission * 100:.2f}% mission)" if area_covered_per_orbit_fraction_total else ""),
        ("", "", ""),

        # --- TIP/CUE process flow ---
        ("Tip observed", n_observed_tip, f"(unique: {len(observed_targets_tip)})"),
        ("Tip confirmed", n_confirmed_tip, ""),
        ("Tip tasks sent", n_tasked_tip, ""),
        ("Cue tasks started", n_tasked_cue, ""),
        ("Cue observed", n_observed_cue, f"(unique: {len(observed_targets_cue)})"),
        ("Cue confirmed", n_confirmed_cue, ""),
        ("", "", ""),

        # --- Latency / off-nadir ---
        ("Average off-nadir angle (deg)", round(avg_offnadir_deg, 2),
         f"(min {min_offnadir_deg:.2f}, max {max_offnadir_deg:.2f}, std {std_offnadir_deg:.2f})"),
        ("Average GSD (m)", round(avg_gsd, 3),
         f"(min {min_gsd:.3f}, max {max_gsd:.3f}, std {std_gsd:.3f})"),
        ("", "", ""),

        ("Average viewing time (s)", round(avg_viewing_time, 2),
         f"(min {min_viewing_time:.2f}, max {max_viewing_time:.2f}, std {std_viewing_time:.2f})"),
        ("Average latency, observation (s)", round(avg_latency_obs_s, 1),
         f"(min {min_latency_obs_s:.1f}, max {max_latency_obs_s:.1f}, std {std_latency_obs_s:.1f})"),
        ("Average latency, confirmation (s)", round(avg_latency_conf_s, 1),
         f"(min {min_latency_conf_s:.1f}, max {max_latency_conf_s:.1f}, std {std_latency_conf_s:.1f})"),

        ("", "", ""),

        # --- Outcome summary ---
        ("Positive confirmed", n_confirmed_pos, f"(unique: {len(confirmed_targets_pos)}, actual: {n_targets_pos})"),
        ("Negative confirmed", n_confirmed_neg, f"(unique: {len(confirmed_targets_neg)}, actual: {n_targets_neg})"),
        ("Throughput (True Positives)", tp, ""),
        ("Throughput (Confirmations)", n_confirmed_cue, ""),
        ("", "", ""),

        # --- Confusion matrix ---
        ("True Positives (TP)", tp, ""),
        ("False Positives (FP)", fp, ""),
        ("True Negatives (TN)", tn, ""),
        ("False Negatives (FN)", fn, ""),
        ("", "", ""),

        # --- Derived classification metrics ---
        ("Precision (%)", round(precision * 100, 2), ""),
        ("Recall (sensitivity, %)", round(recall * 100, 2), ""),
        ("F1-score (%)", round(f1 * 100, 2), ""),
        ("False Alarm Rate (%)", round(false_alarm_rate * 100, 2), ""),
        ("Miss Rate (%)", round(miss_rate * 100, 2), ""),
        ("", "", ""),

        # --- Process efficiencies ---
        ("Tip Observation efficiency (%)", round(observation_efficiency_tip * 100, 2),
         f"of whales ({n_targets} total)"),
        ("Cue Observation efficiency (%)", round(observation_efficiency_cue * 100, 2),
         f"of whales ({n_targets} total)"),
        ("Confirmation efficiency Cue (%)", round(confirmation_efficiency * 100, 2),
         f"of whales ({n_targets} total)"),
        ("Tip Task Utilization (%)", round(tip_task_util * 100, 2), "Tip tasks / Tip confirmations"),
        ("Cue Success Rate (%)", round(cue_success_rate * 100, 2), "Cue confirmations / Cue tasks"),
        ("", "", ""),

        # --- Mission efficiency ---
        ("Tasks per Cue satellite", round(cue_tasks_per_sat, 2), ""),
        ("Observations per Cue satellite", round(obs_per_cue_sat, 2), ""),
        ("Confirmations per Cue satellite", round(conf_per_cue_sat, 2), ""),

        ("True Positives per satellite (all)", round(tp_per_sat, 2), ""),
        ("True Positives per orbit (all sats)", round(tp_per_orbit, 2), ""),
        ("True Positives per day (all sats)", round(tp_per_day, 2), ""),
        ("True Positives per satellite per orbit", round(tp_per_sat_per_orbit, 2), ""),
        ("True Positives per satellite per day", round(tp_per_sat_per_day, 2), ""),

        ("Confirmations per satellite (all)", round(conf_per_sat_all, 2), ""),
        ("Confirmations per orbit (all sats)", round(conf_per_orbit_all, 2), ""),
        ("Confirmations per day (all sats)", round(conf_per_day_all, 2), ""),
        ("Confirmations per satellite per orbit", round(conf_per_sat_per_orbit_all, 2), ""),
        ("Confirmations per satellite per day", round(conf_per_sat_per_day_all, 2), ""),
        ("", "", ""),

        # --- Runtime ---
        ("Total runtime (wall-clock)", format_hms(runtime), ""),
        ("Runtime per simulation hour", format_hms(runtime_per_hour), ""),
        ("Runtime per simulation day", format_hms(runtime_per_day), ""),
    ]

    for row in overview_data:
        ws.append(row)

    # Auto-adjust column widths
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max_length + 2

    wb.save("sim_output.xlsx")
    time.sleep(0.1)

    if os.path.exists("sim_output.xlsx"):
        plot_offnadir_distribution("sim_output.xlsx", bin_size_deg=2.5)
        plot_gsd_distribution("sim_output.xlsx", bin_size_m=0.05)
        plot_latency_distribution("sim_output.xlsx", 'latency_observation', bin_size_sec=15)
        plot_latency_distribution("sim_output.xlsx", 'latency_confirmation', bin_size_sec=15)
        plot_viewing_time_distribution("sim_output.xlsx", 'viewing_time', bin_size_sec=15)
        if verbose:
            print("Created offnadir and latency distribution plots")

print("\n")
at_exit(save_name=sim_name, pl=pl, sun_light=sun_light, verbose_def=False, verbose_error=False)


if plot_footprints:
    print(f"\n\n\tGenerate footprint plots with len {len(fov_polygons_cue)}")
    t1 = time.time()

    fov_polygons_tip = [f for f in fov_polygons_tip if f is not None]
    fov_polygons_cue = [f for f in fov_polygons_cue if f is not None]

    t2 = time.time()

    if len(fov_polygons_tip) > 0:
        plot_all_fov_footprints_plotly(fov_polygons_tip, all_targets, observed_targets_tip, nPlanes_tip, nSats_tip, extension="tip", plot_whale_trajectories=plot_whale_trajectories, whale_trajectories=whale_trajectories)

    if len(fov_polygons_cue) > 0:
        plot_all_fov_footprints_plotly(fov_polygons_cue, all_targets, observed_targets_cue, nPlanes_cue, nSats_cue, extension="cue", plot_whale_trajectories=plot_whale_trajectories, whale_trajectories=whale_trajectories)

    t3 = time.time()
    print(f"\tFOV: footprints plotting time: {format_hms(t3 - t2)}\n ")

if show_orbits:
    plot_orbits(trajectories)

at_exit(save_name=sim_name, pl=pl, sun_light=sun_light, verbose_def=False, verbose_error=False)



