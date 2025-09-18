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

from orekit.pyhelpers import setup_orekit_curdir

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from custom_paseos.utils.point_transformation import Point_ECI2Geodetic, Point_Geodetic2ECI

from simulation.propagate_whales import update_whales, load_land_mask, generate_random_water_targets, init_whales, build_land_mask
from simulation.simulation_functions import init_eo_tools, init_attitude_models, link_eo_attitude, cleanup_timeout_targets, propagate_actor, satellite_in_shadow, daylight_mask, convert_M_to_lv, pointing_cost, count_orbits_completed, compute_coverage_fraction, _clear_actor_task
from simulation.plotting.plot_functions import plot_orbits, plot_all_fov_footprints_plotly, plot_offnadir_distribution, plot_latency_distribution
from simulation.plotting.plot_pyvista import make_plotter_eci, reset_plotter, update_plotter, compute_movie_framerate
from simulation.plotting.plot_constellation import plot_constellation_pyvista_plain
from simulation.sim_logging import init_excel_log, log_tip_observation, log_cue_observation, log_combined_observation, gsd_offnadir, at_exit, Logger, compute_stats
from simulation.onboard_ai.onboard_ai_tip import tip_ai_decision
from simulation.onboard_ai.onboard_ai_cue import cue_ai_decision

show_constellation = False
show_orbits = False
plot_propagation = True # True
plot_footprints = True
plot_whale_trajectories = True

generate_image = False
onboard_ai_tip = True
onboard_ai_cue = True
attitude_mode = 'pause'     # 'pause' or 'planned'

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

# Time setup
# Redirect both stdout and stderr
if logging:
    sys.stdout = Logger("output.log")
    sys.stderr = sys.stdout

print(f"Initiate simulation {sim_name} | Logging {logging}")

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
    # plot_constallation(planet_lst_tip, planet_lst_cue)
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
eul_ang_cue_target = eul_ang_cue_default
offnadir_cue_deg_target = None
offnadir_tip_deg = 0.0
offnadir_unbound = 0.0

n_targets_pos = int(round(n_targets * pos_fraction))
n_targets_neg = n_targets - n_targets_pos

# EO Tools
eo_tools_dict = init_eo_tools(tip_actors, cue_actors, fov_tip, fov_cue, offnadir_limit)
att_models_dict = init_attitude_models(tip_actors, cue_actors, eul_ang_tip_default, eul_ang_cue_default)
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
    pl, earth_actor, earth_state = make_plotter_eci()

    (earth_actor, earth_state, sun_light,
     whales_plot_all, whales_plot_observed_tip, whales_plot_tasked, whales_plot_observed_cue, whales_plot_confirmed_pos, whales_plot_confirmed_neg,
     cloud_tip_sats, cloud_cue_sats, tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
     obs_tip_pts, task_pts, obs_cue_pts, conf_pts_pos, conf_pts_neg, step_text) = reset_plotter(pl, all_targets, n_targets, tip_actors, cue_actors, last_theta=None)

    pl.show(cpos="xy", interactive_update=True, auto_close=False)
    pv_framerate, frames_per_orbit = compute_movie_framerate(a_cue, sim_step_seconds, plot_pyvista_interval, movie_orbit_sec)
    pl.open_movie( "simulation.mp4",  framerate=pv_framerate)

if logging:
    header_tip = ["target_id", "tip_observation_date", "tip_actor", "offnadir_deg", "GSD_m", "target_lat", "target_lon", "target_alt",
                  "tip_lat", "tip_lon", "tip_alt", "x", "y", "z", "vx", "vy", "vz", "tip_observation_counter"]

    header_cue = ["target_id", "cue_observation_date", "cue_actor", "offnadir_deg", "GSD_m", "target_lat", "target_lon", "target_alt",
                  "cue_lat", "cue_lon", "cue_alt", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "cue_observation_counter"]

    header_combined = ["target_id", "tip_observation_date", "tip_actor", "cue_observation_date", "cue_actor", "offnadir_deg", "GSD_m", "latency", "target_lat", "target_lon", "target_alt",
                  "cue_lat", "cue_lon", "cue_alt", "tip_observation_counter", "cue_observation_counter"]

    header_overview = ["Metric", "Value", "Comment"]

    writer_overview = init_excel_log("sim_output.xlsx", header_overview, sheet_name="Overview")
    writer_combined = init_excel_log("sim_output.xlsx", header_combined, sheet_name="CombinedLog")
    writer_tip = init_excel_log("sim_output.xlsx", header_tip, sheet_name="TipLog")
    writer_cue = init_excel_log("sim_output.xlsx", header_cue, sheet_name="CueLog")

    results_dir = os.path.join("results", sim_name)
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

    atexit.register(at_exit, save_name=sim_name, pl=pl if plot_propagation else None)
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
    t_datetime = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=t_pykep.mjd2000)
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
        print(f"!! Removed Target {idx} tasking request")

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

                if in_footprint and whale.state_observing != 1:

                    print(f"!! {actor.name}: Observed Target {whale_idx}")

                    whale.tip_actor = actor.name
                    whale.t_observed_tip = t_datetime
                    whale.delay_confirmation_tip = delay_confirmation_tip
                    whale.state_observing = 1
                    whale.coord_observed = whale.position()

                    observed_targets_tip[whale_idx] = whale
                    observed_idx_tip = whale_idx
                    tip_observed = True
                    n_observed_tip += 1

                    if onboard_ai_tip:
                        whale.confirmed_tip = tip_ai_decision(whale, tip_tpr, tip_tnr, seed_tip)

                    else:
                        whale.confirmed_tip = True



                    h_m = float(np.linalg.norm(np.array(r))) - R_earth
                    gsd_tip = gsd_offnadir(gsd0_tip, h_m, offnadir_tip_deg)

                    if logging:
                        tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r[0], r[1], r[2], t_datetime).flatten()
                        log_tip_observation(writer_tip, whale_idx, t_datetime, actor.name, offnadir_tip_deg, gsd_tip, target_coord[0], target_coord[1], target_coord[2], tip_lat,
                                          tip_lon, tip_alt, r[0], r[1], r[2], v[0], v[1], v[2], tip_observation_counter=1)


                if whale.t_observed_tip != None and whale.state_confirming < 1 and t_datetime > (whale.t_observed_tip + timedelta(seconds=delay_confirmation_tip)):

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

                            print(f"!! {actor.name}: Confirmed Target {whale_idx}={whale.ai_class_predicted}, no Cue assigned (actual={whale.ai_class_true})")

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
                                delta_t_cue, el_min_deg=elevation_min, step=60.0
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

                        # r_future, v_future = eo_tools_dict[actor.name]._kepler_propagate_universal(r_vec, v_vec, sim_step_seconds)
                        # t_future = t_datetime + timedelta(seconds=sim_step_seconds)

                        pointing_vec_brf_target, offnadir_cue_deg_target, offnadir_unbound, time_to_sight = eo_tools_dict[actor.name].point_to_target_bounded(r_eci=r_vec, v_eci=v_vec, target_geodetic=task_coord, t_datetime=t_datetime, offnadir_max=offnadir_limit, mode='max')
                        eo_tools_dict[actor.name].offnadir_unbound_target = offnadir_unbound

                        if eo_tools_dict[actor.name].offnadir_unbound_target == None or eo_tools_dict[actor.name].offnadir_unbound_target - offnadir_unbound >= 0 or offnadir_unbound > 60.0:     # Compare wrt previous state or initial state is unset
                            moving_toward = True
                        else:
                            moving_toward = False

                        will_be_in_view, _ = eo_tools_dict[actor.name].will_be_visible_within(task_coord, r_vec, v_vec, t_datetime, delta_t_cue, el_min_deg=elevation_min, step=60.0)

                        if (in_view or will_be_in_view) and not (offnadir_unbound >= (offnadir_limit + offnadir_margin) and not moving_toward):

                            offnadir_cue_current, _ = att_models_dict[actor.name].offnadir_from_euler(att_models_dict[actor.name]._actor_attitude_deg)
                            eul_ang_cue_target = att_models_dict[actor.name].pointing_attitude_brf(pointing_vec_brf_target)

                        else:
                            # Task finished → reset and pick next later
                            print(f"!! {actor.name}: Target {task_id} out of view, delete task")

                            # reset local EO tool state
                            offnadir_cue_deg_target = 0.0
                            eul_ang_cue_target = eul_ang_cue_default
                            _clear_actor_task(actor.name, task_id, eo_tools_dict, att_models_dict, eul_ang_cue_default)
                            whale.t_observed_tip = None

                            # Global removal happens at the next loop via cleanup_timeout_targets:
                            if task_id in tasked_targets:
                                cleanup_idx.append(task_id)

            if eo_tools_dict[actor.name].current_task is None:
                if not np.allclose(att_models_dict[actor.name]._target_attitude_deg, eul_ang_cue_default, atol=0.1):
                    offnadir_cue_deg_target = 0.0
                    eul_ang_cue_target = eul_ang_cue_default
                    print(f"\t\t {n_steps}!! {actor.name}: Set roll, pitch, yaw target back to default {eul_ang_cue_default[0]:.1f}, {eul_ang_cue_default[1]:.1f}, {eul_ang_cue_default[2]:.1f} deg")


                    # eo_tools_dict[actor.name].offnadir_unbound_target = None

            # --- Handle new target attitude ---
            if not np.allclose(
                    eul_ang_cue_target,
                    att_models_dict[actor.name]._target_attitude_deg,
                    atol=0.1,
            ):

                if attitude_mode == "pause":
                    # Predict slew + stabilization time (bookkeeping)
                    delay_slew_stab, delay_slew, delay_stab = (
                        att_models_dict[actor.name].get_pointing_stabilization_time(
                            current_eul=att_models_dict[actor.name]._actor_attitude_deg,
                            target_eul=eul_ang_cue_target,
                            omega_max_rad=omega_max_rad,
                            alpha_max_rad=alpha_max_rad,
                            zeta=zeta,
                            wn_rad=wn_rad,
                            mode="per_axis",
                            current_w_rad=att_models_dict[actor.name]._actor_angular_velocity,
                            current_a_rad=att_models_dict[actor.name]._actor_angular_acceleration,
                        )
                    )

                    if att_models_dict[actor.name].t_eul_commanded is not None and np.allclose(
                            eul_ang_cue_target,
                            att_models_dict[actor.name]._target_attitude_deg,
                            atol=10.0,
                    ):

                        # Current Euler angles (deg)
                        eul_current_target = np.array(att_models_dict[actor.name]._target_attitude_deg)

                        # New target Euler angles (deg)
                        eul_new_target = np.array(eul_ang_cue_target)

                        # Angular delta (deg)
                        delta_eul = np.linalg.norm(eul_new_target - eul_current_target)

                        # Translate angular delta into additional stabilization time
                        # Use per-axis or worst-axis time model from your get_pointing_stabilization_time
                        _, _, delay_stab_extra = att_models_dict[actor.name].get_pointing_stabilization_time(
                            current_eul=eul_current_target,
                            target_eul=eul_new_target,
                            omega_max_rad=omega_max_rad,
                            alpha_max_rad=alpha_max_rad,
                            zeta=zeta,
                            wn_rad=wn_rad,
                            mode="per_axis"
                        )

                        delay_extra = delta_eul/np.rad2deg(omega_max_rad) + delay_stab_extra            # to cover extra distance + stabilize

                        # Add that as the adjustment
                        att_models_dict[actor.name].delay_slew_stab = (
                                att_models_dict[actor.name].delay_slew_stab + (delay_extra *sim_step_seconds)    # Only one way
                        )           # delay_extra returns t_slew, so * 0.25

                        att_models_dict[actor.name].delay_slew_stab = np.max([att_models_dict[actor.name].delay_slew_stab, delay_slew_stab])
                        task_update_mode = "updated"
                    else:
                        att_models_dict[actor.name].t_eul_commanded = t_datetime
                        att_models_dict[actor.name].delay_slew_stab = delay_slew_stab
                        task_update_mode = "taken"

                        print(
                            f"\t\t!! {n_steps}  {actor.name}: Target {task_id} in reach "
                            f"(current={offnadir_cue_current:.1f} deg, target={offnadir_cue_deg_target:.1f} deg) "
                            f"| Roll, pitch, yaw current={att_models_dict[actor.name]._actor_attitude_deg} deg "
                            f"| set to target={eul_ang_cue_target} deg "
                            f"| Setting time {delay_slew_stab:.1f} s "
                            f"(slew {delay_slew:.1f} s, stab {delay_stab:.1f} s) | update mode {task_update_mode}"
                        )

                    att_models_dict[actor.name].set_target_euler(eul_ang_cue_target)



                elif attitude_mode == "planned":
                    # Only replan if target changed significantly OR no active plan
                    if (not np.allclose(eul_ang_cue_target,
                                        att_models_dict[actor.name]._target_attitude_deg,
                                        atol=10.0)
                            or not att_models_dict[actor.name].slew_active):
                        delay_slew_stab, delay_slew, delay_stab = (
                            att_models_dict[actor.name].get_pointing_stabilization_time(
                                current_eul=att_models_dict[actor.name]._actor_attitude_deg,
                                target_eul=eul_ang_cue_target,
                                omega_max_rad=omega_max_rad,
                                alpha_max_rad=alpha_max_rad,
                                zeta=zeta,
                                wn_rad=wn_rad,
                                mode="per_axis",
                                current_w_rad=att_models_dict[actor.name]._actor_angular_velocity,
                                current_a_rad=att_models_dict[actor.name]._actor_angular_acceleration,
                            )
                        )

                        att_models_dict[actor.name].delay_slew_stab = delay_slew_stab
                        att_models_dict[actor.name].slew_active = True
                        task_update_mode = "taken" if not att_models_dict[actor.name].slew_active else "replanned"

                        att_models_dict[actor.name].plan_slew(
                            target_eul_deg=eul_ang_cue_target,
                            omega_max_rad=omega_max_rad,
                            alpha_max_rad=alpha_max_rad,
                            dt=sim_step_seconds,
                            mode="per_axis",
                            t_start=elapsed_seconds,
                        )
                        att_models_dict[actor.name].set_target_euler(eul_ang_cue_target)

                        print(
                            f"\t\t!! {n_steps}  {actor.name}: Target {task_id} in reach "
                            f"(current={offnadir_cue_current:.1f} deg, target={offnadir_cue_deg_target:.1f} deg) "
                            f"| Roll, pitch, yaw current={att_models_dict[actor.name]._actor_attitude_deg} deg "
                            f"| set to target={eul_ang_cue_target} deg "
                            f"| Planned trajectory time {delay_slew_stab:.1f} s "
                            f"(slew {delay_slew:.1f} s, stab {delay_stab:.1f} s) | update mode {task_update_mode}"
                        )
                    else:
                        # Still chasing → no new plan, just continue
                        print(
                            f"\t\t!! {n_steps}  {actor.name}: Target {task_id} in reach "
                            f"(continuing planned slew -> chasing mode)"
                        )

                elif attitude_mode == "physics":
                    att_models_dict[actor.name].set_target_euler(eul_ang_cue_target)
                    print(
                        f"\t\t!! {n_steps}  {actor.name}: Physics mode → set new target "
                        f"{eul_ang_cue_target}, current={att_models_dict[actor.name]._actor_attitude_deg}"
                    )

            # --- Advance or complete the commanded motion ---
            if attitude_mode == "pause":
                if att_models_dict[actor.name].t_eul_commanded is not None:
                    if t_datetime >= (
                            att_models_dict[actor.name].t_eul_commanded
                            + timedelta(seconds=att_models_dict[actor.name].delay_slew_stab)
                    ):
                        att_models_dict[actor.name].set_actor_euler(
                            att_models_dict[actor.name]._target_attitude_deg
                        )
                        att_models_dict[actor.name].t_eul_commanded = None
                        att_models_dict[actor.name].delay_slew_stab = None
                        print(f"\t\t !! {n_steps} {actor.name}: Completed move (pause mode)")

            elif attitude_mode == "planned":
                if att_models_dict[actor.name].slew_active:
                    att_models_dict[actor.name].follow_planned_slew(elapsed_seconds)

                    if (
                            np.allclose(
                                att_models_dict[actor.name]._actor_attitude_deg,
                                att_models_dict[actor.name]._target_attitude_deg,
                                atol=0.5,
                            )
                            and np.allclose(
                        eul_ang_cue_target,
                        att_models_dict[actor.name]._target_attitude_deg,
                        atol=0.5,
                    )
                    ):
                        att_models_dict[actor.name].delay_slew_stab = None
                        att_models_dict[actor.name].slew_active = False
                        print(f"\t\t !! {n_steps} {actor.name}: Completed move (planned trajectory)")

            elif attitude_mode == "physics":
                att_models_dict[actor.name].update_attitude(sim_step_seconds)

            #
              #   if verbose and n_steps % print_interval == 0:
              #       delta_move_eul = eul_new_deg - prev_eul
              #       print(f"\t\t {actor.name} | Delta roll, pitch, yaw={delta_move_eul[0]:.1f}, {delta_move_eul[1]:.1f}, {delta_move_eul[2]:.1f} deg")

        try:
            # check off-nadir angle, and where the center ray intersects the Earth
            if attitude_mode == 'planned' or (np.all(att_models_dict[actor.name]._actor_angular_velocity) <= 0.1 and np.all(att_models_dict[actor.name]._actor_angular_acceleration) <= 0.1):
                FovPoints = eo_tools_dict[actor.name].get_FovPoints(r_vec, v_vec, t_datetime)
                FovPoints_cue.append(FovPoints)

            else:
                FovPoints=None

            if plot_footprints and n_steps % plot_fov_interval == 0:
                fov_polygons_cue[footprint_idx_cue] = FovPoints
                footprint_idx_cue += 1

        except:
            print(f"!! {actor.name}: no FOV intersection, continue to the next step")
            offnadir_cue_deg_target = 0.0
            att_models_dict[actor.name].set_target_euler(eul_ang_cue_default)
            eo_tools_dict[actor.name].offnadir_unbound_target = None

            continue

        if cue_illuminated and (np.all(att_models_dict[actor.name]._actor_angular_velocity) <= 0.1 and np.all(att_models_dict[actor.name]._actor_angular_acceleration) <= 0.1):         # only observe if stable

            for whale_idx, whale in all_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                target_coord = whale.position()
                in_footprint = eo_tools_dict[actor.name].check_point_in_footprint(target_coord, FovPoints)
                offnadir_cue_deg, _ = att_models_dict[actor.name].offnadir_from_euler(att_models_dict[actor.name]._actor_attitude_deg)

                if in_footprint and whale.state_observing != 2 and offnadir_cue_deg <= (offnadir_limit + offnadir_margin) and att_models_dict[actor.name].t_eul_commanded == None:

                    print(f"!! {n_steps} {actor.name}: Observed Target {whale_idx} at off-nadir {offnadir_cue_deg:.1f} deg")

                    whale.cue_actor = actor.name
                    whale.t_observed_cue = t_datetime
                    whale.coord_observed = target_coord

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

                    _clear_actor_task(actor.name, whale_idx, eo_tools_dict, att_models_dict, eul_default=eul_ang_cue_default)

                    h_m = float(np.linalg.norm(np.array(r))) - R_earth
                    gsd_cue = gsd_offnadir(gsd0_cue, h_m, offnadir_cue_deg)

                    cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r[0], r[1], r[2], t_datetime).flatten()
                    cue_lat, cue_lon, cue_alt = float(cue_lat), float(cue_lon), float(cue_alt)

                    if whale.tip_actor != None and whale.t_observed_tip != None:
                        if t_datetime < whale.t_observed_tip + timedelta(seconds=observation_time_limit):
                            latency = (whale.t_observed_cue - whale.t_observed_tip).total_seconds()
                    else:
                        latency = 0.0

                    if logging:

                        log_cue_observation(writer_cue, whale_idx, t_datetime, actor.name, offnadir_cue_deg, gsd_cue, target_coord[0], target_coord[1], target_coord[2], cue_lat,
                                          cue_lon, cue_alt, r[0], r[1], r[2], v[0], v[1], v[2],     att_models_dict[actor.name]._actor_attitude_deg[0], att_models_dict[actor.name]._actor_attitude_deg[1], att_models_dict[actor.name]._actor_attitude_deg[2], cue_observation_counter=1)

                        log_combined_observation(writer_combined, whale_idx, whale.t_observed_tip, whale.tip_actor, whale.t_observed_cue, whale.cue_actor, offnadir_cue_deg, gsd_cue,
                                               latency, whale.lat, whale.lon, whale.alt, cue_lat, cue_lon, cue_alt, tip_observation_counter=1, cue_observation_counter=1)



                    if generate_image:
                        print("Generate image")

                        DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(
                            img_path, satellite, cue_lat, cue_lon, cue_alt, target_coord[0], target_coord[1],
                            target_coord[2], t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                    if onboard_ai_cue:
                        whale.confirmed_cue = cue_ai_decision(whale, cue_tpr, cue_tnr, seed_cue)

                    else:
                        whale.confirmed_cue = True

                    if whale.t_observed_cue != None and whale.state_confirming < 2 and t_datetime > (whale.t_observed_cue + timedelta(seconds=delay_confirmation_cue)):

                            whale.t_confirmed_cue = t_datetime
                            whale.state_confirming= 2

                            n_confirmed_cue += 1

                            if whale.confirmed_cue:  # Add better decision to prevent twice!!
                                whale.ai_class_predicted="whale"
                                confirmed_targets_pos[whale_idx] = whale
                                n_confirmed_pos += 1

                            elif not whale.confirmed_cue:
                                whale.ai_class_predicted = "not-whale"
                                confirmed_targets_neg[whale_idx] = whale
                                n_confirmed_neg += 1

                            print(f"!! {actor.name}: Confirmed Target {whale_idx}={whale.ai_class_predicted} (actual={whale.ai_class_true})")



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
        obs_tip_pts, task_pts, obs_cue_pts, conf_pts_pos, conf_pts_neg = update_plotter(pl,
                   earth_actor, earth_state,
                   sun_light, cloud_tip_sats, cloud_cue_sats,
                   whales_plot_all, whales_plot_observed_tip, whales_plot_tasked, whales_plot_observed_cue, whales_plot_confirmed_pos, whales_plot_confirmed_neg,
                   tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes,
                   t_datetime, tip_positions, cue_positions,
                   all_targets, observed_targets_tip, tasked_targets, observed_targets_cue, confirmed_targets_pos, confirmed_targets_neg,
                   obs_tip_pts, task_pts, obs_cue_pts, conf_pts_pos, conf_pts_neg,
                   FovPoints_tip, FovPoints_cue, step_text, n_steps)

        try:
            pl.write_frame()

        except:
            print(f"PyVista warning: skipped writing frame at {n_steps}")

    t_end = time.time()

    if n_steps % print_interval == 0:
        print(f" {n_steps} Time iteration: {t_mid - t_start:.2f} | Time plot: {t_end - t_mid:.2f} "
              f"| Simulation time: {int(elapsed_hours)}h {(int(elapsed_seconds) - int(elapsed_hours) * 3600) // 60}m {(int(elapsed_seconds) - int(elapsed_hours) * 3600) % 60}s")


    # for actor in tip_actors + cue_actors:
    #     att_models_dict[actor.name].update_attitude(sim_step_seconds)

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)

    elapsed_seconds += sim_step_seconds
    elapsed_hours = elapsed_seconds / 3600
    n_steps += 1

t_sim_end = time.time()
total_sim_time = t_sim_end - t_sim_start
hours_sim, rem = divmod(total_sim_time, 3600)
minutes_sim, seconds_sim = divmod(rem, 60)

print(f"\nTotal simulation time: {int(hours_sim)}h {int(minutes_sim)}m {seconds_sim:.0f}s | "
      f"Time per iteration: {total_sim_time/n_steps:.2f}s | plot_propagation {plot_propagation}\n")

observation_efficiency_tip = n_observed_tip / n_targets
observation_efficiency_cue = n_observed_cue / n_targets
confirmation_efficiency = n_confirmed_cue / n_targets

# --- Order: Satellites + Whales first ---
print(f"Number of satellites:             {nSats_tip + nSats_cue} (Tip={nSats_tip}, Cue={nSats_cue})")
print(f"Total targets:                    {n_targets} (positive={n_targets_pos}, negative={n_targets_neg})")
print(f"Simulation time:                  {sim_duration_hours} h")

# --- Orbits ---
n_float_tip, n_full_tip, residual_tip, T_tip = count_orbits_completed(a_tip, sim_duration_seconds)
n_float_cue, n_full_cue, residual_cue, T_cue = count_orbits_completed(a_cue, sim_duration_seconds)

if T_tip != T_cue:
    print(f"Tip: orbits completed={n_float_tip:.3f} "
          f"(full={n_full_tip}, residual={residual_tip:.2f}s, period={T_tip:.1f}s)")
    print(f"Cue: orbits completed={n_float_cue:.3f} "
          f"(full={n_full_cue}, residual={residual_cue:.2f}s, period={T_cue:.1f}s)\n")
else:
    print(f"Orbits completed:                 {n_float_tip:.3f} "
          f"(full={n_full_tip}, residual={residual_tip:.2f}s, period={T_tip:.1f}s)\n")

# --- Observation / Efficiencies ---
print(f"Tip observed:                     {n_observed_tip} (without duplicates: {len(observed_targets_tip)})")
print(f"Tip confirmed:                    {n_confirmed_tip}")
print(f"Tip tasks sent:                   {n_tasked_tip}")
print(f"Cue tasks started:                {n_tasked_cue}")
print(f"Cue observed:                     {n_observed_cue} (without duplicates: {len(observed_targets_cue)})")
print(f"Cue confirmed:                    {n_confirmed_cue}\n")

print(f"Positive confirmed:               {n_confirmed_pos} (without duplicates: {len(confirmed_targets_pos)}, actual: {n_targets_pos})")
print(f"Negative confirmed:               {n_confirmed_neg} (without duplicates: {len(confirmed_targets_neg)}, actual: {n_targets_neg})\n")

print(f"Observation efficiency Tip:       {observation_efficiency_tip * 100:.2f}% of whales ({n_targets} total)")
print(f"Observation efficiency Cue:       {observation_efficiency_cue * 100:.2f}% of whales ({n_targets} total)")
print(f"Confirmation efficiency Cue:      {confirmation_efficiency * 100:.2f}% of whales ({n_targets} total)\n")

if plot_footprints:

    t1 = time.time()
    fov_polygons_tip = [f for f in fov_polygons_tip if f is not None]
    fov_polygons_cue = [f for f in fov_polygons_cue if f is not None]

    area_total_km2, area_mission_km2, area_covered_km2, area_covered_per_orbit_km2, area_mission_fraction_total, area_covered_fraction_total, area_covered_fraction_mission, area_covered_per_orbit_fraction_total, area_covered_per_orbit_fraction_mission = compute_coverage_fraction(
        fov_polygons_tip, fov_polygons_cue, R_earth, i_tip_deg, a_tip, sim_duration_seconds)

    print(f"Total coverage:                   {area_covered_km2:,.0f} km² "
          f"({area_covered_fraction_total * 100:.2f}% of Earth, "
          f"{area_covered_fraction_mission * 100:.2f}% of possible surface)")
    print(f"Per orbit coverage:               {area_covered_per_orbit_km2:,.0f} km² "
          f"({area_covered_per_orbit_fraction_total * 100:.2f}% of Earth, "
          f"{area_covered_per_orbit_fraction_mission * 100:.2f}% of possible surface)")

    t2 = time.time()
    print(f"FOV: footprints computation time: {t2-t1:1f} s")

else:
    area_covered_km2 = area_covered_per_orbit_km2 = None
    area_covered_fraction_total = area_covered_fraction_mission = None
    area_covered_per_orbit_fraction_total = area_covered_per_orbit_fraction_mission = None

if logging:
    df_combined = pd.read_excel("sim_output.xlsx", sheet_name="CombinedLog")
    avg_latency_s, min_latency_s, max_latency_s, std_latency_s = compute_stats(df_combined["latency"])
    avg_offnadir_deg, min_offnadir_deg, max_offnadir_deg, std_offnadir_deg = compute_stats(df_combined["offnadir_deg"])

    print(f"Average latency:                  {avg_latency_s:.1f} s "
          f"(min {min_latency_s:.1f} s, max {max_latency_s:.1f} s, std {std_latency_s:.1f} s)")
    print(f"Average off-nadir angle:          {avg_offnadir_deg:.2f}°  "
          f"(min {min_offnadir_deg:.2f}°, max {max_offnadir_deg:.2f}°, std {std_offnadir_deg:.2f}°)\n")

# --- Write Overview log row-wise with blank lines ---
if logging:

    wb = openpyxl.load_workbook("sim_output.xlsx")
    ws = wb["Overview"]

    overview_data = [
        ("Number of satellites", nSats_tip + nSats_cue, f"(Tip={nSats_tip}, Cue={nSats_cue})"),
        ("Total targets", n_targets, f"(positive={n_targets_pos}, negative={n_targets_neg})"),
        ("Simulation time (h)", sim_duration_hours, ""),

        ("Tip orbits completed", round(n_float_tip, 3),
         f"(full={n_full_tip}, residual={residual_tip:.2f}s, period={T_tip:.1f}s)"),
        ("Cue orbits completed", round(n_float_cue, 3),
         f"(full={n_full_cue}, residual={residual_cue:.2f}s, period={T_cue:.1f}s)"),
        ("", "", ""),  # blank row

        ("Tip observed", n_observed_tip, f"(verification: {len(observed_targets_tip)})"),
        ("Tip confirmed", n_confirmed_tip, ""),
        ("Tip tasks sent", n_tasked_tip, ""),
        ("Cue tasks started", n_tasked_cue, ""),
        ("Cue observed", n_observed_cue, f"(verification: {len(observed_targets_cue)})"),
        ("Cue confirmed", n_confirmed_cue, ""),

        ("", "", ""),  # blank row
        ("Positive confirmed", n_confirmed_pos, f"(without duplicates: {len(confirmed_targets_pos)}, actual: {n_targets_pos})"),
        ("Negative confirmed", n_confirmed_neg, f"(without duplicates: {len(confirmed_targets_neg)}, actual: {n_targets_neg})"),

        ("", "", ""),  # blank row
        ("Tip observation efficiency (%)", round(observation_efficiency_tip * 100, 2),
         f"of whales ({n_targets} total)"),
        ("Cue observation efficiency (%)", round(observation_efficiency_cue * 100, 2),
         f"of whales ({n_targets} total)"),
        ("Cue confirmation efficiency (%)", round(confirmation_efficiency * 100, 2),
         f"of whales ({n_targets} total)"),
        ("", "", ""),  # blank row

        ("Total coverage (km²)", round(area_covered_km2, 1) if area_covered_km2 else "",
         f"({area_covered_fraction_total * 100:.2f}% Earth, {area_covered_fraction_mission * 100:.2f}% mission)" if area_covered_fraction_total else ""),
        ("Coverage per orbit (km²)", round(area_covered_per_orbit_km2, 1) if area_covered_per_orbit_km2 else "",
         f"({area_covered_per_orbit_fraction_total * 100:.2f}% Earth, {area_covered_per_orbit_fraction_mission * 100:.2f}% mission)" if area_covered_per_orbit_fraction_total else ""),
        ("", "", ""),  # blank row

        ("Average latency (s)", round(avg_latency_s, 1),
         f"(min {min_latency_s:.1f}, max {max_latency_s:.1f}, std {std_latency_s:.1f})"),
        ("Average off-nadir (deg)", round(avg_offnadir_deg, 2),
         f"(min {min_offnadir_deg:.2f}, max {max_offnadir_deg:.2f}, std {std_offnadir_deg:.2f})"),
    ]

    # Clear old rows (keep header)
    ws.delete_rows(2, ws.max_row)

    # Write new rows
    for row in overview_data:
        ws.append(row)

    # --- Auto-adjust column widths ---
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2  # padding
        ws.column_dimensions[col_letter].width = adjusted_width

    wb.save("sim_output.xlsx")

    if os.path.exists("sim_output.xlsx"):
        plot_offnadir_distribution("sim_output.xlsx", bin_size_deg=5)
        plot_latency_distribution("sim_output.xlsx", bin_size_sec=30)
        if verbose:
            print("Created offnadir and latency distribution plots")

at_exit(sim_name, pl=(pl if plot_propagation else None), verbose=verbose)

if plot_footprints:
    print(f"Generate footprint plots with len {len(fov_polygons_cue)}")
    t1 = time.time()

    if len(fov_polygons_tip) > 0:
        plot_all_fov_footprints_plotly(fov_polygons_tip, all_targets, observed_targets_tip, nPlanes_tip, nSats_tip, extension="tip", plot_whale_trajectories=plot_whale_trajectories, whale_trajectories=whale_trajectories)

    if len(fov_polygons_cue) > 0:
        plot_all_fov_footprints_plotly(fov_polygons_cue, all_targets, observed_targets_cue, nPlanes_cue, nSats_cue, extension="cue", plot_whale_trajectories=plot_whale_trajectories, whale_trajectories=whale_trajectories)

    t2 = time.time()
    print(f"FOV: footprints plotting time: {t2 - t1:1f} s")

if show_orbits:
    plot_orbits(trajectories)

