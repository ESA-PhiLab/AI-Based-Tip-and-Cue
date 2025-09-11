from settings import *

import orekit
import pyvista as pv
import numpy as np
import pykep as pk
from datetime import datetime, timedelta
import atexit
import time
import gc
import os

from orekit.pyhelpers import setup_orekit_curdir

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from custom_paseos.utils.point_transformation import Point_ECI2Geodetic, Point_Geodetic2ECI
from custom_paseos.attitude.controller import StabilizedAttitudeController, SO3PIDACS

from simulation.propagate_whales import update_whales, load_land_mask, generate_random_water_targets, init_whales, build_land_mask
from simulation.simulation_functions import init_eo_tools, init_attitude_controllers, cleanup_tasked_targets, propagate_actor, satellite_in_shadow, daylight_mask, convert_M_to_lv, pointing_cost
from simulation.plotting.plot_functions import plot_constallation, plot_orbits, plot_all_fov_footprints
from simulation.plotting.plot_pyvista import make_plotter_eci, reset_plotter, update_plotter
from simulation.plotting.plot_constellation import plot_constellation_pyvista
from simulation.logging import init_excel_log, log_tip_detection, log_cue_evaluation, gsd_offnadir, at_exit

model_attitude_control = True
show_constellation = False
plot_propagation = False
plot_footprints = True
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

from simulation.constellation import build_constellation

from astropy.utils.iers import conf

conf.auto_max_age = None  # allow predictive values older than 30 days

pv.global_theme.allow_empty_mesh = True
paseos.set_log_level("WARNING")

# Time setup
print(f"Initiate simulation {sim_name} | Attitude control {model_attitude_control} | Logging {logging}")
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
    plot_constellation_pyvista(planet_lst_tip, planet_lst_cue, t0)

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
    ActorBuilder.set_custom_orbit(actor, lambda t, p=propagator: (
        list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getPosition().toArray()),
        list(p.eph((t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC).getPVCoordinates().getVelocity().toArray())
    ), t0_pykep)

    ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)

    (tip_actors if "Tip" in planet.name else cue_actors).append(actor)

eul_ang_tip_default = [0.0, 0.0, 0.0]
eul_ang_cue_default = [0.0, 0.0, 0.0]
offnadir_cue_deg_target = 0.0
offnadir_tip_deg = 0.0
all_fov_polygons = []

# EO Tools
eo_tools_dict = init_eo_tools(tip_actors, cue_actors, fov_tip, fov_cue, eul_ang_tip_default, eul_ang_cue_default)
controllers = init_attitude_controllers(None, cue_actors, eo_tools_dict, None, controller_params)  # Only initialize controller for cue

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
                  "offnadir_deg", "GSD_m", "in_view", "in_footprint", "roll", "pitch", "yaw"]

    writer_tip = init_excel_log("sim_output_tip.xlsx", header_tip, sheet_name="TipLog")
    writer_cue = init_excel_log("sim_output_cue.xlsx", header_cue, sheet_name="CueLog")

    atexit.register(at_exit, save_name=sim_name)
    print("Initiated logging files")

os.makedirs(worldmap_dir, exist_ok=True)
npy_path_full = os.path.join(worldmap_dir, mask_npy)

if not os.path.exists(npy_path_full):
    mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)
else:
    mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)

known_targets = generate_random_water_targets(n_whales, mask, res_deg, seed_val=whale_seed, max_abs_lat_val=max_abs_lat)
all_targets = init_whales(known_targets, seed_val=whale_seed)  # live updated
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

detect_idx = None
eval_idx = None
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

        #  centerray_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, t_datetime).flatten()
        #  ctr_lat, ctr_lon, ctr_alt = centerray_hit.flatten()

        FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
        FovPoints_tip.append(FovPoints)

        #  if plot_footprints:
        #      all_fov_polygons.append(FovPoints)

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

                target_coord = (whale["lat"], whale["lon"], whale["alt"])
                in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

                if in_footprint and whale_idx not in tasked_targets.keys():  # only detect new targets
                    print(f"!! {actor.name}: Target detected", whale_idx)

                    w = {"lat": target_coord[0], "lon": target_coord[1], "alt": target_coord[2], "detection_time": t_datetime, "detection_satellite": actor.name}
                    w["tasking_delay"] = tasking_delay_tip

                    best_cue, best_dist = None, float("inf")

                    for cue_actor in cue_actors:
                        # Propagate cue satellite forward by avg_time_delay
                        t_future = pk.epoch(t_pykep.mjd2000 + avg_time_delay / pk.DAY2SEC)
                        r_future, _, _, _ = propagate_actor(cue_actor, t_future, None, n_steps, show_orbits=False)

                        # Target position in ECI at the same future time
                        tgt_lat, tgt_lon, tgt_alt = target_coord
                        tgt_vec = np.array(Point_Geodetic2ECI(tgt_lat, tgt_lon, tgt_alt, t_datetime + timedelta(seconds=avg_time_delay)))

                        # Distance between satellite and target in ECI
                        dist = np.linalg.norm(r_future - tgt_vec)

                        if dist < best_dist:
                            best_dist = dist
                            best_cue = cue_actor.name

                    w["assigned_cue"] = best_cue
                    print(f"!! {actor.name}: Assigned Target {whale_idx} to {best_cue}")

                    eo_tools_dict[best_cue].task_queue.append({
                        "target_id": whale_idx,
                        "coord": target_coord,
                        "assign_time": t_datetime
                    })

                    tasked_targets[whale_idx], detected_targets[whale_idx] = w, w
                    all_targets[whale_idx]["detected"] = 1
                    detect_idx = whale_idx

                    h_m = float(np.linalg.norm(np.array(r))) - R_earth
                    gsd_tip = gsd_offnadir(GSD0_tip, h_m, offnadir_tip_deg)

                    if logging:
                        log_tip_detection(writer_tip, t_datetime, actor, whale_idx, target_coord, r, v, offnadir_tip_deg, gsd_tip, in_footprint)

                    n_detections += 1
                    tip_detected = True

        if verbose == True and n_steps % print_interval == 0:
            tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
            tip_lat, tip_lon, tip_alt = float(tip_lat), float(tip_lon), float(tip_alt)  # meters above ellipsoid

            print(f"\t\t{actor.name} | {t_datetime.isoformat()} | lat={tip_lat:.1f}, lon={tip_lon:.1f}, alt={tip_alt:.1f} | illuminated={tip_illuminated}")
            if tip_detected == True:
                print(f"\t\tTarget: idx={detect_idx} | gsd={gsd_tip:.2f} | lat={w['lat']:.1f}, lon={w['lon']:.1f}, alt={w['alt']:.1f}")



    for actor in cue_actors:
        n_evaluated = 0
        cue_evaluated = False
        eo_tools = eo_tools_dict[actor.name]

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        cue_positions.append(r)

        try:
            cue_illuminated = not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius())
        except:
            cue_illuminated = True
            print(f"!! {actor.name}: failed to compute illumination state, set to True preventing exclusion.")
            print(r_vec, sun_vec_eci, earth.getEquatorialRadius())

        if cue_illuminated:
            for whale_idx, whale in tasked_targets.items():
                if whale["assigned_cue"] != actor.name and whale["assigned_cue"] != None:  # Skip if not assigned
                    continue

                if whale_idx not in illuminated_targets:
                    continue

                detection_time = tasked_targets[whale_idx]["detection_time"]
                tasking_delay = tasked_targets[whale_idx]["tasking_delay"]

                if t_datetime > detection_time + timedelta(seconds=tasking_delay):  # Skip if not yet transmitted

                    target_coord = (whale["lat"], whale["lon"], whale["alt"])

                    # If no current task, check the queue
                    if eo_tools.current_task is None and eo_tools.task_queue:

                        # Filter tasks that will actually become visible soon
                        ready_tasks = []
                        for task in eo_tools.task_queue:
                            will_be_visible, _ = eo_tools.will_be_visible_within(
                                task["coord"], r_vec, v_vec, t_datetime,
                                delta_t_cue, el_min_deg=elevation_min, step=60.0
                            )
                            if will_be_visible:
                                ready_tasks.append(task)

                        if ready_tasks:
                            # Pick the ready task with least pointing cost
                            eo_tools.current_task = min(
                                ready_tasks,
                                key=lambda task: pointing_cost(task, eo_tools, r_vec, v_vec, t_datetime)
                            )
                            eo_tools.task_queue.remove(eo_tools.current_task)
                            print(f"!! {actor.name}: Starting task for Target {eo_tools.current_task['target_id']}")

                    # If still no current task, assign this whale now
                    if eo_tools.current_task is None and whale["assigned_cue"] == actor.name:
                        eo_tools.current_task = {
                            "target_id": whale_idx,
                            "coord": target_coord
                        }
                        print(f"!! {actor.name}: New task assigned for Target {whale_idx}")

                    # Work on the current task if one exists
                    if eo_tools.current_task:
                        task_id = eo_tools.current_task["target_id"]
                        task_coord = eo_tools.current_task["coord"]

                        in_view = eo_tools.is_in_sight(task_coord, r_vec, v_vec, t_datetime, el_min=elevation_min)
                        will_be_in_view, _ = eo_tools.will_be_visible_within(task_coord, r_vec, v_vec, t_datetime, delta_t_cue, el_min_deg=elevation_min, step=60.0)

                        if in_view or will_be_in_view:

                            offnadir_cue_deg_target, pointing_vec_brf_target = eo_tools.off_nadir_pointing_angle(r_eci=r_vec, v_eci=v_vec, target_geodetic=task_coord, t_datetime=t_datetime)

                            if offnadir_cue_deg_target > offnadir_max:
                                offnadir_cue_deg_target, pointing_vec_brf_target = eo_tools.set_max_offnadir(pointing_vec_brf_target, offnadir_cue_deg_target, offnadir_max)

                            if eo_tools.eul_ang_target == eul_ang_cue_default:
                                eo_tools.eul_ang_target = eo_tools.pointing_attitude_brf(pointing_vec_brf_target)       # compute double, only for print
                                print(f"!! {actor.name}: Target {whale_idx} in reach | Set roll, pitch, yaw to {eo_tools.eul_ang_target[0]:.1f}, {eo_tools.eul_ang_target[1]:.1f}, {eo_tools.eul_ang_target[2]:.1f} deg")

                            eo_tools.eul_ang_target = eo_tools.pointing_attitude_brf(pointing_vec_brf_target)

                        else:
                            # Task finished → reset and pick next later
                            print(f"!! {actor.name}: Target {task_id} out of view")
                            eo_tools.current_task = None
                            eo_tools.eul_ang_target = eul_ang_cue_default
                            offnadir_cue_deg_target = 0.0

            if eo_tools.current_task == None:
                if np.any(eo_tools.eul_ang_target != eul_ang_cue_default):

                    print(f"!! {actor.name}: Set roll, pitch, yaw target back to default {eul_ang_cue_default[0]:.1f}, {eul_ang_cue_default[1]:.1f}, {eul_ang_cue_default[2]:.1f} deg")

                    eo_tools.eul_ang_target = eul_ang_cue_default
                    offnadir_cue_deg_target = 0.0


        if model_attitude_control:
            if not np.allclose(eo_tools.eul_ang_deg, eo_tools.eul_ang_target, atol=1e-6, rtol=0):
                ctrls = controllers[actor.name]
                guid = ctrls["guidance"]
                acs = ctrls["acs"]

                eul_target_stab = guid.update_target(eo_tools.eul_ang_target, sim_step_seconds, eo_tools.eul_ang_deg )

                eul_new_deg = acs.step(eul_target_stab, sim_step_seconds)

                delta_move_eul = eul_new_deg - eo_tools.eul_ang_deg
                eo_tools.eul_ang_deg = eul_new_deg

                if verbose == True and n_steps % print_interval == 0:
                    print(f"\t\t {actor.name} | Delta roll, pitch, yaw={delta_move_eul[0]:.1f}, {delta_move_eul[1]:.1f}, {delta_move_eul[2]:.1f} deg")

        else:
            eo_tools.eul_ang_deg = eo_tools.eul_ang_target

        try:
            #   centerray_hit = eo_tools.get_CenterRay_Intersection_Attitude(r_vec, v_vec, t_datetime).flatten()
            #   ctr_lat, ctr_lon, ctr_alt = centerray_hit

            FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
            FovPoints_cue.append(FovPoints)

            if plot_footprints:
                all_fov_polygons.append(FovPoints)

        except:
            print(f"!! {actor.name}: no FOV intersection, continue to the next step")
            eo_tools.eul_ang_target = eul_ang_cue_default
            offnadir_cue_deg_target = 0.0
            continue

        if cue_illuminated:

            for whale_idx, whale in all_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                target_coord = (whale["lat"], whale["lon"], whale["alt"])
                in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

                if in_footprint and whale_idx != eval_idx:

                    print(f"!! {actor.name}: Target in footprint", whale_idx, " | Evaluate target")
                    w = {"lat": target_coord[0], "lon": target_coord[1], "alt": target_coord[2], "detection_time": t_datetime, "detection_satellite": actor.name}
                    w["tasking_delay"] = tasking_delay_cue
                    detected_targets[whale_idx], tasked_targets[whale_idx] = w, w  # to keep track of history
                    all_targets[whale_idx]["detected"] = 2  # detected by cue
                    eval_idx = whale_idx

                    offnadir_cue_deg, pointing_vec_brf_target = eo_tools.off_nadir_pointing_angle(r_eci=r_vec, v_eci=v_vec, target_geodetic=target_coord, t_datetime=t_datetime)

                    h_m = float(np.linalg.norm(np.array(r))) - R_earth
                    gsd_cue = gsd_offnadir(GSD0_cue, h_m, offnadir_cue_deg)

                    if logging:
                        log_cue_evaluation(writer_cue, t_datetime, actor, whale_idx, target_coord, r, v,
                                           offnadir_cue_deg, gsd_cue, in_view, in_footprint, eo_tools.eul_ang_deg[0], eo_tools.eul_ang_deg[1], eo_tools.eul_ang_deg[2])

                    n_evaluated += 1
                    cue_evaluated = True

                    if generate_image:
                        print("Generate image")
                        cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
                        cue_lat, cue_lon, cue_alt = float(cue_lat), float(cue_lon), float(cue_alt)

                        DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(
                            img_path, satellite, cue_lat, cue_lon, cue_alt, target_coord[0], target_coord[1],
                            target_coord[2], t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                    evaluated_targets[whale_idx] = w
                    del tasked_targets[whale_idx]
                    eo_tools.current_task = None

                    if plot_footprints:
                        all_fov_polygons.append(FovPoints)

        eo_tools_dict[actor.name] = eo_tools

        if verbose == True and n_steps % print_interval == 0:
            cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
            cue_lat, cue_lon, cue_alt = float(cue_lat), float(cue_lon), float(cue_alt)

            print(f"\t\t{actor.name} | {t_datetime.isoformat()} | lat={cue_lat:.1f}, lon={cue_lon:.1f}, alt={cue_alt:.1f} | illuminated={cue_illuminated}")
            if cue_evaluated == True:
                print(f"\t\tTarget: idx={eval_idx} | off nadir angle={offnadir_cue_deg:.1f} | gsd={gsd_cue:.2f} | lat={w['lat']:.1f}, lon={w['lon']:.1f}, alt={w['alt']:.1f}")

    if n_steps % 100 == 0:
        gc.collect()  # Empty garbage

    t_mid = time.time()

    if plot_propagation and n_steps % plot_interval == 0:
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
        print(f" {n_steps} Time iteration: {t_mid - t_start:.1f} | Time plot: {t_end - t_mid:.1f}")

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)
    elapsed_time += sim_step_seconds
    n_steps += 1

if show_orbits:
    plot_orbits(trajectories)

if plot_footprints:
    plot_all_fov_footprints(all_fov_polygons, known_targets)





