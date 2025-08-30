from settings import *

import orekit
import pyvista as pv
import numpy as np
import pykep as pk
from datetime import datetime, timedelta
import csv, atexit

from orekit.pyhelpers import setup_orekit_curdir

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from simulation.constellation import build_constellation
from custom_paseos.utils.point_transformation import Point_ECI2Geodetic

from simulation.plotting.plot_functions import plot_constallation, plot_orbits, plot_all_fov_footprints
from simulation.plotting.plot_pyvista import make_plotter_eci, update_earth_rotation_eci, whales_to_points_eci, sats_to_points_eci, init_fov_layers_eci, update_fov_layers_eci, init_sun_light, update_sun_light_eci, update_points_from_targets
from simulation.propagate_whales import update_whales, load_land_mask, generate_random_water_targets,  init_whales, build_land_mask
from simulation.simulation_functions import init_eo_tools, cleanup_tasked_targets, propagate_actor, log_tip_detection, log_cue_evaluation, satellite_in_shadow, target_illuminated, daylight_mask

show_constellation = True
plot_propagation = True
plot_footprints = False
show_orbits = False
generate_image = False

# Initialize Orekit
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import IERSConventions
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory

pv.global_theme.allow_empty_mesh = True
paseos.set_log_level("WARNING")

# Time setup
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
    propagator = OrekitPropagator(
        orbital_elements=planet.orbital_elements,
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
offnadir_angle_deg = 0.0
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

# Log files
file_tip = open("sim_output_tip.csv", "w", newline="")
file_cue = open("sim_output_cue.csv", "w", newline="")

header_tip = ["date", "actor", "target_id", "target_lat", "target_lon", "target_alt", "x", "y", "z", "vx", "vy", "vz", "in_footprint"]
header_cue = ["date", "actor", "target_id", "target_lat", "target_lon", "target_alt", "x", "y", "z", "vx", "vy", "vz", "offnadir_angle_deg", "in_view", "in_footprint", "yaw", "pitch", "roll"]

writer_tip = csv.writer(file_tip)
writer_cue = csv.writer(file_cue)

writer_tip.writerow(header_tip)
writer_cue.writerow(header_cue)

def close_files():
    file_tip.close()
    file_cue.close()
    print("CSV files closed and saved.")

atexit.register(close_files)

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

    whales_plot_all = pv.PolyData(np.zeros((len(all_targets), 3)))
    pl.add_points(whales_plot_all, color="red", point_size=8, render_points_as_spheres=True)

    eval_pts = np.full((n_whales, 3), np.nan)
    whales_plot_evaluated = pv.PolyData(eval_pts.copy())
    pl.add_points(whales_plot_evaluated, color="green", point_size=9, render_points_as_spheres=True)

    task_pts = np.full((n_whales, 3), np.nan)
    whales_plot_tasked = pv.PolyData(task_pts.copy())
    pl.add_points(whales_plot_tasked, color="orange", point_size=10, render_points_as_spheres=True)

    cloud_tip_sats = pv.PolyData(np.zeros((len(tip_actors), 3)))
    pl.add_points(cloud_tip_sats, color="yellowgreen", point_size=20, render_points_as_spheres=True)

    cloud_cue_sats = pv.PolyData(np.zeros((len(cue_actors), 3)))
    pl.add_points(cloud_cue_sats, color="lightseagreen", point_size=15, render_points_as_spheres=True)

    tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes = init_fov_layers_eci(pl, n_tip=len(tip_actors), n_cue=len(cue_actors), tip_fill_color="orange", cue_fill_color="cyan", tip_edge_color="white", cue_edge_color="white", opacity=0.35, line_width=2.0 )
    sun_light = init_sun_light(pl)

    pl.add_text("Tip and Cue Simulation", font_size=12)
    pl.show(cpos="xy", interactive_update=True, auto_close=False)


elapsed_time, n_steps = 0.0, 0
while elapsed_time <= sim_duration_seconds:

    t_pykep = sim.local_time
    t_datetime = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=t_pykep.mjd2000)
    t_abs = AbsoluteDate(t_datetime.year, t_datetime.month, t_datetime.day, t_datetime.hour, t_datetime.minute, t_datetime.second + t_datetime.microsecond / 1e6, utc)

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
        tip_illuminated = False
        eo_tools = eo_tools_dict[actor.name]

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        tip_positions.append(r)

        if not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius()):
            tip_illuminated = True

            FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_tip, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
            FovPoints_tip.append(FovPoints)

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
                    log_tip_detection(writer_tip, t_datetime, actor, whale_idx, target_coord, r, v, in_footprint)

                    n_detections +=1
                    tip_detected = True

        if tip_detected == False:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"detections={n_detections} | illuminated={tip_illuminated}")

        if tip_detected == True:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"detections={n_detections} | illuminated={tip_illuminated} | {w['lat'], w['lon'], w['alt']}")

        boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_tip, t_datetime)
        if boresight_hit is not None:

            # Print where the tip satellite is positioned
            tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime)
            print(
                f"\tTip position at lat={float(tip_lat):.4f}, "
                f"lon={float(tip_lon):.4f}, alt={float(tip_alt):.1f}"
            )

            # Print where center ray intersects Earth
            lat_b, lon_b, alt_b = boresight_hit
            print(
                f"\tTip boresight at lat={float(lat_b):.4f}, "
                f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}"
            )

        if plot_footprints:
            all_fov_polygons.append(FovPoints)

    for actor in cue_actors:
        n_evaluated = 0
        in_view = False
        cue_evaluated = False
        cue_illuminated = False
        cue_busy = False
        eo_tools = eo_tools_dict[actor.name]

        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        cue_positions.append(r)

        if not satellite_in_shadow(r_vec, sun_vec_eci, earth.getEquatorialRadius()):
            cue_illuminated = True

            for whale_idx, whale in tasked_targets.items():
                if whale_idx not in illuminated_targets:
                    continue

                detection_time = tasked_targets[whale_idx]["detection_time"]
                tasking_delay = tasked_targets[whale_idx]["tasking_delay"]

                if t_datetime > detection_time + timedelta(seconds=tasking_delay): # Skip if not yet transmitted

                    target_coord = (whale["lat"], whale["lon"], whale["alt"])
                    in_view = eo_tools.is_in_sight(target_coord, r_vec, v_vec, t_datetime, el_min=elevation_min)

                    if in_view:

                        offnadir_angle_deg, vec_brf = eo_tools.off_nadir_pointing_angle(z_brf=z_brf, r_eci=r_vec, v_eci=v_vec, target_geodetic=target_coord, eul_angles_deg=[0.0, 0.0, 0.0], time=t_datetime)

                        if offnadir_angle_deg <= offnadir_max:

                            yaw, pitch, roll = eo_tools.pointing_attitude(z_brf, vec_brf, phi_rad, [0.0, 0.0, 0.0], in_view)
                            eul_ang_cue = [yaw, pitch, roll]

                            print("!! Cue: Target in view", whale_idx, " | Set yaw, pitch, roll to:", eul_ang_cue)
                            cue_busy = True

            if not cue_busy:
                eul_ang_cue = [0.0, 0.0, 0.0]
                offnadir_angle_deg = 0.0

            try:
                FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_cue, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
                FovPoints_cue.append(FovPoints)

            except:
                print("!! Cue: target out of sight, continue to the next step")
                eul_ang_cue = [0.0, 0.0, 0.0]
                continue

            for whale_idx, whale in all_targets.items():

                target_coord = (whale["lat"], whale["lon"], whale["alt"])
                in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

                if in_footprint:

                    print("!! Cue: Target in footprint", whale_idx, " | Evaluate target")
                    w = {"lat": target_coord[0], "lon": target_coord[1], "alt": target_coord[2], "detection_time": t_datetime, "detection_satellite": actor.name}
                    w["tasking_delay"] = tasking_delay_cue
                    detected_targets[whale_idx], tasked_targets[whale_idx] = w, w  # to keep track of history
                    all_targets[whale_idx]["detected"] = 2  # detected by cue
                    log_cue_evaluation(writer_cue, t_datetime, actor, whale_idx, target_coord, r, v, offnadir_angle_deg, in_view, 1, yaw, pitch, roll)

                    n_evaluated += 1
                    cue_evaluated = True

                    satellite_lat, satellite_lon, satellite_alt = Point_ECI2Geodetic(r[0], r[1], r[2],t_datetime).flatten()
                    satellite_lat, satellite_lon, satellite_alt = float(satellite_lat), float(satellite_lon), float(satellite_alt)  # meters above ellipsoid

                    if generate_image:
                        print("Generate image")
                        DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(
                            img_path, satellite, satellite_lat, satellite_lon, satellite_alt, target_coord[0], target_coord[1],
                            target_coord[2], t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                    evaluated_targets[whale_idx] = w
                    del tasked_targets[whale_idx]

                    if plot_footprints:
                        all_fov_polygons.append(FovPoints)


        if cue_evaluated == True:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"target={target_coord[:2]} | "
                f"off nadir angle={offnadir_angle_deg:.2f}, in_view={in_view}, in_footprint={in_footprint}"
            )

        if cue_evaluated == False:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"detections={n_evaluated} | illuminated={cue_illuminated}")

        boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_cue, t_datetime)
        if boresight_hit is not None:
            # Print where the tip satellite is positioned
            cue_lat, cue_lon, cue_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime)
            print(
                f"\tCue position  at lat={float(cue_lat):.4f}, "
                f"lon={float(cue_lon):.4f}, alt={float(cue_alt):.1f}"
            )
            lat_b, lon_b, alt_b = boresight_hit
            print(
                f"\tCue boresight at lat={float(lat_b):.4f}, "
                f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}")

            if cue_evaluated == True:
                print(
                    f"\tTarget location at lat={float(target_coord[0]):.4f}, "
                    f"lon={float(target_coord[1]):.4f}, alt={float(target_coord[2]):.1f}"
                )


    if plot_propagation:
        update_earth_rotation_eci(earth_actor, t_datetime, earth_state)
        update_sun_light_eci(sun_light, t_datetime)

        # satellites (ECI)
        cloud_tip_sats.points = sats_to_points_eci(tip_positions)
        cloud_cue_sats.points = sats_to_points_eci(cue_positions)

        # all whales (fixed size)
        whales_plot_all.points = whales_to_points_eci(all_targets, t_datetime)

        # Evaluated whales
        eval_pts = update_points_from_targets(eval_pts, evaluated_targets, t_datetime)
        whales_plot_evaluated.points = eval_pts

        # Tasked whales
        task_pts = update_points_from_targets(task_pts, tasked_targets, t_datetime)
        whales_plot_tasked.points = task_pts

        # FoV polygons
        update_fov_layers_eci(
            tip_fill_meshes, tip_edge_meshes,
            cue_fill_meshes, cue_edge_meshes,
            FovPoints_tip, FovPoints_cue, t_datetime
        )

        pl.update()

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)
    elapsed_time += sim_step_seconds
    n_steps += 1
    print('\n')


if show_orbits:
    plot_orbits(trajectories)

if plot_footprints:
    plot_all_fov_footprints(all_fov_polygons, known_targets)





