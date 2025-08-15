from settings import *

import orekit
import pyvista as pv
import math
import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.time import AbsoluteDate, TimeScalesFactory

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from custom_paseos.propagation.get_constellation import get_constellation
from custom_paseos.observation.EarthObservation import EOTools

from custom_paseos.utils.help_functions import fov_angle_from_swath
from custom_paseos.utils.point_transformation import Point_ECI2Geodetic

from custom_paseos.plot_functions.plot_functions import plot_constallation, plot_orbits
from custom_paseos.plot_functions.plot_pyvista import make_plotter_eci, update_earth_rotation_eci, whales_to_points_eci, sats_to_points_eci, init_fov_layers_eci, update_fov_layers_eci, init_sun_light, update_sun_light_eci

from offnadir_imaging.rendering import generate_image
from simulation.propagate_whales import step_whale, load_land_mask, generate_random_water_targets,  init_whales, build_land_mask

import cartopy.crs as ccrs
import cartopy.feature as cfeature

show_constellation = False
plot_propagation = True
plot_footprints = True
show_orbits = True
generate_image = False

# Initialize Orekit
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)
paseos.set_log_level("WARNING")

# Time setup
utc = TimeScalesFactory.getUTC()
t0_orekit = AbsoluteDate(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond / 1e6, utc)
t0_pykep = pk.epoch_from_string(t0.strftime("%Y-%m-%d %H:%M:%S"))

# Get constellations
planet_lst_tip, sats_tip, orbital_period_tip = get_constellation(
    a_tip, e_tip, i_tip_deg, RAAN_tip_deg, argp_tip_deg, M_tip_deg,
    nSats_tip, nPlanes_tip, t0_pykep, "Tip", verbose=True
)
planet_lst_cue, sats_cue, orbital_period_cue = get_constellation(
    a_cue, e_cue, i_cue_deg, RAAN_cue_deg, argp_cue_deg, M_cue_deg,
    nSats_cue, nPlanes_cue, t0_pykep, "Cue", verbose=True
)

# Combine planets
all_planets = planet_lst_tip + planet_lst_cue

if show_constellation:
    plot_constallation(planet_lst_tip, planet_lst_cue, R_earth=R_earth, plot_margin=500e3)

# Propagator setup
def make_propagator_function(propagator, t0_pykep):
    def fn(t):
        dt = (t.mjd2000 - t0_pykep.mjd2000) * pk.DAY2SEC
        state = propagator.eph(dt)
        r = list(state.getPVCoordinates().getPosition().toArray())
        v = list(state.getPVCoordinates().getVelocity().toArray())
        return r, v
    return fn

# Actor creation and separation
tip_actors = []
cue_actors = []

for planet in all_planets:
    name = planet.name
    orbital_elements = planet.orbital_elements

    propagator = OrekitPropagator(
        orbital_elements=orbital_elements,
        epoch=t0_orekit,
        satellite_mass=satellite_mass,
        area_s=area_s,
        cr_s=cr_s,
        area_d=area_d,
        cd=cd
    )

    actor = ActorBuilder.get_actor_scaffold(name=name, actor_type=SpacecraftActor, epoch=t0_pykep)
    ActorBuilder.set_custom_orbit(actor, make_propagator_function(propagator, t0_pykep), t0_pykep)
    ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)

    if "Tip" in name:
        tip_actors.append(actor)
    elif "Cue" in name:
        cue_actors.append(actor)
    else:
        raise ValueError(f"Unknown actor name: {name}")

observation_time = datetime(2025, 3, 5, 18, 40, 28)
z_brf = np.array([[0], [0], [1]])
eul_ang_tip = [0.0, 0.0, 0.0]
eul_ang_cue = [0.0, 0.0, 0.0]
phi_rad = 0.0

# EO tools per actor (initialized with estimated altitude)
eo_tools_dict = {}

for actor in tip_actors:
    eo_tools_dict[actor.name] = EOTools(
        local_actor=actor,
        actor_initial_attitude_in_deg=eul_ang_tip,
        actor_FOV_ACT_in_deg=[fov_tip],
        actor_FOV_ALT_in_deg=[fov_tip],
        actor_pointing_vector_body=[0.0, 0.0, 1.0]
    )

for actor in cue_actors:
    eo_tools_dict[actor.name] = EOTools(
        local_actor=actor,
        actor_initial_attitude_in_deg=eul_ang_cue,
        actor_FOV_ACT_in_deg=[fov_cue],
        actor_FOV_ALT_in_deg=[fov_cue],
        actor_pointing_vector_body=[0.0, 0.0, 1.0]
    )

# Init simulation
sim = paseos.init_sim(local_actor=tip_actors[0])
for actor in tip_actors[1:] + cue_actors:
    sim.add_known_actor(actor)

# Trajectory storage
trajectories = {actor.name: {"r": [], "v": []} for actor in tip_actors + cue_actors}

print("Total number of simulation steps:", int(sim_duration_seconds / sim_step_seconds))

# Log files
file_tip = open("sim_output_tip.txt", "w")
file_cue = open("sim_output_cue.txt", "w")

header = "date,actor,target_lat,target_lon,target_alt,x,y,z,vx,vy,vz,offnadir_angle_deg,in_view,in_footprint,yaw,pitch,roll\n"
file_tip.write(header)
file_cue.write(header)

# ================================
# Targets

os.makedirs(worldmap_dir, exist_ok=True)

npy_path_full = os.path.join(worldmap_dir, mask_npy)
if not os.path.exists(npy_path_full):
    mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)
else:
    mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)

known_targets = generate_random_water_targets(
    n_whales, mask, res_deg, seed_val=whale_seed, max_abs_lat_val=max_abs_lat
)

os.makedirs(worldmap_dir, exist_ok=True)

npy_path_full = os.path.join(worldmap_dir, mask_npy)
if not os.path.exists(npy_path_full):
    mask = build_land_mask(worldmap_dir, res_deg, mask_tif, mask_npy)
else:
    mask, _ = load_land_mask(worldmap_dir, mask_npy, res_deg)

available_targets_tip = init_whales(known_targets, seed_val=whale_seed)
available_targets_cue = {}

if plot_propagation:
    pl, earth_actor, earth_state = make_plotter_eci()

    whales_tip = pv.PolyData(np.zeros((len(available_targets_tip), 3)))
    pl.add_points(whales_tip, color="red", point_size=10, render_points_as_spheres=True)

    whales_cue = pv.PolyData(np.zeros((len(available_targets_tip), 3)))
    pl.add_points(whales_cue, color="lime", point_size=11, render_points_as_spheres=True)

    cloud_tip_sats = pv.PolyData(np.zeros((len(tip_actors), 3)))
    pl.add_points(cloud_tip_sats, color="yellowgreen", point_size=20, render_points_as_spheres=True)

    cloud_cue_sats = pv.PolyData(np.zeros((len(cue_actors), 3)))
    pl.add_points(cloud_cue_sats, color="lightseagreen", point_size=15, render_points_as_spheres=True)

    # FoV layers (one polygon per actor)
    tip_fill_meshes, tip_edge_meshes, cue_fill_meshes, cue_edge_meshes = init_fov_layers_eci(
        pl, n_tip=len(tip_actors), n_cue=len(cue_actors),
        tip_fill_color="orange", cue_fill_color="cyan",
        tip_edge_color="white", cue_edge_color="white",
        opacity=0.35, line_width=2.0
    )

    # Sun light (directional, updated each frame)
    sun_light = init_sun_light(pl)

    pl.add_text("Whale Simulation (ECI) — red=whales", font_size=12)
    pl.show(cpos="xy", interactive_update=True, auto_close=False)

if plot_footprints:
    fig, ax_map = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax_map.add_feature(cfeature.LAND, edgecolor='black')
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')

    for target_geodetic in known_targets:
        ax_map.plot(target_geodetic[1], target_geodetic[0], marker='o', color='green', markersize=4, transform=ccrs.PlateCarree())
        # ax_map.text(target_geodetic[1] - 7.5, target_geodetic[0] - 7.5, "Target", color='green', transform=ccrs.PlateCarree())

viewed_idx = []

# Simulation loop
while elapsed_time <= sim_duration_seconds:
    t_pykep = sim.local_time
    t_datetime = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=t_pykep.mjd2000)

    tip_positions = []
    cue_positions = []

    FovPoints_tip = []
    FovPoints_cue = []

    for actor in tip_actors + cue_actors:

        # Obtain position and velocity vectors
        r, v = actor.get_position_velocity(t_pykep)

        # Propagate orbits
        trajectories[actor.name]["r"].append(r)
        trajectories[actor.name]["v"].append(v)

    for whale_idx, w in available_targets_tip.items():
        w = step_whale(w, mask, res_deg, dt_sec=sim_step_seconds, whale_propagation=whale_propagation)
        available_targets_tip[whale_idx] = w

        if whale_idx in available_targets_cue.keys():
            temp = available_targets_cue[whale_idx]["detection_time"]
            available_targets_cue[whale_idx] = w
            available_targets_cue[whale_idx]["detection_time"] = temp

    for whale_idx, w in available_targets_cue.items():
        detection_time = available_targets_cue[whale_idx]["detection_time"]

        if t_datetime > detection_time + timedelta(seconds=detection_time_limit):
            del available_targets_cue[whale_idx]

    for actor in tip_actors:

        n_detections = 0
        tip_detected = False
        viewed_idx_iter = []

        eo_tools = eo_tools_dict[actor.name]
        r, v = trajectories[actor.name]["r"][-1], trajectories[actor.name]["v"][-1]

        r_vec = np.array(r).reshape(3, 1)
        v_vec = np.array(v).reshape(3, 1)

        tip_positions.append(r)

        FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_tip, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
        boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_tip, t_datetime)

        FovPoints_tip.append(FovPoints)

        for whale_idx, whale in available_targets_tip.items():

            tgt_lat, tgt_lon, tgt_alt = (whale["lat"], whale["lon"], whale["alt"])
            target_coord = (tgt_lat, tgt_lon, tgt_alt)
            in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

            if in_footprint:
                available_targets_cue[whale_idx] = {"lat": tgt_lat, "lon": tgt_lon, "alt": tgt_alt }
                available_targets_cue[whale_idx]["detection_time"] = t_datetime
                tip_detected = True
                n_detections +=1

        if tip_detected == False:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"detections={n_detections}")

        if tip_detected == True:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"detections={n_detections} | {whale_idx, tgt_lat, tgt_lon, tgt_alt}")

            file_tip.write(
                f"{t_datetime.isoformat()},{actor.name},"
                f"{whale_idx}, {tgt_lat:.4f},{tgt_lon:.4f},{tgt_alt:.1f},"
                f"{r[0]:.3f},{r[1]:.3f},{r[2]:.3f},"
                f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},"
                f"{in_footprint},"
            )

        if boresight_hit is not None:
            # Print where center ray intersects Earth
            lat_b, lon_b, alt_b = boresight_hit
            print(
                f"\tTip boresight at lat={float(lat_b):.4f}, "
                f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}"
            )

            # Print where the tip satellite is positioned
            tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime)
            print(
                f"\tTip position  at lat={float(tip_lat):.4f}, "
                f"lon={float(tip_lon):.4f}, alt={float(tip_alt):.1f}"
            )

        if plot_footprints:
            eo_tools.plot_fov_on_map(FovPoints, ax_map)

    for actor in cue_actors:

        eo_tools = eo_tools_dict[actor.name]
        r, v = trajectories[actor.name]["r"][-1], trajectories[actor.name]["v"][-1]

        r_vec = np.array(r).reshape(3, 1)
        v_vec = np.array(v).reshape(3, 1)

        cue_positions.append(r)

        eul_ang_cue = [0.0, 0.0, 0.0]
        FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_cue, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
        boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_cue, t_datetime)

        for whale_idx, whale in available_targets_cue.items():
            detection_time = available_targets_cue[whale_idx]["detection_time"]

            if not (t_datetime > detection_time + timedelta(seconds=cue_tasking_delay_sec)):
                # Skip if not yet transmitted
                continue

            tgt_lat, tgt_lon, tgt_alt = (whale["lat"], whale["lon"], whale["alt"])
            target_coord = (tgt_lat, tgt_lon, tgt_alt)

            # check if the target is in view
            in_view = eo_tools.is_in_sight(
                target_geodetic=target_coord,
                r_eci=r_vec,
                v_eci=v_vec,
                time=t_datetime,
                el_min=elevation_min
            )

            if in_view:

                # if in_view, compute off-nadir angle
                offnadir_angle_deg, vec_brf = eo_tools.off_nadir_pointing_angle(
                    z_brf=z_brf, r_eci=r_vec, v_eci=v_vec,
                    target_geodetic=target_coord,
                    eul_angles_deg=eul_ang_cue,
                    time=t_datetime
                )

                # Compute yaw, pitch, roll to point to the target
                yaw, pitch, roll = eo_tools.pointing_attitude(z_brf, vec_brf, phi_rad, eul_ang_cue, in_view)
                eul_ang_cue = [yaw, pitch, roll]

                # Plot footprint and check it the target falls within the footprint
                FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_cue, t_datetime) # Obtain off nadir angle and boresight hit (lat, lon, where center ray intersects earth)
                boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_cue, t_datetime)

                in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

                file_cue.write(
                    f"{t_datetime.isoformat()},{actor.name},"
                    f"{whale_idx}, {tgt_lat:.4f},{tgt_lon:.4f},{tgt_alt:.1f},"
                    f"{r[0]:.3f},{r[1]:.3f},{r[2]:.3f},"
                    f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},"
                    f"{offnadir_angle_deg:.4f},{in_view},{in_footprint},"
                    f"{yaw:.2f},{pitch:.2f},{roll:.2f}\n"
                )

                satellite_lat, satellite_lon, satellite_alt = Point_ECI2Geodetic(r[0], r[1], r[2],t_datetime).flatten()
                satellite_lat, satellite_lon, satellite_alt = float(satellite_lat), float(satellite_lon), float(satellite_alt)  # meters above ellipsoid

                # if in_footprint:

                if generate_image:
                    print("Detection made, generate image")
                    DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(
                        img_path, satellite, satellite_lat, satellite_lon, satellite_alt, tgt_lat, tgt_lon,
                        tgt_alt, t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                viewed_idx_iter.append(whale_idx)
                viewed_idx.append(whale_idx)

                print(
                        f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                        f"target={target_coord[:2]} | "
                        f"off nadir angle={offnadir_angle_deg:.2f}, in_view={in_view}, in_footprint={in_footprint}"
                )

                print("\tCue yaw, pitch, roll: ", yaw, pitch, roll, " deg")
                print("\tCue FovPoints: ", FovPoints)

                if boresight_hit is not None:
                    lat_b, lon_b, alt_b = boresight_hit
                    print(
                        f"\tCue boresight at lat={float(lat_b):.4f}, "
                        f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}" )

                    print(
                        f"\tTarget location at lat={float(tgt_lat):.4f}, "
                        f"lon={float(tgt_lon):.4f}, alt={float(tgt_alt):.1f}"
                    )

            if plot_footprints and in_view:
                eo_tools.plot_fov_on_map(FovPoints, ax_map)

        FovPoints_cue.append(FovPoints)

    for whale_idx in viewed_idx_iter:
        del available_targets_cue[whale_idx]

    if plot_propagation:
        update_earth_rotation_eci(earth_actor, t_datetime, earth_state)
        update_sun_light_eci(sun_light, t_datetime)

        # 2) satellites: you already have r (ECI) per actor
        tip_positions_eci = tip_positions  # keep as ECI
        cue_positions_eci = cue_positions
        cloud_tip_sats.points = sats_to_points_eci(tip_positions_eci)
        cloud_cue_sats.points = sats_to_points_eci(cue_positions_eci)

        # 3) whales: convert lat/lon/alt -> ECI at t, then to PV
        whales_tip.points = whales_to_points_eci(available_targets_tip, t_datetime)

        for whale_idx in viewed_idx:
            whales_cue.points[whale_idx] = whales_to_points_eci(available_targets_tip[whale_idx], t_datetime)

        # 4) FoV polygons: your FovPoints_tip / _cue are Nx2 lat/lon arrays per actor
        update_fov_layers_eci(
            tip_fill_meshes, tip_edge_meshes,
            cue_fill_meshes, cue_edge_meshes,
            FovPoints_tip, FovPoints_cue, t_datetime
        )

        pl.update()

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)
    elapsed_time += sim_step_seconds
    n_steps += 1

file_tip.close()
file_cue.close()

if show_orbits:
    # Plotting
    plot_orbits(trajectories)

if plot_footprints:
    ax_map.set_title("Last FOV Footprints")
    plt.show()

