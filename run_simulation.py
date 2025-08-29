from settings import *

import orekit
import pyvista as pv
import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.time import AbsoluteDate, TimeScalesFactory

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from custom_paseos.propagation.get_constellation import get_constellation
from custom_paseos.observation.EarthObservation import EOTools

from custom_paseos.utils.point_transformation import Point_ECI2Geodetic

from custom_paseos.plot_functions.plot_functions import plot_constallation, plot_orbits
from simulation.plot_pyvista import make_plotter_eci, update_earth_rotation_eci, whales_to_points_eci, sats_to_points_eci, init_fov_layers_eci, update_fov_layers_eci, init_sun_light, update_sun_light_eci, points_array_from_targets

from simulation.propagate_whales import step_whale, load_land_mask, generate_random_water_targets,  init_whales, build_land_mask

import cartopy.crs as ccrs
import cartopy.feature as cfeature

show_constellation = True
plot_propagation = True
plot_footprints = True
show_orbits = False
generate_image = False

# Initialize Orekit
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)
pv.global_theme.allow_empty_mesh = True
paseos.set_log_level("WARNING")

# Time setup
utc = TimeScalesFactory.getUTC()
t0_orekit = AbsoluteDate(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond / 1e6, utc)
t0_pykep = pk.epoch_from_string(t0.strftime("%Y-%m-%d %H:%M:%S"))

# Get constellations
if nSats_tip != 0:
    planet_lst_tip, sats_tip, orbital_period_tip = get_constellation(
        a_tip, e_tip, i_tip_deg, RAAN_tip_deg, argp_tip_deg, M_tip_deg,
        nSats_tip, nPlanes_tip, t0_pykep, "Tip", verbose=True
    )
else:
    planet_lst_tip, sats_tip, orbital_period_tip = [], [], []


if nSats_cue != 0:
    planet_lst_cue, sats_cue, orbital_period_cue = get_constellation(
        a_cue, e_cue, i_cue_deg, RAAN_cue_deg, argp_cue_deg, M_cue_deg,
        nSats_cue, nPlanes_cue, t0_pykep, "Cue", verbose=True
    )
else:
    planet_lst_cue, sats_cue, orbital_period_cue = [], [], []


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

if len(tip_actors) != 0:
    sim = paseos.init_sim(local_actor=tip_actors[0])
    for actor in tip_actors[1:] + cue_actors:
        sim.add_known_actor(actor)

else:
    sim = paseos.init_sim(local_actor=cue_actors[0])
    for actor in cue_actors[1:]:
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

all_targets = init_whales(known_targets, seed_val=whale_seed)       # live updated
tasked_targets = {}                                                 # live updated

detected_targets = {}                                           # to keep track of history
evaluated_targets = {}


if plot_propagation:
    pl, earth_actor, earth_state = make_plotter_eci()

    # All whales (fixed size == len(all_targets)): keep as-is
    whales_plot_all = pv.PolyData(np.zeros((len(all_targets), 3)))
    pl.add_points(whales_plot_all, color="red", point_size=8, render_points_as_spheres=True)

    # Evaluated and Tasked whales are dynamic-size clouds: initialize EMPTY PolyData
    whales_plot_evaluated = pv.PolyData(np.empty((0, 3)))
    pl.add_points(whales_plot_evaluated, color="green", point_size=9, render_points_as_spheres=True)

    whales_plot_tasked = pv.PolyData(np.empty((0, 3)))
    pl.add_points(whales_plot_tasked, color="orange", point_size=10, render_points_as_spheres=True)

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

    for whale_idx, w in all_targets.items():
        w = step_whale(w, mask, res_deg, dt_sec=sim_step_seconds, whale_propagation=whale_propagation)
        all_targets[whale_idx] = w

        if whale_idx in tasked_targets.keys():
           tasked_targets[whale_idx]["lat"] = w['lat']
           tasked_targets[whale_idx]["lon"] = w['lon']
           tasked_targets[whale_idx]["alt"] = w['alt']


    # delete tasked target list if out of range
    del_idxs = []
    for whale_idx, w in tasked_targets.items():
        detection_time = tasked_targets[whale_idx]["detection_time"]

        if t_datetime > detection_time + timedelta(seconds=detection_time_limit):
            del_idxs.append(whale_idx)

    for whale_idx in del_idxs:
        print("!!  Time-out: remove tasking request", whale_idx)
        del tasked_targets[whale_idx]

    for actor in tip_actors:

        n_detections = 0
        tip_detected = False

        eo_tools = eo_tools_dict[actor.name]
        r, v = trajectories[actor.name]["r"][-1], trajectories[actor.name]["v"][-1]

        r_vec = np.array(r).reshape(3, 1)
        v_vec = np.array(v).reshape(3, 1)

        tip_positions.append(r)

        FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_tip, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth
        boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_tip, t_datetime)

        FovPoints_tip.append(FovPoints)

        for whale_idx, whale in all_targets.items():

            tgt_lat, tgt_lon, tgt_alt = (whale["lat"], whale["lon"], whale["alt"])
            target_coord = (tgt_lat, tgt_lon, tgt_alt)
            in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

            if in_footprint:

                w = {"lat": tgt_lat, "lon": tgt_lon, "alt": tgt_alt, "detection_time": t_datetime, "detection_satellite": actor.name}
                tasked_targets[whale_idx] = w       # for live updates
                tasked_targets[whale_idx]["tasking_delay"] = tasking_delay_tip
                detected_targets[whale_idx] = w            # to keep track of history
                
                all_targets[whale_idx]["detected"] = 1

                n_detections +=1

                print("!! Tip: Target detected", whale_idx)

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

            # Print where the tip satellite is positioned
            tip_lat, tip_lon, tip_alt = Point_ECI2Geodetic(r_vec[0], r_vec[1], r_vec[2], t_datetime)
            print(
                f"\tTip position  at lat={float(tip_lat):.4f}, "
                f"lon={float(tip_lon):.4f}, alt={float(tip_alt):.1f}"
            )

            # Print where center ray intersects Earth
            lat_b, lon_b, alt_b = boresight_hit
            print(
                f"\tTip boresight at lat={float(lat_b):.4f}, "
                f"lon={float(lon_b):.4f}, alt={float(alt_b):.1f}"
            )

        if plot_footprints:
            eo_tools.plot_fov_on_map(FovPoints, ax_map)

    for actor in cue_actors:
        n_evaluated = 0
        cue_evaluated = False

        eo_tools = eo_tools_dict[actor.name]
        r, v = trajectories[actor.name]["r"][-1], trajectories[actor.name]["v"][-1]

        r_vec = np.array(r).reshape(3, 1)
        v_vec = np.array(v).reshape(3, 1)

        cue_positions.append(r)

        cue_busy = False

        for whale_idx, whale in tasked_targets.items():
            detection_time = tasked_targets[whale_idx]["detection_time"]
            tasking_delay = tasked_targets[whale_idx]["tasking_delay"]

            if t_datetime > detection_time + timedelta(seconds=tasking_delay): # Skip if not yet transmitted

                tgt_lat, tgt_lon, tgt_alt = (whale["lat"], whale["lon"], whale["alt"])
                target_coord = (tgt_lat, tgt_lon, tgt_alt)

                in_view = eo_tools.is_in_sight(
                    target_geodetic=target_coord,
                    r_eci=r_vec,
                    v_eci=v_vec,
                    time=t_datetime,
                    el_min=elevation_min
                )

                if in_view:

                    offnadir_angle_deg, vec_brf = eo_tools.off_nadir_pointing_angle(
                        z_brf=z_brf, r_eci=r_vec, v_eci=v_vec,
                        target_geodetic=target_coord,
                        eul_angles_deg=[0.0, 0.0, 0.0],
                        time=t_datetime
                    )

                    if offnadir_angle_deg <= offnadir_max:

                        yaw, pitch, roll = eo_tools.pointing_attitude(z_brf, vec_brf, phi_rad, [0.0, 0.0, 0.0], in_view)
                        eul_ang_cue = [yaw, pitch, roll]

                        print("!! Cue: Target in view", whale_idx, " | Set yaw, pitch, roll to:", eul_ang_cue)
                        cue_busy = True

        if not cue_busy:
            eul_ang_cue = [0.0, 0.0, 0.0]


        try:
            FovPoints = eo_tools.get_FovPoints(r_vec, v_vec, eul_ang_cue, t_datetime)  # check off-nadir angle, and where the center ray intersects the Earth

        except:
            print("!! Cue: target set out of sight, continue to the next step")
            eul_ang_cue = [0.0, 0.0, 0.0]
            continue

        boresight_hit = eo_tools.get_CenterRay_Intersection(r_vec, v_vec, eul_ang_cue, t_datetime)
        FovPoints_cue.append(FovPoints)
            
        for whale_idx, whale in all_targets.items():

            tgt_lat, tgt_lon, tgt_alt = (whale["lat"], whale["lon"], whale["alt"])
            target_coord = (tgt_lat, tgt_lon, tgt_alt)
            
            in_footprint = eo_tools.check_point_in_footprint(target_coord, FovPoints)

            if in_footprint:

                print("!! Cue: Target in footprint", whale_idx, " | Evaluate target")
                w = {"lat": tgt_lat, "lon": tgt_lon, "alt": tgt_alt, "detection_time": t_datetime, "detection_satellite": actor.name}
                detected_targets[whale_idx] = w  # to keep track of history
                tasked_targets[whale_idx] = w  # for live updates
                tasked_targets[whale_idx]["tasking_delay"] = tasking_delay_cue
                n_evaluated += 1

                all_targets[whale_idx]["detected"] = 2      # detected by cue

                file_cue.write(
                    f"{t_datetime.isoformat()},{actor.name},"
                    f"{whale_idx}, {tgt_lat:.4f},{tgt_lon:.4f},{tgt_alt:.1f},"
                    f"{r[0]:.3f},{r[1]:.3f},{r[2]:.3f},"
                    f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},"
                    f"{offnadir_angle_deg:.4f},{in_view},{in_footprint},"
                    f"{yaw:.2f},{pitch:.2f},{roll:.2f}"
                )

                cue_evaluated = True

                satellite_lat, satellite_lon, satellite_alt = Point_ECI2Geodetic(r[0], r[1], r[2],t_datetime).flatten()
                satellite_lat, satellite_lon, satellite_alt = float(satellite_lat), float(satellite_lon), float(satellite_alt)  # meters above ellipsoid

                if generate_image:
                    print("Generate image")
                    DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(
                        img_path, satellite, satellite_lat, satellite_lon, satellite_alt, tgt_lat, tgt_lon,
                        tgt_alt, t_datetime, sensor_characteristics, wave_properties, bools, dem_seed)

                evaluated_targets[whale_idx] = w
                del tasked_targets[whale_idx]

                if plot_footprints:
                    eo_tools.plot_fov_on_map(FovPoints, ax_map)

                print(
                    f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                    f"target={target_coord[:2]} | "
                    f"off nadir angle={offnadir_angle_deg:.2f}, in_view={in_view}, in_footprint={in_footprint}"
                )

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

                    print(
                        f"\tTarget location at lat={float(tgt_lat):.4f}, "
                        f"lon={float(tgt_lon):.4f}, alt={float(tgt_alt):.1f}"
                    )

        if cue_evaluated == False:
            print(
                f"{n_steps} {actor.name} | {t_datetime.isoformat()} | "
                f"detections={n_evaluated}")

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

    print("\n")

    if plot_propagation:
        update_earth_rotation_eci(earth_actor, t_datetime, earth_state)
        update_sun_light_eci(sun_light, t_datetime)

        # satellites (ECI)
        tip_positions_eci = tip_positions
        cue_positions_eci = cue_positions
        cloud_tip_sats.points = sats_to_points_eci(tip_positions_eci)
        cloud_cue_sats.points = sats_to_points_eci(cue_positions_eci)

        # all whales (fixed size)
        whales_plot_all.points = whales_to_points_eci(all_targets, t_datetime)

        # evaluated whales (dynamic): rebuild the PolyData each frame via shallow_copy
        eval_pts = points_array_from_targets(evaluated_targets, t_datetime)
        whales_plot_evaluated.shallow_copy(pv.PolyData(eval_pts))

        # tasked whales (dynamic): rebuild each frame; removed ones disappear automatically
        task_pts = points_array_from_targets(tasked_targets, t_datetime)
        whales_plot_tasked.shallow_copy(pv.PolyData(task_pts))

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


file_tip.close()
file_cue.close()

if show_orbits:
    # Plotting
    plot_orbits(trajectories)

if plot_footprints:
    ax_map.set_title("Last FOV Footprints")
    plt.show()

