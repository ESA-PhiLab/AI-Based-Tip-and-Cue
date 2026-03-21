# from offnadir_imaging.functions.get_satellite_data import get_satellite, get_spatial_res
from urllib.request import proxy_bypass

from paseos.custom_paseos.utils.help_functions import compute_orbital_period, fov_angle_from_swath, estimate_box_inertia, pass_time_from_nadir
from paseos.custom_paseos.utils.constants import R_earth, mu_earth

from simulation.satellite_config_utils import get_satellite_group_configs

from datetime import datetime, timezone, timedelta
import numpy as np
import math
import os

# from offnadir_imaging.functions.get_satellite_data import get_satellite

# ================================================================================
# SIMULATION

images_folder = "dataset/whales_from_space/"
img_file = "Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG"

csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
img_path = os.path.join(images_folder, img_file)

anns_folder = "dataset/create_dataset/"
anns_path = os.path.join(anns_folder, "final_annotations.json")

real_run = True

if real_run:
    print_values = False
    plot_3d = False
    plot_result = False
    max_glint = False
    generate_radiation = True
    generate_nadir = False
    flat_dem = False
    exclude_dark = True
else:
    print_values = False
    plot_3d = True
    plot_result = False
    max_glint = False
    generate_radiation = True
    generate_nadir = False
    flat_dem = False
    exclude_dark = True


t0 = datetime(2025, 9, 20, 2, 33, 39, tzinfo=timezone.utc)
whale_seed = 17

if not real_run:
    t0 += timedelta(seconds=-3 * 60)

if not real_run:
    sim_duration_hours = 15 / 60
    sim_time = "slow"
else:
    sim_duration_hours = 24
    sim_time = "slow"

# ================================================================================
# ORBIT / SATELLITE CONFIG

sat_group_cfg = get_satellite_group_configs()

tip_cfg = sat_group_cfg["Tip"]
cue_cfg = sat_group_cfg["Cue"]

nPlanes_tip = tip_cfg["nPlanes"]
nSats_tip = tip_cfg["nSats"]

nPlanes_cue = cue_cfg["nPlanes"]
nSats_cue = cue_cfg["nSats"]

a_tip = tip_cfg["a"]
a_cue = cue_cfg["a"]

i_tip_deg = tip_cfg["orbit"]["i_deg"] if tip_cfg["build_constellation"] else tip_cfg["i_deg"]
i_cue_deg = cue_cfg["orbit"]["i_deg"] if cue_cfg["build_constellation"] else cue_cfg["i_deg"]

params_tip = {
    "build_constellation": tip_cfg["build_constellation"],
    "nPlanes": tip_cfg["nPlanes"],
    "nSats": tip_cfg["nSats"],
}

params_cue = {
    "build_constellation": cue_cfg["build_constellation"],
    "nPlanes": cue_cfg["nPlanes"],
    "nSats": cue_cfg["nSats"],
}

if tip_cfg["build_constellation"]:
    params_tip.update({
        "a": tip_cfg["a"],
        "e": tip_cfg["e"],
        "i": tip_cfg["orbit"]["i_deg"],
        "RAAN": tip_cfg["orbit"]["RAAN_deg"],
        "argp": tip_cfg["orbit"]["argp_deg"],
        "M": tip_cfg["orbit"]["M_deg"],
        "sensor": tip_cfg["sensor"],
        "fov_deg": tip_cfg["fov_deg"],
    })
else:
    params_tip.update({
        "satellites": tip_cfg["satellites"],
    })

if cue_cfg["build_constellation"]:
    params_cue.update({
        "a": cue_cfg["a"],
        "e": cue_cfg["e"],
        "i": cue_cfg["orbit"]["i_deg"],
        "RAAN": cue_cfg["orbit"]["RAAN_deg"],
        "argp": cue_cfg["orbit"]["argp_deg"],
        "M": cue_cfg["orbit"]["M_deg"],
        "sensor": cue_cfg["sensor"],
        "fov_deg": cue_cfg["fov_deg"],
    })
else:
    params_cue.update({
        "satellites": cue_cfg["satellites"],
    })

# ================================================================================
# SIMULATION NAME / FILE NAMING

if not real_run:
    sim_name = "test1"
else:
    nm_ext = str()
    nm_ext += "T" if nSats_tip * nPlanes_tip > 0 else ""
    nm_ext += "C" if nSats_cue * nPlanes_cue > 0 else ""

    if tip_cfg["build_constellation"]:
        offnadir_label_tip = params_tip["sensor"]["offnadir_limit_deg"]
    else:
        offnadir_label_tip = max(sat["sensor"]["offnadir_limit_deg"] for sat in params_tip["satellites"]) if params_tip["satellites"] else 0.0

    sim_name = f"{nm_ext}_{nPlanes_cue}x{nSats_cue}sat_{int(offnadir_label_tip)}deg_{whale_seed}sd"

    if not tip_cfg["build_constellation"] or not cue_cfg["build_constellation"]:
        sim_name += "_independent"

sim_name = "".join(c if c not in '\\/:*?"<>|' else "_" for c in sim_name).rstrip(". ")

# ================================================================================
# TIME SETTINGS

if sim_time == "slow":
    sim_step_seconds = 1
    plot_fov_interval = 1
    plot_pyvista_interval = 20
    print_interval = 10
    movie_orbit_sec = 10.0
elif sim_time == "fast":
    sim_step_seconds = 6
    plot_fov_interval = 1
    plot_pyvista_interval = 5
    print_interval = 5
    movie_orbit_sec = 60.0
else:
    sim_step_seconds = 0.5
    plot_fov_interval = 1
    plot_pyvista_interval = 1
    print_interval = 10
    movie_orbit_sec = 30.0

# ================================================================================
# MISSION DELAYS / TASKING

delay_confirmation_tip = 90
delay_transmission_TC = 10
delay_confirmation_cue = 60
avg_time_delay = delay_confirmation_tip + delay_transmission_TC

# ================================================================================
# ONBOARD AI

parallel_observation_confirmation = False

tip_tpr = 0.85
tip_tnr = 0.85
seed_ai_tip = 42

cue_tpr = 0.85
cue_tnr = 0.85
seed_ai_cue = seed_ai_tip * 2

# ================================================================================
# SATELLITE BUS / ATTITUDE

sat_mass = 2800.0
sat_length, sat_width, sat_height = 4.5, 2.4, 2.2
area_d = 5.0
area_s = 12.0
cr_s = 1.5
cd = 2.2
J_sat = estimate_box_inertia(sat_mass, sat_length, sat_width, sat_height)

elevation_min = 10.0
offnadir_margin = 0.0

try:
    satellite = get_satellite(img_path=None, csv_path=None, fixed_sat="WV3")
except:
    satellite = "WV3"

refl_mode = "proxy"
refl_scale = None
refl_offset = None

omega_max_rad = np.deg2rad(3.86)
alpha_max_rad = np.deg2rad(1.43)
zeta = 0.8
wn_rad = 0.42

omega_stab_res = omega_max_rad / 10
alpha_stab_res = alpha_max_rad / 10

# ================================================================================
# WHALES

if real_run:
    n_targets = 500
else:
    n_targets = 3000


pos_fraction = 1.0

worldmap_dir = "dataset/worldmaps"
res_deg = 0.05
mask_tif = "land_mask.tif"
mask_npy = "land_mask.npy"

max_abs_lat = 70.0
observation_time_limit = 24 * 60 * 60

speed_mean = 1.5
speed_min = 0.2
speed_max = 6.0
speed_mean_reversion_per_s = 1.0 / 900.0
speed_noise_sigma = 0.30
turn_std_deg_per_sqrt_s = 2.0
land_avoid_max_tries = 12

# ================================================================================
# WAVES

seed_dem = 42
wind_speed = 10.0
num_waves = 50

if flat_dem:
    wave_min = 0.0
    wave_max = 0.0
else:
    wave_min = 0.05
    wave_max = 0.5

# ================================================================================
# DICTIONARIES

wave_properties = {}
wave_properties["wind_speed"] = wind_speed
wave_properties["num_waves"] = num_waves
wave_properties["wave_min"] = wave_min
wave_properties["wave_max"] = wave_max

bools = {}
bools["plot_3d"] = plot_3d
bools["plot_result"] = plot_result
bools["max_glint"] = max_glint
bools["print_values"] = print_values
bools["generate_radiation"] = generate_radiation
bools["generate_nadir"] = generate_nadir

sensor_characteristics = {}
sensor_characteristics["resolution"] = None
sensor_characteristics["sample_count"] = None
sensor_characteristics["specular_weight"] = None
sensor_characteristics["GSD"] = None
sensor_characteristics["swath_m"] = None
sensor_characteristics["offnadir_limit_deg"] = None

whale_propagation = {}
whale_propagation["speed_mean"] = speed_mean
whale_propagation["speed_min"] = speed_min
whale_propagation["speed_max"] = speed_max
whale_propagation["speed_mean_reversion_per_s"] = speed_mean_reversion_per_s
whale_propagation["speed_noise_sigma"] = speed_noise_sigma
whale_propagation["turn_std_deg_per_sqrt_s"] = turn_std_deg_per_sqrt_s
whale_propagation["land_avoid_max_tries"] = land_avoid_max_tries