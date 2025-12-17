# from offnadir_imaging.functions.get_satellite_data import get_satellite, get_spatial_res


from paseos.custom_paseos.utils.help_functions import compute_orbital_period, fov_angle_from_swath, estimate_box_inertia, pass_time_from_nadir
from paseos.custom_paseos.utils.constants import R_earth, mu_earth

from datetime import datetime, timezone, timedelta
import numpy as np
import math
import os

# ================================================================================
# SIMULATION

# images_folder = "dataset/whales_from_space/"
# img_file = 'Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG'

# csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
# img_path = os.path.join(images_folder, img_file)

print_values = True
plot_3d = False
plot_result = True
max_glint = False
crop_black_border = True
generate_radiation = True
flat_dem = False
exclude_dark = True

real_run = True

nSats_tip = 2
nSats_cue = 2

nPlanes_tip = 4
nPlanes_cue = 4

offnadir_limit = 40.0        # Maximum off-nadir observation angle (deg), max 62.5 deg
delta_t_tipcue = 5*60           # Time delay between Tip and Cue satellite (s)

whale_seed = 42

if not real_run:
    sim_duration_hours = 0.20
    sim_time = 'slow'

else:
    sim_duration_hours = 24
    sim_time = 'slow'

if not real_run:
    sim_name = "test1"
else:
    nm_ext = str()
    nm_ext += "T" if nSats_tip * nPlanes_tip > 0 else ""
    nm_ext += "C" if nSats_cue * nPlanes_cue > 0 else ""

    extension = f"_{int(offnadir_limit)}deg_{int(delta_t_tipcue/60)}min" if nSats_tip * nPlanes_tip > 0 else ""
    sim_name = f"{nm_ext}_{nPlanes_cue}x{nSats_cue}sat" + extension + f"_{whale_seed}sd"

sim_name = "".join(c if c not in '\\/:*?"<>|' else "_" for c in sim_name).rstrip(". ")

if sim_time == 'slow':
    sim_step_seconds = 1
    plot_fov_interval =  1
    plot_pyvista_interval = 20
    print_interval = 10
    movie_orbit_sec = 10.0

elif sim_time == 'fast':
    sim_step_seconds = 6
    plot_fov_interval = 1
    plot_pyvista_interval = 5
    print_interval = 5
    movie_orbit_sec = 60.0

else:
    sim_step_seconds = 1
    plot_fov_interval =  1
    plot_pyvista_interval = 1
    print_interval = 10
    movie_orbit_sec = 30.0

# ================================================================================
# ORBIT

t0 = datetime(2025, 9, 20, 2, 33, 39, tzinfo=timezone.utc)

hp           = 615.7e3              # Perigee altitude [m]
ha           = 624.6e3              # Apogee altitude [m]
i_cue_deg    = 97.8703              # Inclination [deg]
RAAN_cue_deg = 336.4191             # RAAN [deg]
argp_cue_deg = 110.0511             # Argument of periapsis [deg]
M_cue_deg    = 250.1394             # Mean anomaly [deg]

if not real_run:
    M_cue_deg += 147
    t0 += timedelta(seconds=-3*60)

rp = R_earth + hp
ra = R_earth + ha
a_cue = 0.5 * (ra + rp)             # Semi-major axis [m]
e_cue = (ra - rp) / (ra + rp)       # Eccentricity [-]

a_tip = a_cue                       # Semi-major axis [m]
e_tip = e_cue                       # Eccentricity
i_tip_deg = i_cue_deg               # Inclination [deg]
RAAN_tip_deg = RAAN_cue_deg         # RAAN [deg]
argp_tip_deg = argp_cue_deg         # Argument of periapsis [deg]

delta_M = 360.0 * (delta_t_tipcue / compute_orbital_period(a_cue))
M_tip_deg = M_cue_deg + delta_M

delay_confirmation_tip = 90      # https://www.jpl.nasa.gov/news/how-nasa-is-testing-ai-to-make-earth-observing-satellites-smarter
delay_transmission_TC = 10
delay_confirmation_cue = 60      # Time delay after detection by cue [sec]
avg_time_delay = delta_t_tipcue

# ================================================================================
# ONBOARD AI
parallel_observation_confirmation = False

tip_tpr = 0.85    # probability Tip correctly identifies a positive whale
tip_tnr = 0.85    # probability Tip correctly ignores a negative whale
seed_ai_tip = 42

cue_tpr = 0.85    # probability Tip correctly identifies a positive whale
cue_tnr = 0.85    # probability Tip correctly ignores a negative whale
seed_ai_cue = seed_ai_tip*2

# ================================================================================
# SATELLITE

sat_mass = 2800.0                                   # kg
sat_length, sat_width, sat_height = 4.5, 2.4, 2.2   # m (length, width, height)
area_d = 5.0                                        # drag area
area_s = 12.0                                       # sat area + solar panels
cr_s = 1.5
cd = 2.2
J_sat = estimate_box_inertia(sat_mass, sat_length, sat_width, sat_height)                               # Principal moments of inertia Cue satellite (kg m^2/s)

# ================================================================================
# SENSOR

elevation_min = 10.0                                    # Minimal elevation [deg]
offnadir_margin = offnadir_limit * 0.02                 # Margin to add allow observation at offnadir max

resolution = 124                                        # Resolution of render
sample_count = 512                                      # 8192 min, 2048 * 2**7 max

swath_tip = 290  * 10**3  # m
swath_cue = 13.1 * 10**3  # m

fov_tip = math.degrees(2 * math.atan(swath_tip / (2 * (a_tip - R_earth))) )         # deg
fov_cue = math.degrees(2 * math.atan(swath_cue / (2 * (a_cue - R_earth))) )         # deg

gsd0_tip = 10.0       # m
gsd0_cue = 0.31     # m

try:
    satellite = get_satellite(img_path, csv_path)

except:
    # print("Got default settings")
    satellite = 'WV3'

# ================================================================================
# ATTITUDE

omega_max_rad = np.deg2rad(3.86)
alpha_max_rad = np.deg2rad(1.43)
zeta = 0.8
wn_rad = 0.42

omega_stab_res = omega_max_rad/10
alpha_stab_res = alpha_max_rad/10

# ================================================================================
# WHALES

if real_run:
    n_targets = 500

if not real_run:
    n_targets = 500
    whale_seed = 42

pos_fraction = 1.0

worldmap_dir = "dataset/worldmaps"      # Folder with GSHHS shapefiles; mask .tif/.npy will be stored here
res_deg = 0.05                          # Raster resolution for land mask (deg/pixel)
mask_tif = "land_mask.tif"
mask_npy = "land_mask.npy"


max_abs_lat = 70.0                   # Optional: exclude very high latitudes (avoid polar mask artifacts)
observation_time_limit = 20*60       # Observation time limit

# Whale kinematics
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
wind_speed = 10.0  # m/s
num_waves = 50

if flat_dem:
    wave_min = 0.0  # m
    wave_max = 0.0  # m

else:
    wave_min = 0.05  # m
    wave_max = 0.5  # m

# ================================================================================
# DICTIONARIES

params_tip = {}
params_tip["a"] = a_tip
params_tip["e"] = e_tip
params_tip["i"] = i_tip_deg
params_tip["RAAN"] = RAAN_tip_deg
params_tip["argp"] = argp_tip_deg
params_tip["M"] = M_tip_deg
params_tip["nSats"] = nSats_tip
params_tip["nPlanes"] = nPlanes_tip

params_cue = {}
params_cue["a"] = a_cue
params_cue["e"] = e_cue
params_cue["i"] = i_cue_deg
params_cue["RAAN"] = RAAN_cue_deg
params_cue["argp"] = argp_cue_deg
params_cue["M"] = M_cue_deg
params_cue["nSats"] = nSats_cue
params_cue["nPlanes"] = nPlanes_cue

wave_properties = {}
wave_properties['wind_speed'] = wind_speed
wave_properties['num_waves'] = num_waves
wave_properties['wave_min'] = wave_min
wave_properties['wave_max'] = wave_max

bools = {}
bools['plot_3d'] = plot_3d
bools['plot_result'] = plot_result
bools['max_glint'] = max_glint
bools['print_values'] = print_values
bools['crop_black_border'] = crop_black_border
bools['generate_radiation'] = generate_radiation

sensor_characteristics = {}
sensor_characteristics['resolution'] = resolution
sensor_characteristics['sample_count'] = sample_count
sensor_characteristics['GSD'] = gsd0_cue

whale_propagation = {}
whale_propagation["speed_mean"] = speed_mean
whale_propagation["speed_min"] = speed_min
whale_propagation["speed_max"] = speed_max
whale_propagation["speed_mean_reversion_per_s"] = speed_mean_reversion_per_s
whale_propagation["speed_noise_sigma"] = speed_noise_sigma
whale_propagation["turn_std_deg_per_sqrt_s"] = turn_std_deg_per_sqrt_s
whale_propagation["land_avoid_max_tries"] = land_avoid_max_tries


