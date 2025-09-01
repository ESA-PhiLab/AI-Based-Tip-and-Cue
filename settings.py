import os
from offnadir_imaging.functions.get_satellite_data import get_satellite, get_spatial_res
from custom_paseos.utils.help_functions import compute_orbital_period, fov_angle_from_swath
from datetime import datetime, timezone
import numpy as np
import math


# ================================================================================
# SIMULATION SETTINGS

images_folder = "dataset/whales_from_space/"
img_file = 'Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG'

csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
img_path = os.path.join(images_folder, img_file)

print_values = True
plot_3d = True
plot_result = True
max_glint = False
crop_black_border = True
generate_radiation = True
flat_dem = False
exclude_dark = True

R_earth = 6371e3
t0 = datetime(2025, 8, 19, 12, 53, 22, tzinfo=timezone.utc)

# Simulation config
sim_duration_hours = 24 * 3
sim_step_seconds = 10
sim_duration_seconds = sim_duration_hours * 3600

reset_propagation_interval = 12
reset_plot_interval = 10000

# ================================================================================
# SATELLITE

nSats_tip = 0
nSats_cue = 1

nPlanes_tip = 0
nPlanes_cue = 2

a_tip = 613e3 + R_earth        # Semi-major axis [m]
e_tip = 0.0004466              # Eccentricity
i_tip_deg = 97.8739            # Inclination [deg]
RAAN_tip_deg = 305.5939        # RAAN [deg]
argp_tip_deg = 201.0444         # Argument of periapsis [deg]
M_tip_deg = 180+ 159.0587           # Mean anomaly [deg]

delta_t_cue = 5*60  # seconds
tasking_delay_tip = 3 * 60  # Time delay between tip and cue transfer
tasking_delay_cue = 10      # Time delay after detection by cue

a_cue = a_tip                     # Semi-major axis [m]
e_cue = e_tip                     # Eccentricity
i_cue_deg = i_tip_deg             # Inclination [deg]
RAAN_cue_deg = RAAN_tip_deg       # RAAN [deg]
argp_cue_deg = argp_tip_deg       # Argument of periapsis [deg]

delta_M_cue = 360.0 * (delta_t_cue / compute_orbital_period(a_tip))
M_cue_deg = M_tip_deg - delta_M_cue

satellite_mass = 9.54
area_d = 0.164
area_s = 0.162
cr_s = 1.5
cd = 2.2

# ================================================================================
# WHALES

worldmap_dir = "dataset/worldmaps"     # Folder with GSHHS shapefiles; mask .tif/.npy will be stored here
res_deg = 0.05                     # Raster resolution for land mask (deg/pixel)
mask_tif = "land_mask.tif"
mask_npy = "land_mask.npy"

n_whales = 300
whale_seed = 17
max_abs_lat = 70.0                 # Optional: exclude very high latitudes (avoid polar mask artifacts)
detection_time_limit = 20*60       # Detection time limit

# Whale kinematics
speed_mean = 1.5
speed_min = 0.2
speed_max = 6.0
speed_mean_reversion_per_s = 1.0 / 900.0
speed_noise_sigma = 0.30
turn_std_deg_per_sqrt_s = 2.0
land_avoid_max_tries = 12

# ================================================================================
# SENSOR

elevation_min = 10.0 # degrees
offnadir_max = 50.0     # max 62.5 deg

resolution = 124  # pixels of render
sample_count = 512  # 8192 min, 2048 * 2**7 max

swath_tip = 290  * 10**3  # m
swath_cue = 13.1 * 10**3  # m

fov_tip = math.degrees(2 * math.atan(swath_tip / (2 * (a_tip - R_earth))) )         # deg
fov_cue = math.degrees(2 * math.atan(swath_cue / (2 * (a_cue - R_earth))) )         # deg

try:
    satellite = get_satellite(img_path, csv_path)
    GSD = 500 # get_spatial_res(img_path, csv_path)

except:
    print("Got default settings")
    satellite = 'WV3'
    GSD = 0.5

# ================================================================================
# WAVES

dem_seed = 42
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
sensor_characteristics['GSD'] = GSD

whale_propagation = {}
whale_propagation["speed_mean"] = speed_mean
whale_propagation["speed_min"] = speed_min
whale_propagation["speed_max"] = speed_max
whale_propagation["speed_mean_reversion_per_s"] = speed_mean_reversion_per_s
whale_propagation["speed_noise_sigma"] = speed_noise_sigma
whale_propagation["turn_std_deg_per_sqrt_s"] = turn_std_deg_per_sqrt_s
whale_propagation["land_avoid_max_tries"] = land_avoid_max_tries



