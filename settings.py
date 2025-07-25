import os
from offnadir_imaging.functions.get_satellite_data import get_satellite, get_spatial_res
from datetime import datetime, timezone

images_folder = "dataset/whales_from_space/"
img_file = 'Pelagos2016/PelagosIm2_FW_WV3_PS_20160619_B11.PNG'

csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
img_path = os.path.join(images_folder, img_file)

print_values = True
plot_3d = False
plot_result = True
max_glint = False
crop_black_border = True

satellite_lat, satellite_lon, satellite_alt = 58, -5, 617000.0  # lat, lon, m
target_lat, target_lon, target_alt = 53, 0, 0.0  # lat, lon, meters
datetime_utc = datetime(2025, 6, 11, 14, 30, 0, tzinfo=timezone.utc)

satellite = get_satellite(img_path, csv_path)
GSD = get_spatial_res(img_path, csv_path)
resolution = 124  # pixels of render
sample_count = 512  # 8192 min, 2048 * 2**7 max

dem_seed = 42

wind_speed = 10.0  # m/s
num_waves = 50
wave_min = 0.05  # m
wave_max = 0.5  # m

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

sensor_characteristics = {}
sensor_characteristics['resolution'] = resolution
sensor_characteristics['sample_count'] = sample_count
sensor_characteristics['GSD'] = GSD