import os
from pathlib import Path

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

from settings import *
from offnadir_imaging.rendering import generate_image
from settings import *

images_folder = "dataset/whales_from_space/"
img_file = 'Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG'

csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
img_path = os.path.join(images_folder, img_file)

sat_lat, sat_lon, sat_alt = 58.0, -5.0, 617000.0  # lat, lon, m
target_lat, target_lon, target_alt = 53.0, 0.0, 0.0  # lat, lon, me
datetime_utc = datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)

DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(img_path, satellite, sat_lat, sat_lon, sat_alt, target_lat, target_lon, target_alt, datetime_utc, sensor_characteristics, wave_properties, bools, seed_dem)


