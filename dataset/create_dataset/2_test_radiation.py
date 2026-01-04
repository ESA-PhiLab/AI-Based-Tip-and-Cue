import os
from pathlib import Path
from read_and_write_data import pick_random_pose

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

from settings import *
from offnadir_imaging.rendering import generate_image
from settings import *

images_folder = "dataset/whales_from_space/"
black_folder = "dataset/utils_images/"

# img_file = 'Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG'
img_file = 'Pelagos2016/PelagosIm2_FW_WV3_PS_20160619_B2.PNG'
# img_file = 'Pelagos2016/PelagosIm5_FW_WV3_PS_20160626_B1.PNG'

csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
img_path = os.path.join(images_folder, img_file)

SCRIPT_DIR = Path(__file__).resolve().parent
poses_xlsx = SCRIPT_DIR / "combined_results.xlsx"

sat_lat, sat_lon, sat_alt = 58.0, -5.0, 617000.0  # lat, lon, m
tgt_lat, tgt_lon, tgt_alt = 53.0, 0.0, 0.0  # lat, lon, me
dt = datetime(2025, 6, 11, 12, 0, 0, tzinfo=timezone.utc)

# result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(poses_xlsx, pick_pose_seed=14)
# dt = ( datetime.fromisoformat(datetime_utc.replace("Z", "+00:00")).astimezone(timezone.utc) if datetime_utc else datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc) )

DN255_texture, DN255_no_glint, DN255_glint, radiance_glint, black_mask_full, scale = generate_image(img_path, satellite, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, dt, sensor_characteristics, wave_properties, bools, seed_dem)


