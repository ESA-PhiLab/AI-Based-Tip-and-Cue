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
# img_file = "Witsand2009/Witsand_SRW_GE1_PS_20090809_B13.PNG"

csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
img_path = os.path.join(images_folder, img_file)

anns_folder = Path(__file__).resolve().parent
anns_path = str(anns_folder / "final_annotations.json")

SCRIPT_DIR = Path(__file__).resolve().parent
poses_xlsx = SCRIPT_DIR / "combined_results.xlsx"

sat_lat, sat_lon, sat_alt = 52.377956,  4.897070, 617000.0  # lat, lon, m
tgt_lat, tgt_lon, tgt_alt = 41.902782,  12.496366, 0.0  # lat, lon, me
dt = datetime(2025, 6, 21, 3, 36, 0, tzinfo=timezone.utc)


bools["plot_3d"] = True

# result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(poses_xlsx, pick_pose_seed=14)
# dt = ( datetime.fromisoformat(datetime_utc.replace("Z", "+00:00")).astimezone(timezone.utc) if datetime_utc else datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc) )

texture_disp, radiance_no_glint, radiance_disp_no_glint, rho_no_glint, rho_disp_no_glint, radiance_final, radiance_disp_final, rho_final, rho_disp_final, black_mask_full, scale, offnadir_deg = generate_image(img_path, anns_path, satellite, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, dt, sensor_characteristics, wave_properties, bools, seed_dem)


