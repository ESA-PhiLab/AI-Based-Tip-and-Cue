import os
import pandas as pd
import re

def get_band_data(satellite, spd_folder):

    if satellite == 'WV3':

        band_data = {}
        band_data['red'] = {};                  band_data['green'] = {};                 band_data['blue'] = {}

        R_spd = os.path.join(spd_folder, str(satellite) + '_Red.spd')
        G_spd = os.path.join(spd_folder, str(satellite) + '_Green.spd')
        B_spd = os.path.join(spd_folder, str(satellite) + '_Blue.spd')

        band_data['red']['spd'] = R_spd
        band_data['green']['spd'] = G_spd
        band_data['blue']['spd'] = B_spd

    return band_data

def get_spatial_res(image_path, csv_path):
    df = pd.read_csv(csv_path)
    image_key = os.path.splitext(os.path.basename(image_path))[0].strip()
    image_key = re.sub(r"_\d+$", "", image_key)

    match = df[df['BoxID/ImageChip'].str.strip() == image_key]

    if match.empty:
        raise ValueError(f"No match found for image '{image_key}'")

    # Force native Python float
    return float(float(str(match.iloc[0]['SpatialRes']).replace('m', '').strip()))

def get_satellite(image_path, csv_path, fixed_sat = None):
    df = pd.read_csv(csv_path)
    image_key = os.path.splitext(os.path.basename(image_path))[0].strip()
    image_key = re.sub(r"_\d+$", "", image_key)

    match = df[df['BoxID/ImageChip'].str.strip() == image_key]

    if match.empty:
        raise ValueError(f"No match found for image '{image_key}'")

    if fixed_sat != None:
        return fixed_sat

    # Force native Python float
    return str(match.iloc[0]['Satellite'])

