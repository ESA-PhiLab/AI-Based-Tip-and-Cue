import os
import pandas as pd

def get_band_data(satellite, spd_folder):

    if satellite == 'WV3':
        # WV-3 https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/207/Radiometric_Use_of_WorldView-3_v2.pdf?
        abs_cal_factor = 9.295654 * 10 ** -3  # dummy value, differs per image
        Gain_R, offset_R, eff_bw_R = 0.945, -1.350, 5.85 * 10 ** -2
        Gain_G, offset_G, eff_bw_G = 0.907, -3.287, 6.18 * 10 ** -2
        Gain_B, offset_B, eff_bw_B = 0.905, -4.189, 5.40 * 10 ** -2

    band_data = {}
    band_data['abs_cal_factor'] = abs_cal_factor
    band_data['red'] = {};                  band_data['green'] = {};                 band_data['blue'] = {}
    band_data['red']['gain']   = Gain_R;    band_data['green']['gain']   = Gain_G;   band_data['blue']['gain']   = Gain_B
    band_data['red']['offset'] = offset_R;  band_data['green']['offset'] = offset_G; band_data['blue']['offset'] = offset_B;
    band_data['red']['eff_bw'] = eff_bw_R;  band_data['green']['eff_bw'] = eff_bw_G; band_data['blue']['eff_bw'] = eff_bw_B;

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
    match = df[df['BoxID/ImageChip'].str.strip() == image_key]

    if match.empty:
        raise ValueError(f"No match found for image '{image_key}'")

    # Force native Python float
    return float(float(str(match.iloc[0]['SpatialRes']).replace('m', '').strip()))

def get_satellite(image_path, csv_path):
    df = pd.read_csv(csv_path)
    image_key = os.path.splitext(os.path.basename(image_path))[0].strip()
    match = df[df['BoxID/ImageChip'].str.strip() == image_key]

    if match.empty:
        raise ValueError(f"No match found for image '{image_key}'")

    # Force native Python float
    return str(match.iloc[0]['Satellite'])

