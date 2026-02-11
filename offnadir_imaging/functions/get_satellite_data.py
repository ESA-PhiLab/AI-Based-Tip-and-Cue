import os
import pandas as pd
import re
from pathlib import Path

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

def normalize_image_key(image_path: str) -> str:
    """normalize_image_key(image_path) -> str: Strip patch suffixes and drop trailing chip index after band (e.g. ..._B0_1 -> ..._B0)."""
    key = os.path.splitext(os.path.basename(image_path))[0].strip()
    parts = key.split("_")

    strip_tokens = {"F", "H", "O", "nadir", "offnadir", "rot", "flip", "mirror"}

    # 1) remove trailing patch tokens like _F_nadir
    while parts and parts[-1] in strip_tokens:
        parts.pop()

    # 2) remove trailing numeric patch index if present (e.g. ..._B0_1 -> remove _1)
    if len(parts) >= 2 and parts[-1].isdigit() and re.fullmatch(r"B\d+", parts[-2]):
        parts.pop()

    return "_".join(parts)




def get_spatial_res(image_path: str, csv_path: str) -> float:
    """get_spatial_res(image_path,csv_path) -> float: Spatial resolution in meters from WhaleFromSpace CSV."""
    df = pd.read_csv(csv_path)

    image_key = normalize_image_key(image_path)
    match = df[df["BoxID/ImageChip"].astype(str).str.strip() == image_key]

    if match.empty:
        raise ValueError(f"No match found for image '{os.path.basename(image_path)}' (normalized to '{image_key}')")

    return float(str(match.iloc[0]["SpatialRes"]).replace("m", "").strip())


def get_satellite(image_path: str, csv_path: str, fixed_sat: str | None = None) -> str:
    """get_satellite(image_path,csv_path,fixed_sat=None) -> str: Satellite from WhaleFromSpace CSV."""
    if fixed_sat is not None:
        return fixed_sat

    df = pd.read_csv(csv_path)

    image_key = normalize_image_key(image_path)
    match = df[df["BoxID/ImageChip"].astype(str).str.strip() == image_key]

    if match.empty:
        raise ValueError(f"No match found for image '{os.path.basename(image_path)}' (normalized to '{image_key}')")

    return str(match.iloc[0]["Satellite"])

