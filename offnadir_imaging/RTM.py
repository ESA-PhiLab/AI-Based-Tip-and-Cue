import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from pySMARTS.main import SMARTSTimeLocation, IOUT_to_code
from matplotlib import pyplot as plt
import os

def get_combined_wavelength_range(spd_file_list, margin=200):
    """
    Combine wavelengths from multiple SPD files and return extended min/max range.
    """
    all_wavelengths = []
    for file_path in spd_file_list:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() and not line.strip().startswith("#"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            all_wavelengths.append(float(parts[0]))
                        except ValueError:
                            continue
    if not all_wavelengths:
        raise ValueError("No valid wavelength data found in SPD files.")
    return max(min(all_wavelengths) - margin, 0), max(all_wavelengths) + margin


def generate_solar_spd_from_target(
    datetime_utc,
    target_lat,
    target_lon,
    target_alt,
    sun_direction,
    output_path,
    wavelength_range=[350, 2500],
    timezone_offset=0,
    material="Water", plot_spd=False
):
    """
    Generate Mitsuba-compatible solar SPD using SMARTS for ground-level spectral irradiance.

    Parameters
    ----------
    datetime_utc : datetime.datetime
        UTC datetime of observation
    target_lat : float
        Target latitude in degrees
    target_lon : float
        Target longitude in degrees
    target_alt : float
        Altitude in meters
    sun_direction : np.ndarray
        Sun direction vector (must be 3D and normalized or normalized inside)
    output_path : str
        Path to write Mitsuba-compatible .spd file
    sensor_spd_files : list of str
        List of .spd files from the sensor to extract wavelength bounds
    timezone_offset : int
        Local timezone offset from UTC
    material : str
        SMARTS material (default: "Water")
    """


    min_wvl, max_wvl = wavelength_range[0], wavelength_range[1]  # default safe range

    # 2. Get correct SMARTS IOUT code
    IOUT_code = IOUT_to_code('Direct normal photon flux per wavelength cm-2 s-1 nm-1')
    if IOUT_code is None:
        raise ValueError("SMARTS IOUT type not supported or misspelled.")

    # 3. Run SMARTS
    df = SMARTSTimeLocation(
        IOUT=IOUT_code,
        YEAR=str(datetime_utc.year),
        MONTH=str(datetime_utc.month),
        DAY=str(datetime_utc.day),
        HOUR=f"{datetime_utc.hour + datetime_utc.minute / 60 + timezone_offset:.2f}",
        LATIT=str(target_lat),
        LONGIT=str(target_lon),
        ALTIT=str(target_alt / 1000.0),  # in km
        ZONE=str(timezone_offset),
        min_wvl=str(min_wvl),
        max_wvl=str(max_wvl),
        material=material
    )

    # 4. Project irradiance to ground surface
    sun_vec = np.array(sun_direction)
    sun_vec = sun_vec / np.linalg.norm(sun_vec)
    cos_theta = max(np.dot(sun_vec, np.array([0, 0, 1])), 0)

    # 5. Convert photon flux to spectral irradiance
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    cm2_to_m2 = 1e4
    nm_to_m = 1e-9

    wavelengths = df.iloc[:, 0].values  # [nm]
    photon_flux = df.iloc[:, 1].values  # [photons / cm² / s / nm]
    energies_per_photon = h * c / (wavelengths * nm_to_m)  # [J]
    irradiance_direct = photon_flux * energies_per_photon * cm2_to_m2  # [W/m²/nm]

    irradiance_ground = irradiance_direct * cos_theta

    # 6. Write to Mitsuba-compatible SPD file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for wl, val in zip(wavelengths, irradiance_ground):
            f.write(f"{wl:.2f} {val:.6f}\n")

    if plot_spd == True:
        plt.plot(wavelengths/1000, irradiance_ground*1000)
        plt.xlabel('Wavelength [um]')
        plt.ylabel('Spectral Irradiance [W/m²/um]')
        plt.grid(True)
        plt.show()


# === Example usage ===
if __name__ == "__main__":
    datetime_utc = datetime(2025, 6, 11, 16, 0, 0)
    target_lat = 0.0
    target_lon = 0.0
    target_alt = 0.0  # meters

    sun_direction = np.array([0.372249, -0.667976, 0.644391])  # Z-up frame

    script_dir = os.path.dirname(os.path.abspath(__file__))
    spd_dir = os.path.join(script_dir, 'spd_files')
    output_path = fr"{spd_dir}\generated_solar_spectrum.spd"
   #  sensor_spd_files = [
   #      fr"{spd_dir}\WV3_Red.spd",
   #      fr"{spd_dir}\WV3_Green.spd",
   #      fr"{spd_dir}\WV3_Blue.spd"
   #  ]

    generate_solar_spd_from_target(
        datetime_utc=datetime_utc,
        target_lat=target_lat,
        target_lon=target_lon,
        target_alt=target_alt,
        sun_direction=sun_direction,
        output_path=output_path,
        wavelength_range=[250, 1000],
        timezone_offset=0,
        material="Water", plot_spd=True
    )
