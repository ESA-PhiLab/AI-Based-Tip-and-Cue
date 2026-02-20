from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcinv


# ---------------- USER SETTINGS ----------------
OUTPUT_DIR: str = r"generated_output"
POINTS_PER_BAND: int = 2000
EDGE_FRAC: float = 0.05          # transition width relative to effective bandwidth
EPSILON: float = 1 - 1 / math.sqrt(2)   # δ = 1/sqrt(2)  (-3.01 dB)
PAD_MULT: float = 4.0            # extend wavelength range to see both edges
NORMALIZE_PEAK_TO_1: bool = True
RGB_ONLY: bool = False
# ------------------------------------------------


# ===== EXACT TABLE FROM YOUR IMAGE =====
# Center wavelength in nm
# Effective bandwidth Δλ in µm (converted to nm below)
WV3_TABLE = pd.DataFrame(
    [
        ["Panchromatic", 649.4, 0.2896],
        ["Coastal", 427.4, 0.0405],
        ["Blue", 481.9, 0.0540],
        ["Green", 547.1, 0.0618],
        ["Yellow", 604.3, 0.0381],
        ["Red", 660.1, 0.0585],
        ["Red Edge", 722.7, 0.0387],
        ["NIR1", 824.0, 0.1004],
        ["NIR2", 913.6, 0.0889],
        ["SWIR1", 1209.1, 0.0330],
        ["SWIR2", 1571.6, 0.0397],
        ["SWIR3", 1661.1, 0.0373],
        ["SWIR4", 1729.5, 0.0416],
        ["SWIR5", 2163.7, 0.0389],
        ["SWIR6", 2202.2, 0.0409],
        ["SWIR7", 2259.3, 0.0476],
        ["SWIR8", 2329.2, 0.0679],
    ],
    columns=["Band", "Center_nm", "Bandwidth_um"],
)


BAND_COLORS = {
    "Panchromatic": "black",
    "Coastal": "deepskyblue",
    "Blue": "blue",
    "Green": "green",
    "Yellow": "gold",
    "Red": "red",
    "Red Edge": "crimson",
    "NIR1": "purple",
    "NIR2": "magenta",
}


def make_erfc_bandpass(center_nm, bandwidth_um,
                       num=2000,
                       edge_frac=0.10,
                       epsilon=0.292893218,
                       pad_mult=4.0,
                       normalize_peak=True):
    """
    erfc-based band-pass exactly following Apostolov (2024).
    """
    bw_nm = bandwidth_um * 1000.0  # convert µm → nm

    L = center_nm - bw_nm / 2.0
    U = center_nm + bw_nm / 2.0

    delta_lambda = edge_frac * bw_nm
    beta = float(erfcinv(2.0 * epsilon) / delta_lambda)

    wl_min = L - pad_mult * delta_lambda
    wl_max = U + pad_mult * delta_lambda
    wl = np.linspace(wl_min, wl_max, num)

    H_hp = 1.0 - 0.5 * erfc(2.0 * beta * (wl - L))
    H_lp = 0.5 * erfc(2.0 * beta * (wl - U))
    H = H_hp * H_lp

    if normalize_peak and np.max(H) > 0:
        H = H / np.max(H)

    return wl.astype(np.float64), H.astype(np.float64)


def save_spd(path, wl, resp):
    with open(path, "w", encoding="utf-8") as f:
        for x, y in zip(wl, resp):
            f.write(f"{x:.6f} {y:.6f}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    table = WV3_TABLE.copy()

    if RGB_ONLY:
        table = table[table["Band"].isin(["Blue", "Green", "Red"])]

    spd_records = []

    for _, row in table.iterrows():
        band = row["Band"]
        center = row["Center_nm"]
        bw_um = row["Bandwidth_um"]

        wl, rr = make_erfc_bandpass(
            center_nm=center,
            bandwidth_um=bw_um,
            num=POINTS_PER_BAND,
            edge_frac=EDGE_FRAC,
            epsilon=EPSILON,
            pad_mult=PAD_MULT,
            normalize_peak=NORMALIZE_PEAK_TO_1,
        )

        out_name = f"WV3_{band.replace(' ', '')}.spd"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        save_spd(out_path, wl, rr)

        spd_records.append((band, wl, rr))

    # Plot
    plt.figure(figsize=(10, 6))
    for band, wl, rr in spd_records:
        plt.plot(wl, rr, label=band, linewidth=2,
                 color=BAND_COLORS.get(band, None))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Response")
    plt.title("WV3 Spectral Response (erfc band-pass)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "WV3_bandpass_erfc.png"), dpi=200)
    plt.close()

    print("Done. SPDs written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
