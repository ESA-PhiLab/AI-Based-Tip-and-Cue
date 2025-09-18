"""
WV3 table-to-SPD generator (no clicking):
- Optionally OCR a PNG/JPG of the WV3 band table to extract
  Center, Lower, Upper wavelengths (nm).
- Falls back to hardcoded WV3 table if OCR misses anything.
- Generates smooth band response curves (triangular / Hann / Gaussian)
  with dense sampling (>= 1000 points per band).
- Writes one .spd per band and a summary plot.

Requirements (install in your PyCharm env as needed):
  pip install numpy pandas opencv-python pillow pytesseract matplotlib

Also install Tesseract OCR engine on your OS and make sure the binary is on PATH
(or set TESSERACT_CMD below).

Usage:
  - Set TABLE_IMAGE to your PNG path (or leave None to skip OCR and use fallback).
  - Run.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------- USER SETTINGS -----------
TABLE_IMAGE = None  # e.g., r"C:\path\to\WV3_table.png"  (set to None to skip OCR and use fallback)
OUTPUT_DIR = r"./wv3_spd_output"
CURVE_SHAPE = "hann"   # one of: "triangle", "hann", "gaussian"
POINTS_PER_BAND = 1500  # output density per band (>=100 is fine; 1000+ captures subtle shape)
RGB_ONLY = False        # True -> only make Blue/Green/Red .spd files
TESSERACT_CMD = None    # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe" or None if already on PATH
# ------------------------------------


# Known WV3 bands and canonical names in the public brochure/table
CANONICAL_BANDS = {
    "Panchromatic": ["panchromatic", "panchro", "pan"],
    "Coastal Blue": ["coastal blue", "coastal", "cb"],
    "Blue": ["blue"],
    "Green": ["green"],
    "Yellow": ["yellow"],
    "Red": ["red"],
    "Red Edge": ["red edge", "rededge"],
    "NIR1": ["nir1", "nir 1", "nir i"],
    "NIR2": ["nir2", "nir 2", "nir ii"],
    # SWIR bands present in some tables; supported if OCR finds them
    "SWIR1": ["swir1", "swir 1"],
    "SWIR2": ["swir2", "swir 2"],
    "SWIR3": ["swir3", "swir 3"],
    "SWIR4": ["swir4", "swir 4"],
    "SWIR5": ["swir5", "swir 5"],
    "SWIR6": ["swir6", "swir 6"],
    "SWIR7": ["swir7", "swir 7"],
    "SWIR8": ["swir8", "swir 8"],
}

# Colors for plotting/legend
BAND_COLORS = {
    "Panchromatic": "black",
    "Coastal Blue": "deepskyblue",
    "Blue": "blue",
    "Green": "green",
    "Yellow": "gold",
    "Red": "red",
    "Red Edge": "crimson",
    "NIR1": "purple",
    "NIR2": "magenta",
    "SWIR1": None,
    "SWIR2": None,
    "SWIR3": None,
    "SWIR4": None,
    "SWIR5": None,
    "SWIR6": None,
    "SWIR7": None,
    "SWIR8": None,
}

# Fallback WV3 values from the official brochure table (nm)
FALLBACK_TABLE = pd.DataFrame([
    ["Panchromatic", 627, 445, 808],
    ["Coastal Blue", 426, 397, 454],
    ["Blue",         481, 445, 517],
    ["Green",        547, 507, 586],
    ["Yellow",       605, 580, 629],
    ["Red",          661, 626, 696],
    ["Red Edge",     724, 698, 749],
    ["NIR1",         832, 765, 899],
    ["NIR2",         948, 857, 1039],
    ["SWIR1",       1210, 1184, 1235],
    ["SWIR2",       1572, 1546, 1598],
    ["SWIR3",       1661, 1636, 1686],
    ["SWIR4",       1730, 1702, 1759],
    ["SWIR5",       2164, 2137, 2191],
    ["SWIR6",       2203, 2174, 2232],
    ["SWIR7",       2260, 2228, 2292],
    ["SWIR8",       2329, 2285, 2373],
], columns=["Band", "Center", "Lower", "Upper"])


def try_ocr_table(image_path):
    """
    Attempt to OCR the WV3 table image and return a DataFrame with columns:
    Band, Center, Lower, Upper
    Returns None if OCR or parsing fails.
    """
    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageFilter
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        img = Image.open(image_path).convert("L")
        # Boost contrast and sharpness a little to help OCR on screenshots
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)

        # Use tesseract to get text by lines
        raw_text = pytesseract.image_to_string(img)
        lines = [re.sub(r"[^\x20-\x7E]", "", ln).strip() for ln in raw_text.splitlines()]
        lines = [ln for ln in lines if ln]

        # Collect rows by matching a band name and three numbers
        data = []
        for ln in lines:
            low = ln.lower()
            band_match = None
            for canon, aliases in CANONICAL_BANDS.items():
                for alias in aliases:
                    if alias in low:
                        band_match = canon
                        break
                if band_match:
                    break
            if not band_match:
                continue

            # Extract numbers (nm); expect 3 numbers on the row
            nums = re.findall(r"\d{2,4}", ln)
            nums = list(map(int, nums))
            # Heuristic: prefer (Center, Lower, Upper) order, else try to infer
            if len(nums) >= 3:
                # Choose the combination most consistent with Lower < Center < Upper
                best_triplet = None
                best_score = float("inf")
                for i in range(len(nums) - 2):
                    trip = nums[i:i+3]
                    # score by monotonicity and spread; lower is better
                    spread = abs(trip[2] - trip[0])
                    monotonic_penalty = 0 if (trip[0] <= trip[1] <= trip[2]) else 1e6
                    score = monotonic_penalty + (10000 - spread)  # prefer larger spread
                    if score < best_score:
                        best_triplet = trip
                        best_score = score
                if best_triplet is None:
                    continue
                L, C, U = best_triplet[0], best_triplet[1], best_triplet[2]
                # If looks like Center not in the middle, reorder by nearest-to-mid
                if not (L <= C <= U):
                    sorted_trip = sorted(best_triplet)
                    L, C, U = sorted_trip[0], sorted_trip[1], sorted_trip[2]
                # Some tables list Center first; correct if needed
                # Ensure C is the middle of L and U by nearest check
                mid = (L + U) / 2.0
                if abs(C - mid) > abs(best_triplet[1] - mid):
                    C = best_triplet[1]
                data.append([band_match, int(C), int(L), int(U)])

        if not data:
            return None

        # Deduplicate by band (prefer the first occurrence)
        rows = {}
        for band, C, L, U in data:
            if band not in rows:
                rows[band] = (C, L, U)
        df = pd.DataFrame([(b, *v) for b, v in rows.items()], columns=["Band", "Center", "Lower", "Upper"])
        return df

    except Exception:
        return None


def merge_with_fallback(ocr_df, fallback_df):
    """
    Merge OCR results (if any) with fallback values; OCR takes precedence for bands it captured.
    Returns a clean DataFrame with numeric columns.
    """
    fb = fallback_df.copy()
    fb["Band"] = fb["Band"].astype(str)

    if ocr_df is None or ocr_df.empty:
        return fb

    ocr = ocr_df.copy()
    ocr["Band"] = ocr["Band"].astype(str)

    # Normalize band names to canonical keys
    def canonicalize(name):
        low = name.lower().strip()
        for canon, aliases in CANONICAL_BANDS.items():
            if canon.lower() == low:
                return canon
            for a in aliases:
                if a == low:
                    return canon
        # fuzzy simple contains
        for canon, aliases in CANONICAL_BANDS.items():
            if canon.lower() in low:
                return canon
        return name

    ocr["Band"] = ocr["Band"].map(canonicalize)

    # Merge: start from fallback, replace rows where OCR has the band
    merged = fb.set_index("Band")
    for _, row in ocr.iterrows():
        b = row["Band"]
        if b in merged.index:
            merged.loc[b, ["Center", "Lower", "Upper"]] = [row["Center"], row["Lower"], row["Upper"]]
        else:
            merged.loc[b, ["Center", "Lower", "Upper"]] = [row["Center"], row["Lower"], row["Upper"]]
    merged = merged.reset_index()

    # If RGB_ONLY requested, filter
    if RGB_ONLY:
        merged = merged[merged["Band"].isin(["Blue", "Green", "Red"])].copy()

    # Ensure numeric and valid ranges
    merged[["Center", "Lower", "Upper"]] = merged[["Center", "Lower", "Upper"]].apply(pd.to_numeric, errors="coerce")
    merged = merged.dropna(subset=["Center", "Lower", "Upper"])
    merged = merged[merged["Lower"] < merged["Upper"]]
    return merged


def make_curve(lower, center, upper, num=1000, shape="hann"):
    """
    Generate a smooth passband curve in [lower, upper] peaking at center with max=1.0
    shape:
      - "triangle": linear rise/fall (exactly 0 at edges, 1 at center)
      - "hann": raised-cosine (0 at edges, smooth taper)
      - "gaussian": gaussian centered at C, sigma chosen so values ~0 at edges
    """
    lower, center, upper = float(lower), float(center), float(upper)
    wl = np.linspace(lower, upper, int(max(num, 100)))

    if shape == "triangle":
        # piecewise linear to center, then down
        left = (wl <= center)
        right = ~left
        y = np.zeros_like(wl)
        if center > lower:
            y[left] = (wl[left] - lower) / (center - lower)
        if upper > center:
            y[right] = (upper - wl[right]) / (upper - center)
        y = np.clip(y, 0.0, 1.0)
        return wl, y

    if shape == "hann":
        # map wl to [0, 1] across [lower, upper], then Hann window
        t = (wl - lower) / (upper - lower)
        y = 0.5 * (1 - np.cos(2 * np.pi * t))
        # Normalize to peak 1 at center (numerical precision)
        y /= y.max() if y.max() > 0 else 1.0
        return wl, y

    if shape == "gaussian":
        # choose sigma so edges are ~1% of peak
        # For a gaussian, exp(-0.5*((d/sigma)^2)) = 0.01 at d = (upper-lower)/2
        half_span = (upper - lower) / 2.0
        target = 0.01
        sigma = half_span / math.sqrt(2 * math.log(1/target))
        y = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
        return wl, y

    # default: hann
    return make_curve(lower, center, upper, num=num, shape="hann")


def save_spd(path, wl, rr):
    with open(path, "w") as f:
        for x, y in zip(wl, rr):
            f.write(f"{x:.6f} {y:.6f}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ocr_df = None
    if TABLE_IMAGE is not None:
        ocr_df = try_ocr_table(TABLE_IMAGE)

    table = merge_with_fallback(ocr_df, FALLBACK_TABLE)

    # Build SPDs
    spd_files = []
    for _, row in table.iterrows():
        band = str(row["Band"])
        C = float(row["Center"])
        L = float(row["Lower"])
        U = float(row["Upper"])

        wl, rr = make_curve(L, C, U, num=POINTS_PER_BAND, shape=CURVE_SHAPE)
        out_name = f"WV3_{band.replace(' ', '')}.spd"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        save_spd(out_path, wl, rr)
        spd_files.append((band, out_path, wl, rr))

    # Plot summary
    plt.figure(figsize=(10, 6))
    for band, _, wl, rr in spd_files:
        color = BAND_COLORS.get(band, None)
        if band in ("Blue", "Green", "Red"):
            # enforce true RGB where available
            if band == "Blue":
                color = "blue"
            elif band == "Green":
                color = "green"
            elif band == "Red":
                color = "red"
        plt.plot(wl, rr, label=band, linewidth=2, color=color)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Response")
    plt.title(f"WV3 Bandpass Curves ({CURVE_SHAPE})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, f"WV3_bandpass_{CURVE_SHAPE}.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Print outputs
    print("Generated SPD files:")
    for band, path, _, _ in spd_files:
        print(f"  {band:12s} -> {os.path.abspath(path)}")
    print(f"Summary plot -> {os.path.abspath(summary_path)}")

    # Also write a CSV summary of the extracted/used table
    table_csv = os.path.join(OUTPUT_DIR, "WV3_band_table_used.csv")
    table.sort_values("Lower", inplace=True)
    table.to_csv(table_csv, index=False)
    print(f"Table CSV -> {os.path.abspath(table_csv)}")


if __name__ == "__main__":
    main()
