#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import rasterio


def _read_band1(path: Path) -> np.ndarray:
    """Read single-band TIFF -> float32 array with NaN for nodata."""
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype("float32").filled(np.nan)
    return arr


def _read_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read 3-band RGB TIFF -> (R,G,B) float32 arrays."""
    with rasterio.open(path) as ds:
        if ds.count < 3:
            raise ValueError(f"RGB file must have >=3 bands, got {ds.count}")
        r = ds.read(1, masked=True).astype("float32").filled(np.nan)
        g = ds.read(2, masked=True).astype("float32").filled(np.nan)
        b = ds.read(3, masked=True).astype("float32").filled(np.nan)
    return r, g, b


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over finite pixels."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return float("nan")
    x = a[m].astype("float64")
    y = b[m].astype("float64")
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt((x * x).mean()) * np.sqrt((y * y).mean())
    if denom == 0:
        return float("nan")
    return float((x * y).mean() / denom)


def _band_index_from_name(p: Path) -> int | None:
    """Extract band index from filename like *_12_3.tif."""
    m = re.search(r"_12_(\d+)\.(tif|tiff)$", p.name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_rgb_file(folder: Path) -> Path:
    """Find *_RGB.tif or any >=3 band tif."""
    rgb = sorted(folder.glob("*_RGB.tif*"))
    if len(rgb) == 1:
        return rgb[0]

    for p in sorted(folder.glob("*.tif*")):
        with rasterio.open(p) as ds:
            if ds.count >= 3:
                return p

    rgb = sorted(folder.glob("*_RGB.tiff*"))
    if len(rgb) == 1:
        return rgb[0]

    for p in sorted(folder.glob("*.tiff*")):
        with rasterio.open(p) as ds:
            if ds.count >= 3:
                return p

    raise FileNotFoundError("No RGB TIFF found.")


def main(folder: Path) -> None:
    rgb_path = find_rgb_file(folder)
    r_rgb, g_rgb, b_rgb = _read_rgb(rgb_path)

    candidates = [
        p for p in sorted(folder.glob("*.tif*"))
        if p.resolve() != rgb_path.resolve()
    ]

    results = []

    for p in candidates:
        with rasterio.open(p) as ds:
            if ds.count != 1:
                continue

        band = _read_band1(p)
        cr = _corr(band, r_rgb)
        cg = _corr(band, g_rgb)
        cb = _corr(band, b_rgb)

        results.append((p, cr, cg, cb, _band_index_from_name(p)))

    if not results:
        raise RuntimeError("No single-band TIFFs found.")

    def best(channel_index: int):
        valid = [(p, vals[channel_index], bi)
                 for (p, *vals, bi) in results
                 if np.isfinite(vals[channel_index])]
        if not valid:
            return None
        return max(valid, key=lambda t: abs(t[1]))

    best_r = best(0)
    best_g = best(1)
    best_b = best(2)

    print(f"\nRGB file: {rgb_path.name}\n")
    print("Best matches (by absolute correlation):\n")

    if best_r:
        print(f"R channel <- {best_r[0].name}  corr={best_r[1]:.4f}  band={best_r[2]}")
    if best_g:
        print(f"G channel <- {best_g[0].name}  corr={best_g[1]:.4f}  band={best_g[2]}")
    if best_b:
        print(f"B channel <- {best_b[0].name}  corr={best_b[1]:.4f}  band={best_b[2]}")

    print("\nAll correlations:\n")
    print("file".ljust(45), "corr(R)".rjust(10), "corr(G)".rjust(10), "corr(B)".rjust(10))
    for p, cr, cg, cb, bi in results:
        print(p.name.ljust(45), f"{cr:10.4f}", f"{cg:10.4f}", f"{cb:10.4f}")


if __name__ == "__main__":
    folder = Path(
        r"C:\Users\nadine\OneDrive - Delft University of Technology\Documenten\Phi-Lab_MasterThesis\3_Software\AI-TC-Ultimate\AI-Based-Tip-and-Cue\dataset\phisat-2_data\dataset\offnadir_ocean2\PHISAT-2_L1_000001987_20250410143947_20250410143950_B05E6C3E\bands"
    )
    main(folder)