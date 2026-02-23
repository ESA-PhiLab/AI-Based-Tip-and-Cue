import re
from pathlib import Path

import numpy as np
import rasterio


def _is_stage(p: Path, token: str) -> bool:
    s = p.stem.lower()
    t = token.lower()
    return (f"_{t}_" in s) or s.endswith(f"_{t}") or s.startswith(f"{t}_") or (f"_{t}" in s) or (f"{t}_" in s)


def _pick_best_tiff(bands_dir: Path, prefer: str) -> Path:
    tiffs = list(bands_dir.glob("*.tif")) + list(bands_dir.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"No TIFFs found in: {bands_dir}")

    prefer = prefer.strip().upper()
    if prefer not in {"RR", "BC", "AC"}:
        raise ValueError("prefer must be one of: RR, BC, AC")

    stages = [prefer] + [s for s in ["RR", "BC", "AC"] if s != prefer]
    for st in stages:
        cand = [p for p in tiffs if _is_stage(p, st)]
        if cand:
            cand = sorted(cand, key=lambda p: p.name.lower())
            return cand[0]

    return sorted(tiffs, key=lambda p: p.name.lower())[0]


def _band_stats_raw(ds: rasterio.io.DatasetReader, b: int, win=None) -> dict:
    x = ds.read(b, window=win, masked=True)
    if hasattr(x, "mask") and np.all(x.mask):
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan}
    v = x.compressed().astype(np.float64) if hasattr(x, "compressed") else x.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan}
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def _interpretation_hint(raw_min: float, raw_max: float, stage: str) -> str:
    # Heuristics only (because your TIFF has no units/scales/offsets set).
    if stage == "RR":
        return "RR stage -> very likely TOA reflectance stored as integer (often refl*10000)."
    if raw_max > 255:
        dtype_hint = "values >255 -> cannot be uint8; likely uint16/int16."
    else:
        dtype_hint = "values <=255 -> could be uint8/uint16/int16 (check dtype below)."

    if raw_max <= 5:
        phys_hint = "range is tiny; could be reflectance (0..1) already stored as float, or radiance with strong scaling."
    elif raw_max <= 80:
        phys_hint = "range ~10..80 is plausible for spectral radiance in W/m^2/sr/um (quantized)."
    else:
        phys_hint = "range ~100..1000 often matches 'reflectance*10000' (0.01..0.10) OR radiance with a scale."

    return f"{dtype_hint} {phys_hint} If you have RR outputs, use RR for reflectance comparisons; otherwise BC/AC are radiance."


def inspect_product(product_dir: str, prefer_stage: str = "RR", crop: int = 512) -> None:
    """inspect_product(product_dir,prefer_stage='RR',crop=512) -> None: Print raw min/max/mean + dtype + quick unit hints for chosen TIFF."""
    product_dir = Path(product_dir).expanduser().resolve()
    bands_dir = product_dir / "bands"
    if not bands_dir.exists():
        raise FileNotFoundError(f"Missing bands dir: {bands_dir}")

    tiff = _pick_best_tiff(bands_dir, prefer=prefer_stage)
    stage = "RR" if _is_stage(tiff, "RR") else ("BC" if _is_stage(tiff, "BC") else ("AC" if _is_stage(tiff, "AC") else "UNKNOWN"))

    with rasterio.open(tiff) as ds:
        print("\n--- SELECTED TIFF ---")
        print(f"path  : {tiff}")
        print(f"stage : {stage}")
        print(f"size  : {ds.width} x {ds.height}  bands={ds.count}")
        print(f"dtype : {ds.dtypes}")
        print(f"nodata: {ds.nodata}  nodatavals={ds.nodatavals}")
        print(f"units : {ds.units}")
        print(f"scales: {ds.scales}")
        print(f"offs  : {ds.offsets}")
        print(f"tags1 : {ds.tags(1)}")

        # center crop window (same logic you used)
        crop = int(crop)
        x0 = max(0, (ds.width - crop) // 2)
        y0 = max(0, (ds.height - crop) // 2)
        win = rasterio.windows.Window(col_off=x0, row_off=y0, width=min(crop, ds.width), height=min(crop, ds.height))

        print(f"\n--- RAW INPUT VALUES (no scaling), center crop {win.width}x{win.height} at x0={x0}, y0={y0} ---")
        for b in range(1, min(ds.count, 8) + 1):
            s = _band_stats_raw(ds, b, win=win)
            print(f"Band {b:02d} RAW: n={s['n']}  min={s['min']:.3f}  max={s['max']:.3f}  mean={s['mean']:.3f}")

        # quick hint based on band 1 range
        s1 = _band_stats_raw(ds, 1, win=win)
        print("\n--- QUICK INTERPRETATION (heuristic) ---")
        print(_interpretation_hint(s1["min"], s1["max"], stage))

        # show what refl would look like if scaled by 10000 (common convention)
        print("\n--- IF IT WERE reflectance*10000 (just a diagnostic) ---")
        for b in range(1, min(ds.count, 3) + 1):
            s = _band_stats_raw(ds, b, win=win)
            print(f"Band {b:02d} refl~=RAW/10000 -> min={s['min']/10000.0:.6f}  max={s['max']/10000.0:.6f}  mean={s['mean']/10000.0:.6f}")


if __name__ == "__main__":
    # Example:
    # python inspect_phisat_inputs.py "C:/.../PHISAT-2_L1_..._XXXXXXX" RR 512
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python inspect_phisat_inputs.py <product_dir> [RR|BC|AC] [crop]")

    prod = sys.argv[1]
    pref = sys.argv[2] if len(sys.argv) >= 3 else "RR"
    crop = int(sys.argv[3]) if len(sys.argv) >= 4 else 512
    inspect_product(prod, prefer_stage=pref, crop=crop)