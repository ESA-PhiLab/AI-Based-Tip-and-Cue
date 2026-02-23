import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ProductPaths:
    product_dir: Path
    bands_dir: Path
    geoloc_dir: Path
    bands_tiff: Path
    geoloc_json: Path
    metadata_json: Path | None


def extract_acquisition_times_from_product_path(any_path_in_product: str) -> tuple[datetime, datetime, datetime]:
    """Parse start/end UTC timestamps from PHISAT-2 product folder name and return (start, end, midpoint)."""
    parts = os.path.normpath(any_path_in_product).split(os.sep)
    product = next((p for p in parts if p.startswith("PHISAT-2_") and ("_L1_" in p or "_L2_" in p)), "")

    m = re.search(r"_([0-9]{14})_([0-9]{14})_", product)
    if not m:
        raise ValueError(f"Could not find timestamps in product name: {product!r}")

    t0 = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    tm = t0 + (t1 - t0) / 2
    return t0, t1, tm


def find_product_files(base_dir: Path, product_name: str) -> ProductPaths:
    """Find bands TIFF, geolocation JSON, and (optionally) a metadata JSON inside a PHISAT-2 product folder."""
    product_dir = (base_dir / "dataset" / product_name).resolve()
    if not product_dir.exists():
        raise FileNotFoundError(f"Product folder not found: {product_dir}")

    bands_dir = product_dir / "bands"
    geoloc_dir = product_dir / "geolocation"
    if not bands_dir.exists():
        raise FileNotFoundError(f"'bands' folder not found: {bands_dir}")
    if not geoloc_dir.exists():
        raise FileNotFoundError(f"'geolocation' folder not found: {geoloc_dir}")

    tiffs = sorted(bands_dir.glob("session_*_BC.tif*"))
    if len(tiffs) == 0:
        raise FileNotFoundError(f"No session_*_BC.tif(f) found in: {bands_dir}")
    if len(tiffs) > 1:
        names = "\n  - " + "\n  - ".join([p.name for p in tiffs])
        raise RuntimeError(f"Found multiple candidate TIFFs in {bands_dir}:{names}\nKeep exactly one or refine the glob.")

    bands_tiff = tiffs[0]

    gl_scene = geoloc_dir / "GL_scene_0.json"
    if not gl_scene.exists():
        raise FileNotFoundError(f"Geolocation file not found: {gl_scene}")

    meta_candidates = sorted(product_dir.glob("*metadata*.json")) + sorted(product_dir.glob("**/*metadata*.json"))
    meta_candidates = [p for p in meta_candidates if p.name.lower().endswith(".json")]

    metadata_json = meta_candidates[0] if meta_candidates else None

    return ProductPaths(
        product_dir=product_dir,
        bands_dir=bands_dir,
        geoloc_dir=geoloc_dir,
        bands_tiff=bands_tiff,
        geoloc_json=gl_scene,
        metadata_json=metadata_json,
    )


def print_band_stats(tiff_path: Path) -> None:
    """Print per-band reflectance stats assuming values are reflectance*10000."""
    with rasterio.open(tiff_path) as ds:
        print(f"\nFile: {tiff_path}")
        print(f"Size: {ds.width} x {ds.height}")
        print(f"Bands: {ds.count}")
        print(f"Dtype: {ds.dtypes}")
        print(f"NoData (dataset): {ds.nodata}")
        print(f"NoData (per-band): {ds.nodatavals}")
        print(f"Has internal mask: {ds.mask_flag_enums}")

        for b in range(1, ds.count + 1):
            arr = ds.read(b, masked=True)
            if arr.mask.all():
                print(f"Band {b}: no valid data (all masked)")
                continue

            data = arr.compressed().astype(np.float64)
            mn = data.min()
            mx = data.max()
            dm = data.mean()
            p1, p50, p99 = np.percentile(data, [1, 50, 99])

            print(
                f"Band {b}: min={mn/10000:.3f},\t max={mx/10000:.3f},\t mean={dm/10000:.3f},\t "
                f"p1={p1/10000:.3f},\t p50={p50/10000:.3f},\t p99={p99/10000:.3f},\t "
                f"valid_pixels={data.size}"
            )


def extract_corners_and_center(gl_scene_json_path: Path, size: int = 4096) -> tuple[dict, tuple[float, float]]:
    """Extract corner and center lat/lon from a GL_scene geolocation JSON; return extracted dict and (lat, lon) center."""
    with open(gl_scene_json_path, "r", encoding="utf-8") as f:
        gl = json.load(f)

    pts = gl["Geolocated_Points"]
    idx = {(p["X_coordinate"], p["Y_coordinate"]): p for p in pts}

    mid = size // 2

    points = {
        "top_left": (0, 0),
        "top_right": (size, 0),
        "bottom_right": (size, size),
        "bottom_left": (0, size),
        "center": (mid, mid),
    }

    print("\nExtracted points (lat, lon):")
    extracted: dict[str, dict] = {}

    for name, (x, y) in points.items():
        p = idx.get((x, y))
        if p is None:
            print(f"{name}: missing (X={x}, Y={y})")
            continue
        extracted[name] = p
        print(f"{name}: X={x}, Y={y}, lat={p['Lat']}, lon={p['Lon']}, alt={p.get('Alt')}")

    tl = extracted.get("top_left")
    tr = extracted.get("top_right")
    br = extracted.get("bottom_right")
    bl = extracted.get("bottom_left")
    ctr = extracted.get("center")

    if tl and tr and br and bl:
        print("\nWKT footprint (lon lat):")
        print(
            "POLYGON(("
            f"{tl['Lon']} {tl['Lat']}, "
            f"{tr['Lon']} {tr['Lat']}, "
            f"{br['Lon']} {br['Lat']}, "
            f"{bl['Lon']} {bl['Lat']}, "
            f"{tl['Lon']} {tl['Lat']}"
            "))"
        )

    if ctr is not None:
        lat_center, lon_center = float(ctr["Lat"]), float(ctr["Lon"])
        label = "center (from GL grid)"
    elif tl and tr and br and bl:
        lon_center = (float(tl["Lon"]) + float(tr["Lon"]) + float(br["Lon"]) + float(bl["Lon"])) / 4.0
        lat_center = (float(tl["Lat"]) + float(tr["Lat"]) + float(br["Lat"]) + float(bl["Lat"])) / 4.0
        label = "computed_center (corner centroid)"
    else:
        lat_center, lon_center = float("nan"), float("nan")
        label = "center (unavailable)"

    print(f"\n{label}: lat={lat_center}, lon={lon_center}")
    return extracted, (lat_center, lon_center)


def load_band_center_wavelengths(metadata_json: Path) -> dict[str, float]:
    """Load band center wavelengths (nm) from a session metadata JSON if present; returns dict like {'Band 1': 490, ...}."""
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if not isinstance(meta, dict) or len(meta) == 0:
        return {}

    root_key = next(iter(meta.keys()))
    imager_cfg = meta.get(root_key, {}).get("ImagerConfig", {})
    bcw = imager_cfg.get("BandCentreWavelength", {})
    if not isinstance(bcw, dict):
        return {}

    out: dict[str, float] = {}
    for k, v in bcw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def choose_rgb_bands_from_wavelengths(center_wavelengths_nm: dict[str, float], tiff_band_count: int) -> tuple[int, int, int]:
    """Choose (R,G,B) 1-based band indices by nearest wavelength to 665/560/490 nm."""
    targets = {"B": 481.9, "G": 547.1, "R": 660.1}

    entries: list[tuple[int, float]] = []
    for name, wl in center_wavelengths_nm.items():
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        band_num = int(m.group(1))  # metadata uses Band 0..7
        entries.append((band_num, wl))

    if not entries:
        return (3, 2, 1)

    def nearest(target_nm: float) -> int:
        band_num, _ = min(entries, key=lambda t: abs(t[1] - target_nm))
        idx_1based = band_num + 1  # Band 0 -> TIFF band 1, Band 1 -> TIFF band 2, ...
        if not (1 <= idx_1based <= tiff_band_count):
            raise RuntimeError(
                f"RGB selection produced out-of-range TIFF index {idx_1based} from metadata band {band_num} "
                f"(TIFF has {tiff_band_count} bands)."
            )
        return idx_1based

    r = nearest(targets["R"])
    g = nearest(targets["G"])
    b = nearest(targets["B"])
    return (r, g, b)


def stretch01_2d(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Percentile stretch a 2D array with NaNs to [0,1]."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)

    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.clip(band, 0.0, 1.0)
        out[~np.isfinite(out)] = 0.0
        return out.astype(np.float32)

    out = (band - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def show_rgb_reflectance(tiff_path: Path, rgb_bands_1based: tuple[int, int, int]) -> None:
    """Display an RGB composite in reflectance from a multiband TIFF (reflectance = DN/10000)."""
    r_b, g_b, b_b = rgb_bands_1based

    with rasterio.open(tiff_path) as ds:
        if ds.count < max(rgb_bands_1based):
            raise ValueError(f"TIFF has {ds.count} bands, but you requested {rgb_bands_1based}.")

        r = ds.read(r_b, masked=True).astype("float32").filled(np.nan) / 10000.0
        g = ds.read(g_b, masked=True).astype("float32").filled(np.nan) / 10000.0
        b = ds.read(b_b, masked=True).astype("float32").filled(np.nan) / 10000.0

    rgb = np.dstack([stretch01_2d(r), stretch01_2d(g), stretch01_2d(b)])

    plt.figure(figsize=(7, 7))
    plt.imshow(rgb)
    plt.title(f"PHI-SAT-2 RGB reflectance (R,G,B bands = {r_b},{g_b},{b_b})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    base_dir = Path.cwd()

    product_name = "offnadir_ocean2/" + "PHISAT-2_L1_000002103_20250423144634_20250423144637_B69C2A81"
    paths = find_product_files(base_dir, product_name)

    print("Resolved paths:")
    print(f"  product_dir : {paths.product_dir}")
    print(f"  bands_tiff  : {paths.bands_tiff.name}")
    print(f"  geoloc_json : {paths.geoloc_json.name}")
    print(f"  metadata_json: {paths.metadata_json.name if paths.metadata_json else '(none found)'}")

    t0, t1, tm = extract_acquisition_times_from_product_path(str(paths.product_dir))
    print("\nAcquisition time (UTC):")
    print(f"  start   : {t0.isoformat()}")
    print(f"  end     : {t1.isoformat()}")
    print(f"  midpoint: {tm.isoformat()}")

    print_band_stats(paths.bands_tiff)
    extract_corners_and_center(paths.geoloc_json, size=4096)

    rgb_bands = (4,3,2)  # fallback
    if paths.metadata_json is not None:
        wl = load_band_center_wavelengths(paths.metadata_json)
        if wl:
            print("\nBand center wavelengths from metadata (nm):")
            for k in sorted(wl.keys(), key=lambda s: int(re.search(r'(\d+)', s).group(1)) if re.search(r'(\d+)', s) else 999):
                print(f"  {k}: {wl[k]}")
            with rasterio.open(paths.bands_tiff) as ds:
                rgb_bands = choose_rgb_bands_from_wavelengths(wl, ds.count)

            print(f"\nSelected RGB from wavelengths (nearest to 665/560/490 nm): R,G,B = {rgb_bands}")
        else:
            print("\nNo BandCentreWavelength found in metadata; using fallback RGB = (3,2,1).")
    else:
        print("\nNo metadata JSON found; using fallback RGB = (3,2,1).")

    show_rgb_reflectance(paths.bands_tiff, rgb_bands)
