import json
import os
import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import requests


# ----------------------------
# Product paths / parsing
# ----------------------------

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
    """Find bands TIFF, geolocation JSON, and (optionally) metadata JSON inside a PHISAT-2 product folder."""
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


# ----------------------------
# Stats + geolocation
# ----------------------------

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

    ctr = extracted.get("center")
    if ctr is not None:
        lat_center, lon_center = float(ctr["Lat"]), float(ctr["Lon"])
        label = "center (from GL grid)"
    else:
        lat_center, lon_center = float("nan"), float("nan")
        label = "center (unavailable)"

    print(f"\n{label}: lat={lat_center}, lon={lon_center}")
    return extracted, (lat_center, lon_center)


# ----------------------------
# Band wavelengths -> RGB selection
# ----------------------------

def load_band_center_wavelengths(metadata_json: Path) -> dict[str, float]:
    """Load band center wavelengths (nm) from session metadata JSON."""
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
    targets = {"B": 490.0, "G": 560.0, "R": 665.0}

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
        idx_1based = band_num + 1
        if not (1 <= idx_1based <= tiff_band_count):
            raise RuntimeError(f"RGB selection out-of-range: {idx_1based} (TIFF bands={tiff_band_count})")
        return idx_1based

    r = nearest(targets["R"])
    g = nearest(targets["G"])
    b = nearest(targets["B"])
    return (r, g, b)


# ----------------------------
# RGB display + SAVE
# ----------------------------

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


def make_rgb_uint8_from_tiff(tiff_path: Path, rgb_bands_1based: tuple[int, int, int]) -> np.ndarray:
    """Read reflectance RGB (DN/10000), stretch, and return uint8 RGB image (H,W,3)."""
    r_b, g_b, b_b = rgb_bands_1based
    with rasterio.open(tiff_path) as ds:
        r = ds.read(r_b, masked=True).astype("float32").filled(np.nan) / 10000.0
        g = ds.read(g_b, masked=True).astype("float32").filled(np.nan) / 10000.0
        b = ds.read(b_b, masked=True).astype("float32").filled(np.nan) / 10000.0

    rgb01 = np.dstack([stretch01_2d(r), stretch01_2d(g), stretch01_2d(b)])
    rgb8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    return rgb8


def show_rgb_reflectance(tiff_path: Path, rgb_bands_1based: tuple[int, int, int]) -> None:
    """Display an RGB composite in reflectance from a multiband TIFF (reflectance = DN/10000)."""
    rgb8 = make_rgb_uint8_from_tiff(tiff_path, rgb_bands_1based)
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb8)
    r_b, g_b, b_b = rgb_bands_1based
    plt.title(f"PHI-SAT-2 RGB reflectance (R,G,B bands = {r_b},{g_b},{b_b})")
    plt.axis("off")
    plt.show()


def save_rgb_png(tiff_path: Path, rgb_bands_1based: tuple[int, int, int], out_path: Path) -> Path:
    """Save RGB reflectance composite as 8-bit PNG and return output path."""
    rgb8 = make_rgb_uint8_from_tiff(tiff_path, rgb_bands_1based)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path.as_posix(), rgb8)
    return out_path


# ----------------------------
# Time -> Julian Date
# ----------------------------

def datetime_to_jd(dt_utc: datetime) -> float:
    """Convert timezone-aware UTC datetime to Julian Date."""
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware")
    dt_utc = dt_utc.astimezone(timezone.utc)

    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day
    hh = dt_utc.hour
    mm = dt_utc.minute
    ss = dt_utc.second + dt_utc.microsecond / 1e6

    if m <= 2:
        y -= 1
        m += 12

    a = y // 100
    b = 2 - a + (a // 4)
    jd0 = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    frac = (hh + (mm + ss / 60.0) / 60.0) / 24.0
    return jd0 + frac


# ----------------------------
# Fetch TLE automatically (SatChecker)
# ----------------------------

def fetch_nearest_tle_from_satchecker(norad_id: int, jd_epoch: float, data_source: str | None = None) -> tuple[str, str, str]:
    """Fetch nearest TLE lines (and TLE epoch) for a NORAD ID near a Julian Date using SatChecker."""
    url = "https://satchecker.cps.iau.org/tools/get-nearest-tle/"
    params = {"id": str(norad_id), "id_type": "catalog", "epoch": str(jd_epoch)}
    if data_source:
        params["data_source"] = data_source

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if not isinstance(payload, list) or len(payload) == 0:
        raise RuntimeError(f"Unexpected SatChecker response: {payload!r}")

    tle_data = payload[0].get("tle_data", [])
    if not tle_data:
        raise RuntimeError(f"No TLE returned for NORAD {norad_id} at JD {jd_epoch}")

    item = tle_data[0]
    tle1 = item["tle_line1"]
    tle2 = item["tle_line2"]
    tle_epoch = item.get("epoch", "unknown")
    return tle1, tle2, tle_epoch


# ----------------------------
# SGP4 -> LLA (needs pip install sgp4)
# ----------------------------

@dataclass(frozen=True)
class SatLLA:
    lat_deg: float
    lon_deg: float
    alt_km: float


def _gmst_rad(dt_utc: datetime) -> float:
    """GMST angle in radians (approx)."""
    jd = datetime_to_jd(dt_utc)
    t = (jd - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t * t * t
    return (gmst_sec % 86400.0) * (2.0 * math.pi / 86400.0)


def _ecef_from_teme_km(r_teme_km: tuple[float, float, float], dt_utc: datetime) -> tuple[float, float, float]:
    """Approx TEME->ECEF via GMST rotation."""
    x, y, z = r_teme_km
    th = _gmst_rad(dt_utc)
    c, s = math.cos(th), math.sin(th)
    return (c * x + s * y, -s * x + c * y, z)


def _lla_from_ecef_km(x_km: float, y_km: float, z_km: float) -> SatLLA:
    """ECEF (km) -> geodetic lat/lon/alt (km), WGS84."""
    a = 6378.137
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)

    lon = math.atan2(y_km, x_km)
    p = math.hypot(x_km, y_km)

    lat = math.atan2(z_km, p * (1 - e2))
    for _ in range(8):
        s = math.sin(lat)
        n = a / math.sqrt(1 - e2 * s * s)
        alt = p / max(1e-12, math.cos(lat)) - n
        lat = math.atan2(z_km, p * (1 - e2 * (n / (n + alt))))

    s = math.sin(lat)
    n = a / math.sqrt(1 - e2 * s * s)
    alt = p / max(1e-12, math.cos(lat)) - n

    lon_deg = ((math.degrees(lon) + 540.0) % 360.0) - 180.0
    return SatLLA(lat_deg=math.degrees(lat), lon_deg=lon_deg, alt_km=alt)


def sat_lla_from_tle_at_time(tle1: str, tle2: str, dt_utc: datetime) -> SatLLA:
    """Compute satellite geodetic lat/lon/alt at dt_utc given a 2-line TLE."""
    from sgp4.api import Satrec

    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (UTC)")
    dt_utc = dt_utc.astimezone(timezone.utc)

    sat = Satrec.twoline2rv(tle1.strip(), tle2.strip())
    jd = datetime_to_jd(dt_utc)
    jd_int = float(int(jd))
    fr = jd - jd_int

    err, r, _v = sat.sgp4(jd_int, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code: {err}")

    x_ecef, y_ecef, z_ecef = _ecef_from_teme_km((r[0], r[1], r[2]), dt_utc)
    return _lla_from_ecef_km(x_ecef, y_ecef, z_ecef)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    base_dir = Path.cwd()

    product_name = "offnadir_ocean2/PHISAT-2_L1_000002103_20250423144634_20250423144637_B69C2A81"
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
    extracted, (tgt_lat, tgt_lon) = extract_corners_and_center(paths.geoloc_json, size=4096)

    rgb_bands = (3, 2, 1)
    if paths.metadata_json is not None:
        wl = load_band_center_wavelengths(paths.metadata_json)
        if wl:
            print("\nBand center wavelengths from metadata (nm):")
            def _band_sort_key(s: str) -> int:
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) if m else 999
            for k in sorted(wl.keys(), key=_band_sort_key):
                print(f"  {k}: {wl[k]}")
            with rasterio.open(paths.bands_tiff) as ds:
                rgb_bands = choose_rgb_bands_from_wavelengths(wl, ds.count)
            print(f"\nSelected RGB from wavelengths (nearest to 665/560/490 nm): R,G,B = {rgb_bands}")
        else:
            print("\nNo BandCentreWavelength found in metadata; using fallback RGB = (3,2,1).")
    else:
        print("\nNo metadata JSON found; using fallback RGB = (3,2,1).")

    show_rgb_reflectance(paths.bands_tiff, rgb_bands)

    # Save RGB PNG next to product folder
    r_b, g_b, b_b = rgb_bands
    out_png = paths.product_dir / f"rgb_reflectance_R{r_b}_G{g_b}_B{b_b}.png"
    saved = save_rgb_png(paths.bands_tiff, rgb_bands, out_png)
    print(f"\nSaved RGB PNG: {saved}")

    # ---- Automatically fetch nearest TLE and compute satellite LLA at midpoint ----
    norad_id = 60470  # PHISAT-2
    jd_mid = datetime_to_jd(tm)

    try:
        tle1, tle2, tle_epoch = fetch_nearest_tle_from_satchecker(norad_id, jd_mid, data_source=None)
        lla = sat_lla_from_tle_at_time(tle1, tle2, tm)

        print("\nPHISAT-2 nearest TLE (SatChecker):")
        print(f"  requested JD : {jd_mid:.6f}")
        print(f"  TLE epoch    : {tle_epoch}")
        print(f"  line1        : {tle1}")
        print(f"  line2        : {tle2}")

        print("\nPHISAT-2 position at midpoint (SGP4 from nearest TLE):")
        print(f"  sat_lat = {lla.lat_deg:.6f} deg")
        print(f"  sat_lon = {lla.lon_deg:.6f} deg")
        print(f"  sat_alt = {lla.alt_km:.3f} km")

        print("\nTarget position (scene center from GL grid):")
        print(f"  tgt_lat = {tgt_lat:.6f} deg")
        print(f"  tgt_lon = {tgt_lon:.6f} deg")
        print(f"  tgt_alt = {0.000:.3f} km")

        print(f"\nmidpoint time = {tm.isoformat()}")

    except Exception as e:
        print("\nCould not fetch/compute PHISAT-2 position automatically.")
        print(f"Reason: {e}")



