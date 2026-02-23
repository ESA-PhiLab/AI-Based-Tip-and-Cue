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


@dataclass(frozen=True)
class ProductPaths:
    product_dir: Path
    bands_dir: Path
    geoloc_dir: Path
    bands_tiff: Path
    geoloc_json: Path
    metadata_json: Path | None


def extract_acquisition_times_from_product_path(any_path_in_product: str) -> tuple[datetime, datetime, datetime]:
    """extract_acquisition_times_from_product_path(any_path_in_product) -> tuple[datetime,datetime,datetime]: Parse UTC start/end from product folder; return (start,end,mid)."""
    parts = os.path.normpath(any_path_in_product).split(os.sep)
    product = next((p for p in parts if p.startswith("PHISAT-2_") and ("_L1_" in p or "_L2_" in p)), "")
    m = re.search(r"_([0-9]{14})_([0-9]{14})_", product)
    if not m:
        raise ValueError(f"Could not find timestamps in product name: {product!r}")
    t0 = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    tm = t0 + (t1 - t0) / 2
    return t0, t1, tm

def save_rgb_png(tiff_path: Path, rgb_bands_1based: tuple[int, int, int], out_path: Path) -> Path:
    """Save RGB reflectance composite as 8-bit PNG and return output path."""
    rgb8 = make_rgb_uint8_from_tiff(tiff_path, rgb_bands_1based)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path.as_posix(), rgb8)
    return out_path

def find_product_files(base_dir: Path, product_name: str, prefer_level: str = "auto") -> ProductPaths:
    """find_product_files(base_dir,product_name,prefer_level='auto') -> ProductPaths: Select best multiband TIFF by product level (BC/AC/RR/auto) + GL_scene + metadata."""
    product_dir = (base_dir / "dataset" / product_name).resolve()
    if not product_dir.exists():
        raise FileNotFoundError(f"Product folder not found: {product_dir}")

    bands_dir = product_dir / "bands"
    geoloc_dir = product_dir / "geolocation"
    if not bands_dir.exists():
        raise FileNotFoundError(f"'bands' folder not found: {bands_dir}")
    if not geoloc_dir.exists():
        raise FileNotFoundError(f"'geolocation' folder not found: {geoloc_dir}")

    # Collect TIFFs
    tiffs = list(bands_dir.glob("*.tif")) + list(bands_dir.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"No TIFFs found in: {bands_dir}")

    # Deduplicate robustly
    uniq: dict[Path, Path] = {}
    for p in tiffs:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        uniq[rp] = rp
    tiffs = sorted(uniq.values(), key=lambda p: p.name.lower())

    def has_level_token(p: Path, token: str) -> bool:
        s = p.stem.lower()
        tok = token.lower()
        # Accept common naming patterns: *_bc_*, *_bc, bc_*, etc.
        return (f"_{tok}_" in s) or s.endswith(f"_{tok}") or s.startswith(f"{tok}_") or (f"_{tok}" in s) or (f"{tok}_" in s)

    prefer = (prefer_level or "auto").strip().upper()
    if prefer not in {"AUTO", "BC", "AC", "RR"}:
        raise ValueError(f"prefer_level must be one of: auto, BC, AC, RR (got {prefer_level!r})")

    # Priority order by intent:
    # - For radiance: prefer BC (coregistered radiance), then AC (radiance)
    # - For reflectance: prefer RR
    if prefer == "AUTO":
        priority = ["BC", "AC", "RR"]
    elif prefer == "BC":
        priority = ["BC", "AC", "RR"]
    elif prefer == "AC":
        priority = ["AC", "BC", "RR"]
    else:  # prefer == "RR"
        priority = ["RR", "BC", "AC"]

    # Filter candidates by the first available token in priority
    chosen_level = None
    candidates: list[Path] = []
    for lvl in priority:
        lvl_matches = [p for p in tiffs if has_level_token(p, lvl)]
        if lvl_matches:
            chosen_level = lvl
            candidates = lvl_matches
            break

    if not candidates:
        names = "\n  - " + "\n  - ".join([p.name for p in tiffs])
        raise FileNotFoundError(f"No BC/AC/RR TIFF found in {bands_dir}. Found:{names}")

    def band_count(p: Path) -> int:
        try:
            with rasterio.open(p) as ds:
                return int(ds.count)
        except Exception:
            return -1

    def file_size(p: Path) -> int:
        try:
            return int(p.stat().st_size)
        except Exception:
            return -1

    def score(p: Path) -> tuple[int, int, int, str]:
        # Higher is better for first three fields
        s = p.stem.lower()
        is_multi = 1 if "multiband" in s else 0
        nb = band_count(p)
        sz = file_size(p)
        return (is_multi, nb, sz, p.name.lower())

    ranked = sorted(candidates, key=score, reverse=True)
    best = ranked[0]

    best_score = score(best)
    ties = [p for p in ranked if score(p)[:3] == best_score[:3]]
    ties = list({p.resolve() if p.exists() else p: p for p in ties}.values())
    if len(ties) > 1:
        names = "\n  - " + "\n  - ".join([f"{p.name} (bands={band_count(p)}, size={file_size(p)})" for p in ties])
        print(f"WARNING: Multiple {chosen_level} TIFFs look equivalent; using: {best.name}\nCandidates:{names}")
    else:
        print(f"Selected level: {chosen_level}  TIFF: {best.name}")

    gl_scene = geoloc_dir / "GL_scene_0.json"
    if not gl_scene.exists():
        raise FileNotFoundError(f"Geolocation file not found: {gl_scene}")

    meta_candidates = sorted(product_dir.glob("*metadata*.json")) + sorted(product_dir.glob("**/*metadata*.json"))
    meta_candidates = [p for p in meta_candidates if p.is_file() and p.suffix.lower() == ".json"]
    metadata_json = meta_candidates[0] if meta_candidates else None

    return ProductPaths(product_dir, bands_dir, geoloc_dir, best, gl_scene, metadata_json)


def extract_corners_and_center(gl_scene_json_path: Path, size: int = 4096) -> tuple[dict, tuple[float, float]]:
    """extract_corners_and_center(gl_scene_json_path,size=4096) -> tuple[dict,tuple[float,float]]: Extract corner+center lat/lon from GL_scene."""
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

    extracted: dict[str, dict] = {}
    for name, (x, y) in points.items():
        p = idx.get((x, y))
        if p is not None:
            extracted[name] = p

    ctr = extracted.get("center")
    if ctr is not None:
        return extracted, (float(ctr["Lat"]), float(ctr["Lon"]))
    return extracted, (float("nan"), float("nan"))


def load_band_center_wavelengths(metadata_json: Path) -> dict[str, float]:
    """load_band_center_wavelengths(metadata_json) -> dict[str,float]: Read BandCentreWavelength (nm) from session metadata JSON."""
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
    """choose_rgb_bands_from_wavelengths(center_wavelengths_nm,tiff_band_count) -> tuple[int,int,int]: Choose 1-based (R,G,B) nearest to 665/560/490 nm."""
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

    return (nearest(targets["R"]), nearest(targets["G"]), nearest(targets["B"]))


def read_band_phys(ds: rasterio.io.DatasetReader, b: int, window: rasterio.windows.Window | None = None) -> np.ndarray:
    """read_band_phys(ds,b,window=None) -> np.ndarray: Read band and apply per-band scale/offset if present; returns float32 with NaNs."""
    arr = ds.read(b, window=window, masked=True).astype(np.float32).filled(np.nan)

    scale = None
    offset = None

    try:
        if ds.scales and len(ds.scales) >= b and ds.scales[b - 1] not in (None, 0):
            scale = float(ds.scales[b - 1])
        if ds.offsets and len(ds.offsets) >= b and ds.offsets[b - 1] is not None:
            offset = float(ds.offsets[b - 1])
    except Exception:
        pass

    if scale is None or offset is None:
        tags = ds.tags(b)
        if scale is None:
            for k in ("scale", "Scale", "scale_factor", "SCALING_FACTOR", "GAIN"):
                if k in tags:
                    try:
                        scale = float(tags[k])
                        break
                    except Exception:
                        pass
        if offset is None:
            for k in ("offset", "Offset", "add_offset", "ADD_OFFSET", "BIAS"):
                if k in tags:
                    try:
                        offset = float(tags[k])
                        break
                    except Exception:
                        pass

    if scale is None:
        scale = 1.0
    if offset is None:
        offset = 0.0

    out = arr * scale + offset
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def print_radiance_band_stats(tiff_path: Path) -> None:
    """print_radiance_band_stats(tiff_path) -> None: Print per-band stats in physical units using scale/offset."""
    with rasterio.open(tiff_path) as ds:
        print(f"\nFile: {tiff_path}")
        print(f"Size: {ds.width} x {ds.height}")
        print(f"Bands: {ds.count}")
        print(f"Dtype: {ds.dtypes}")
        print(f"Units: {ds.units}")
        print(f"Scales: {ds.scales}")
        print(f"Offsets: {ds.offsets}")
        print(f"NoData (dataset): {ds.nodata}")
        print(f"NoData (per-band): {ds.nodatavals}")

        for b in range(1, ds.count + 1):
            x = read_band_phys(ds, b, window=None)
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                print(f"Band {b}: no valid data")
                continue
            mn, mx, mean = float(np.min(finite)), float(np.max(finite)), float(np.mean(finite))
            p1, p50, p99 = np.percentile(finite, [1, 50, 99])
            print(f"Band {b}: min={mn:.6g}  max={mx:.6g}  mean={mean:.6g}  p1={p1:.6g}  p50={p50:.6g}  p99={p99:.6g}  n={finite.size}")


def stretch01_2d(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """stretch01_2d(band,p_low=2,p_high=98) -> np.ndarray: Percentile stretch 2D array with NaNs to [0,1]."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = band.copy()
        out[~np.isfinite(out)] = 0.0
        out = out - np.nanmin(out)
        denom = np.nanmax(out)
        return (out / denom).astype(np.float32) if denom > 0 else np.zeros_like(out, dtype=np.float32)
    out = (band - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def make_rgb_uint8_from_radiance_tiff(tiff_path: Path, rgb_bands_1based: tuple[int, int, int]) -> np.ndarray:
    """make_rgb_uint8_from_radiance_tiff(tiff_path,rgb_bands_1based) -> np.ndarray: Read radiance RGB (scaled), stretch, return uint8 RGB."""
    r_b, g_b, b_b = rgb_bands_1based
    with rasterio.open(tiff_path) as ds:
        r = read_band_phys(ds, r_b)
        g = read_band_phys(ds, g_b)
        b = read_band_phys(ds, b_b)
    rgb01 = np.dstack([stretch01_2d(r), stretch01_2d(g), stretch01_2d(b)])
    return np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)


def show_rgb_radiance(tiff_path: Path, rgb_bands_1based: tuple[int, int, int]) -> None:
    """show_rgb_radiance(tiff_path,rgb_bands_1based) -> None: Display RGB composite made from radiance bands."""
    rgb8 = make_rgb_uint8_from_radiance_tiff(tiff_path, rgb_bands_1based)
    r_b, g_b, b_b = rgb_bands_1based
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb8)
    plt.title(f"PHI-SAT-2 RGB radiance (R,G,B bands = {r_b},{g_b},{b_b})")
    plt.axis("off")
    plt.show()


def save_rgb_png_radiance(tiff_path: Path, rgb_bands_1based: tuple[int, int, int], out_path: Path) -> Path:
    """save_rgb_png_radiance(tiff_path,rgb_bands_1based,out_path) -> Path: Save radiance RGB composite as 8-bit PNG."""
    rgb8 = make_rgb_uint8_from_radiance_tiff(tiff_path, rgb_bands_1based)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path.as_posix(), rgb8)
    return out_path


def datetime_to_jd(dt_utc: datetime) -> float:
    """datetime_to_jd(dt_utc) -> float: Convert timezone-aware UTC datetime to Julian Date."""
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


def fetch_nearest_tle_from_satchecker(norad_id: int, jd_epoch: float, data_source: str | None = None) -> tuple[str, str, str]:
    """fetch_nearest_tle_from_satchecker(norad_id,jd_epoch,data_source=None) -> tuple[str,str,str]: Fetch nearest TLE lines and epoch string from SatChecker."""
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
    return item["tle_line1"], item["tle_line2"], item.get("epoch", "unknown")


@dataclass(frozen=True)
class SatLLA:
    lat_deg: float
    lon_deg: float
    alt_km: float


def _gmst_rad(dt_utc: datetime) -> float:
    """_gmst_rad(dt_utc) -> float: Approx GMST angle in radians."""
    jd = datetime_to_jd(dt_utc)
    t = (jd - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t * t * t
    return (gmst_sec % 86400.0) * (2.0 * math.pi / 86400.0)


def _ecef_from_teme_km(r_teme_km: tuple[float, float, float], dt_utc: datetime) -> tuple[float, float, float]:
    """_ecef_from_teme_km(r_teme_km,dt_utc) -> tuple[float,float,float]: Approx TEME->ECEF rotation via GMST."""
    x, y, z = r_teme_km
    th = _gmst_rad(dt_utc)
    c, s = math.cos(th), math.sin(th)
    return (c * x + s * y, -s * x + c * y, z)


def _lla_from_ecef_km(x_km: float, y_km: float, z_km: float) -> SatLLA:
    """_lla_from_ecef_km(x_km,y_km,z_km) -> SatLLA: ECEF (km) -> geodetic lat/lon/alt (km), WGS84."""
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
    """sat_lla_from_tle_at_time(tle1,tle2,dt_utc) -> SatLLA: Compute satellite lat/lon/alt from TLE at dt_utc."""
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


if __name__ == "__main__":
    base_dir = Path.cwd()

    # Point this to your product (relative to base_dir/dataset/)
    product_name = "phisat-2_data/dataset/PHISAT-2_L1_000004559_20260202210025_20260202210029_C9E695C7"

    # Prefer BC (coregistered radiance); fall back to AC (radiance) if BC not found
    paths = find_product_files(base_dir, product_name, prefer="BC")

    print("Resolved paths:")
    print(f"  product_dir  : {paths.product_dir}")
    print(f"  bands_tiff   : {paths.bands_tiff.name}")
    print(f"  geoloc_json  : {paths.geoloc_json.name}")
    print(f"  metadata_json: {paths.metadata_json.name if paths.metadata_json else '(none found)'}")

    t0, t1, tm = extract_acquisition_times_from_product_path(str(paths.product_dir))
    print("\nAcquisition time (UTC):")
    print(f"  start   : {t0.isoformat()}")
    print(f"  end     : {t1.isoformat()}")
    print(f"  midpoint: {tm.isoformat()}")

    with rasterio.open(paths.bands_tiff) as ds:
        size = int(max(ds.width, ds.height))

    extracted, (tgt_lat, tgt_lon) = extract_corners_and_center(paths.geoloc_json, size=size)
    print(f"\nScene center from GL: lat={tgt_lat:.6f}, lon={tgt_lon:.6f}")

    print_radiance_band_stats(paths.bands_tiff)

    rgb_bands = (3, 2, 1)
    if paths.metadata_json is not None:
        wl = load_band_center_wavelengths(paths.metadata_json)
        if wl:
            with rasterio.open(paths.bands_tiff) as ds:
                rgb_bands = choose_rgb_bands_from_wavelengths(wl, ds.count)

    show_rgb_radiance(paths.bands_tiff, rgb_bands)

    r_b, g_b, b_b = rgb_bands
    out_png = paths.product_dir / f"rgb_radiance_R{r_b}_G{g_b}_B{b_b}.png"
    saved = save_rgb_png_radiance(paths.bands_tiff, rgb_bands, out_png)
    print(f"\nSaved radiance RGB PNG: {saved}")