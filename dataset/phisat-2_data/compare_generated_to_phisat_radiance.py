import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# Your project imports (keep these as-is in your repo)
from settings import *
from offnadir_imaging.rendering import generate_image
from offnadir_imaging.functions.get_satellite_data import get_band_data


@dataclass(frozen=True)
class ProductPaths:
    product_dir: Path
    bands_dir: Path
    geoloc_dir: Path
    bands_tiff: Path
    geoloc_json: Path
    metadata_json: Path | None


def extract_acquisition_times_from_product_path(any_path_in_product: str) -> tuple[datetime, datetime, datetime]:
    """extract_acquisition_times_from_product_path(any_path_in_product)->(start_utc,end_utc,mid_utc): Parse UTC times from PHISAT-2 product folder name."""
    parts = os.path.normpath(any_path_in_product).split(os.sep)
    product = next((p for p in parts if p.startswith("PHISAT-2_") and ("_L1_" in p or "_L2_" in p)), "")
    m = re.search(r"_([0-9]{14})_([0-9]{14})_", product)
    if not m:
        raise ValueError(f"Could not find timestamps in product name: {product!r}")
    t0 = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    tm = t0 + (t1 - t0) / 2
    return t0, t1, tm


def find_product_files(base_dir: Path, product_name_or_dir: str, prefer_level: str = "BC") -> ProductPaths:
    """find_product_files(base_dir,product_name_or_dir,prefer_level='BC')->ProductPaths: Resolve product folder and pick best multiband TIFF + GL_scene_0 + metadata json."""
    p = Path(product_name_or_dir)

    # Accept either full product_dir OR product name relative to base_dir/dataset/
    if p.exists() and p.is_dir():
        product_dir = p.resolve()
    else:
        product_dir = (base_dir / "dataset" / "phisat-2_data" / "dataset"/ product_name_or_dir).resolve()

    if not product_dir.exists():
        raise FileNotFoundError(f"Product folder not found: {product_dir}")

    bands_dir = product_dir / "bands"
    geoloc_dir = product_dir / "geolocation"
    if not bands_dir.exists():
        raise FileNotFoundError(f"'bands' folder not found: {bands_dir}")
    if not geoloc_dir.exists():
        raise FileNotFoundError(f"'geolocation' folder not found: {geoloc_dir}")

    tiffs = list(bands_dir.glob("*.tif")) + list(bands_dir.glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"No TIFFs found in: {bands_dir}")

    def has_level_token(pth: Path, token: str) -> bool:
        s = pth.stem.lower()
        tok = token.lower()
        return (f"_{tok}_" in s) or s.endswith(f"_{tok}") or s.startswith(f"{tok}_") or (f"_{tok}" in s) or (f"{tok}_" in s)

    prefer = (prefer_level or "BC").strip().upper()
    if prefer not in {"BC", "AC", "RR"}:
        raise ValueError(f"prefer_level must be one of: BC, AC, RR (got {prefer_level!r})")

    priority = [prefer] + [x for x in ["BC", "AC", "RR"] if x != prefer]
    chosen_level, candidates = None, []
    for lvl in priority:
        lvl_matches = [pth for pth in tiffs if has_level_token(pth, lvl)]
        if lvl_matches:
            chosen_level, candidates = lvl, lvl_matches
            break
    if not candidates:
        names = "\n  - " + "\n  - ".join([pth.name for pth in sorted(tiffs)])
        raise FileNotFoundError(f"No BC/AC/RR TIFF found in {bands_dir}. Found:{names}")

    def score(pth: Path) -> tuple[int, int, int, str]:
        s = pth.stem.lower()
        is_multi = 1 if "multiband" in s else 0
        try:
            with rasterio.open(pth) as ds:
                nb = int(ds.count)
        except Exception:
            nb = -1
        try:
            sz = int(pth.stat().st_size)
        except Exception:
            sz = -1
        return (is_multi, nb, sz, pth.name.lower())

    best = sorted(candidates, key=score, reverse=True)[0]
    print(f"Selected level: {chosen_level}  TIFF: {best.name}")

    gl_scene = geoloc_dir / "GL_scene_0.json"
    if not gl_scene.exists():
        raise FileNotFoundError(f"Geolocation file not found: {gl_scene}")

    meta_candidates = sorted(product_dir.glob("*metadata*.json")) + sorted(product_dir.glob("**/*metadata*.json"))
    meta_candidates = [pth for pth in meta_candidates if pth.is_file() and pth.suffix.lower() == ".json"]
    metadata_json = meta_candidates[0] if meta_candidates else None

    return ProductPaths(product_dir, bands_dir, geoloc_dir, best, gl_scene, metadata_json)


def extract_corners_and_center(gl_scene_json_path: Path, size: int = 4096) -> tuple[dict, tuple[float, float]]:
    """extract_corners_and_center(gl_scene_json_path,size=4096)->(points_dict,(lat,lon)): Pull corner+center lat/lon from GL_scene_0.json."""
    with open(gl_scene_json_path, "r", encoding="utf-8") as f:
        gl = json.load(f)

    pts = gl["Geolocated_Points"]
    idx = {(p["X_coordinate"], p["Y_coordinate"]): p for p in pts}
    mid = size // 2
    want = {"top_left": (0, 0), "top_right": (size, 0), "bottom_right": (size, size), "bottom_left": (0, size), "center": (mid, mid)}

    out = {}
    for k, (x, y) in want.items():
        if (x, y) in idx:
            out[k] = idx[(x, y)]

    ctr = out.get("center")
    if ctr is None:
        return out, (float("nan"), float("nan"))
    return out, (float(ctr["Lat"]), float(ctr["Lon"]))


def load_band_center_wavelengths(metadata_json: Path) -> dict[str, float]:
    """load_band_center_wavelengths(metadata_json)->dict: Read BandCentreWavelength (nm) from session metadata json."""
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict) or len(meta) == 0:
        return {}
    root_key = next(iter(meta.keys()))
    imager_cfg = meta.get(root_key, {}).get("ImagerConfig", {})
    bcw = imager_cfg.get("BandCentreWavelength", {})
    if not isinstance(bcw, dict):
        return {}
    out = {}
    for k, v in bcw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


def choose_rgb_bands_from_wavelengths(center_wavelengths_nm: dict[str, float], tiff_band_count: int) -> tuple[int, int, int]:
    """choose_rgb_bands_from_wavelengths(center_wavelengths_nm,tiff_band_count)->(R,G,B): Pick 1-based bands closest to 665/560/490 nm."""
    targets = {"B": 490.0, "G": 560.0, "R": 665.0}
    entries = []
    for name, wl in center_wavelengths_nm.items():
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        band_num = int(m.group(1))  # metadata bands are 0..7
        entries.append((band_num, float(wl)))
    if not entries:
        return (4, 3, 2)  # sensible default for this product type

    def nearest(target_nm: float) -> int:
        band_num, _ = min(entries, key=lambda t: abs(t[1] - target_nm))
        idx_1based = band_num + 1
        if not (1 <= idx_1based <= tiff_band_count):
            raise RuntimeError(f"RGB selection out-of-range: {idx_1based} (TIFF bands={tiff_band_count})")
        return idx_1based

    return (nearest(targets["R"]), nearest(targets["G"]), nearest(targets["B"]))


def center_crop_uint8(img_rgb: np.ndarray, crop: int) -> tuple[np.ndarray, tuple[int, int]]:
    """center_crop_uint8(img_rgb,crop)->(cropped,(x0,y0)): Center crop HxWx3 uint8 image."""
    h, w = img_rgb.shape[:2]
    if crop > h or crop > w:
        raise ValueError(f"Crop {crop} too large for image {w}x{h}")
    x0 = (w - crop) // 2
    y0 = (h - crop) // 2
    return img_rgb[y0:y0 + crop, x0:x0 + crop].copy(), (x0, y0)


def read_band_phys(ds: rasterio.io.DatasetReader, b: int, win: rasterio.windows.Window) -> np.ndarray:
    """read_band_phys(ds,b,win)->array: Read band window and apply rasterio scales/offsets if present."""
    arr = ds.read(b, window=win, masked=True).astype(np.float32).filled(np.nan)

    scale = 1.0
    offset = 0.0
    try:
        if ds.scales and len(ds.scales) >= b and ds.scales[b - 1] not in (None, 0):
            scale = float(ds.scales[b - 1])
        if ds.offsets and len(ds.offsets) >= b and ds.offsets[b - 1] is not None:
            offset = float(ds.offsets[b - 1])
    except Exception:
        pass

    out = arr * scale + offset
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def read_radiance_rgb_from_tiff_crop(tiff_path: Path, rgb_bands_1based: tuple[int, int, int], x0: int, y0: int, crop: int) -> np.ndarray:
    """read_radiance_rgb_from_tiff_crop(tiff_path,(r,g,b),x0,y0,crop)->HxWx3: Read 3 bands from multiband TIFF window."""
    r_b, g_b, b_b = rgb_bands_1based
    win = rasterio.windows.Window(col_off=int(x0), row_off=int(y0), width=int(crop), height=int(crop))
    with rasterio.open(tiff_path) as ds:
        r = read_band_phys(ds, r_b, win)
        g = read_band_phys(ds, g_b, win)
        b = read_band_phys(ds, b_b, win)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def spd_effective_bandwidth_nm(spd_path: str) -> float:
    """spd_effective_bandwidth_nm(spd_path)->float: Effective bandwidth Δλ = ∫R(λ)dλ / max(R) (nm)."""
    arr = np.loadtxt(spd_path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad SPD format: {spd_path}")
    wl = arr[:, 0]
    r = arr[:, 1]
    ok = np.isfinite(wl) & np.isfinite(r)
    wl = wl[ok]
    r = r[ok]
    if wl.size < 2:
        return 0.0
    rmax = float(np.max(r))
    if not np.isfinite(rmax) or rmax <= 0:
        return 0.0
    area = float(np.trapezoid(r, wl))
    return area / rmax


def stats_rgb(arr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """stats_rgb(arr,mask=None)->dict: Per-channel min/max/mean/p1/p50/p99."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got {arr.shape}")
    m = mask.astype(bool) if mask is not None else None
    out = {"channels": {}}
    for ci, ch in enumerate(["R", "G", "B"]):
        x = arr[..., ci].astype(np.float64)
        x = x[m] if m is not None else x.reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out["channels"][ch] = {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "p1": np.nan, "p50": np.nan, "p99": np.nan}
            continue
        p1, p50, p99 = np.percentile(x, [1, 50, 99])
        out["channels"][ch] = {"n": int(x.size), "min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "p1": float(p1), "p50": float(p50), "p99": float(p99)}
    return out


def print_stats(title: str, arr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """print_stats(title,arr,mask=None)->dict: Print per-channel stats and return dict."""
    s = stats_rgb(arr, mask=mask)
    print(f"\n--- {title} ---")
    for ch in ["R", "G", "B"]:
        c = s["channels"][ch]
        print(f"{ch}: n={c['n']}  min={c['min']:.6g}  max={c['max']:.6g}  mean={c['mean']:.6g}  p1={c['p1']:.6g}  p50={c['p50']:.6g}  p99={c['p99']:.6g}")
    return s


def convert_generated_to_phisat_units(gen: np.ndarray, delta_nm: tuple[float, float, float], orig: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """convert_generated_to_phisat_units(gen,(dR,dG,dB),orig,mask)->(gen_conv,info): Pick best unit conversion so gen matches orig scale."""
    d_nm = np.array(delta_nm, dtype=np.float64)
    d_um = np.maximum(1e-12, d_nm / 1000.0).astype(np.float64)

    # Hypothesis A: gen is band-integrated radiance (W/m^2/sr). Convert to spectral per um by dividing Δλ_um.
    gen_A = gen.astype(np.float64) / d_um.reshape(1, 1, 3)

    # Hypothesis B: gen is spectral per nm (W/m^2/sr/nm). Convert to per um by multiplying 1000.
    gen_B = gen.astype(np.float64) * 1000.0

    # Hypothesis C: already per um (no change)
    gen_C = gen.astype(np.float64)

    def score(candidate: np.ndarray) -> float:
        m = mask.astype(bool) if mask is not None else None
        s = 0.0
        for i in range(3):
            a = orig[..., i].astype(np.float64)
            b = candidate[..., i].astype(np.float64)
            if m is not None:
                a = a[m]
                b = b[m]
            ok = np.isfinite(a) & np.isfinite(b)
            a = a[ok]
            b = b[ok]
            if a.size < 100:
                continue
            ra = np.percentile(a, 50)
            rb = np.percentile(b, 50)
            if ra <= 0 or rb <= 0:
                continue
            ratio = rb / ra
            s += abs(math.log(ratio))
        return float(s)

    scores = {"divide_by_dlambda_um": score(gen_A), "times_1000_nm_to_um": score(gen_B), "no_change": score(gen_C)}
    best_key = min(scores, key=scores.get)
    best = {"divide_by_dlambda_um": gen_A, "times_1000_nm_to_um": gen_B, "no_change": gen_C}[best_key]

    info = {"chosen": best_key, "scores": scores, "delta_nm": delta_nm, "delta_um": (float(d_um[0]), float(d_um[1]), float(d_um[2]))}
    return best.astype(np.float32), info


def main() -> None:
    """main()->None: Compare generated radiance to PHI-SAT crop with unit harmonization."""
    ROOT = Path(__file__).resolve().parents[2]
    os.chdir(ROOT)

    # ---- CONFIG YOU WILL EDIT ----
    product = str(ROOT / "dataset" / "phisat-2_data" / "dataset" / "offnadir_ocean2" / "PHISAT-2_L1_000001987_20250410143947_20250410143950_B05E6C3E")  # or full path to product dir
    # product = str(ROOT / "dataset" / "phisat-2_data" / "dataset" / "offnadir_ocean2" / "PHISAT-2_L1_000002103_20250423144634_20250423144637_B69C2A81")  # or full path to product dir

    prefer_level = "RR"
    crop_sz = 512

    wind_speed = 3.0  # m/s
    wave_properties['wind_speed'] = wind_speed
    bools["plot_3d"] = False




    # img_path = str(ROOT / "dataset" / "phisat-2_data" / "Auckland_SRW_WV2_PS_20110827_B26_002042_O_nadir.PNG")
    anns_path = None
    # ------------------------------

    paths = find_product_files(ROOT, product, prefer_level=prefer_level)

    # Print TIFF metadata we actually have
    with rasterio.open(paths.bands_tiff) as ds:
        print("\n--- PHI-SAT TIFF metadata ---")
        print(f"bands_tiff: {paths.bands_tiff}")
        print(f"chosen_level: {prefer_level}")
        print(f"units: {ds.units}")
        print(f"scales: {ds.scales}")
        print(f"offsets: {ds.offsets}")
        print(f"band1 tags: {ds.tags(1)}")

    t0, t1, tm = extract_acquisition_times_from_product_path(str(paths.product_dir))

    # Use TIFF dims for GL indexing (PHI-SAT is typically 4096x4096)
    with rasterio.open(paths.bands_tiff) as ds:
        gl_size = int(ds.width)

    extracted, (tgt_lat, tgt_lon) = extract_corners_and_center(paths.geoloc_json, size=gl_size)
    tgt_alt = float(extracted.get("center", {}).get("Alt", 0.0) or 0.0)

    # Determine RGB band indices from metadata wavelengths (your metadata snippet is perfect for this)
    rgb_bands = (4, 3, 2)
    if paths.metadata_json is not None:
        wl = load_band_center_wavelengths(paths.metadata_json)
        if wl:
            with rasterio.open(paths.bands_tiff) as ds:
                rgb_bands = choose_rgb_bands_from_wavelengths(wl, ds.count)
    r_b, g_b, b_b = rgb_bands

    # Make a quicklook (only to pick crop window)
    quicklook = paths.product_dir / f"quicklook_R{r_b}_G{g_b}_B{b_b}.png"
    if not quicklook.exists():
        # simple quicklook using percent stretch on raw values (display only)
        rgb = read_radiance_rgb_from_tiff_crop(paths.bands_tiff, rgb_bands, 0, 0, min(gl_size, gl_size))
        x = rgb.astype(np.float64)
        out = np.zeros_like(x, dtype=np.uint8)
        for i in range(3):
            v = x[..., i]
            v = v[np.isfinite(v)]
            lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0.0, 1.0)
            y = np.clip((x[..., i] - lo) / max(1e-12, (hi - lo)), 0.0, 1.0)
            out[..., i] = (255.0 * y).astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(quicklook)

    img_full = np.asarray(Image.open(quicklook).convert("RGB"))
    _, (x0, y0) = center_crop_uint8(img_full, crop_sz)

    # Build sensor dict used by your generator
    spd_folder = ROOT / "offnadir_imaging" / "spd_files"
    band_data = get_band_data(satellite, str(spd_folder))

    delta_R = spd_effective_bandwidth_nm(band_data["red"]["spd"])
    delta_G = spd_effective_bandwidth_nm(band_data["green"]["spd"])
    delta_B = spd_effective_bandwidth_nm(band_data["blue"]["spd"])
    print("\n--- Your SPD effective bandwidths (nm) ---")
    print("Δλ_R:", delta_R)
    print("Δλ_G:", delta_G)
    print("Δλ_B:", delta_B)

    # Geometry from your existing pipeline (kept as you had it)
    norad_id = 60470
    from read_image_L1_radiance_full import datetime_to_jd, fetch_nearest_tle_from_satchecker, sat_lla_from_tle_at_time
    jd_mid = datetime_to_jd(tm)
    tle1, tle2, tle_epoch = fetch_nearest_tle_from_satchecker(norad_id, jd_mid)
    lla = sat_lla_from_tle_at_time(tle1, tle2, tm)

    sat_lat = float(lla.lat_deg)
    sat_lon = float(lla.lon_deg)
    sat_alt = float(lla.alt_km * 1000.0)
    dt = tm

    # GSD from GL corners
    def haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = 6378137.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = p2 - p1
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
        return 2.0 * R * math.asin(math.sqrt(a))

    tl = extracted.get("top_left")
    tr = extracted.get("top_right")
    bl = extracted.get("bottom_left")
    if tl is None or tr is None or bl is None:
        raise RuntimeError("GL_scene_0.json missing required corner points.")
    width_m = haversine_m(float(tl["Lat"]), float(tl["Lon"]), float(tr["Lat"]), float(tr["Lon"]))
    height_m = haversine_m(float(tl["Lat"]), float(tl["Lon"]), float(bl["Lat"]), float(bl["Lon"]))
    gsd = float(0.5 * (width_m / float(gl_size) + height_m / float(gl_size)))

    sensor_phisat = dict(sensor_characteristics)
    sensor_phisat["resolution"] = int(crop_sz)
    sensor_phisat["GSD"] = float(gsd)

    # Run generation
    bools["use_annotations"] = False
    bools["generate_radiation"] = True
    bools["plot_result"] = False  # avoid blocking popups during comparisons

    (
        _texture_disp,
        _radiance_no_glint,
        _radiance_disp_no_glint,
        _rho_no_glint,
        _rho_disp_no_glint,
        radiance_final,
        _radiance_disp_final,
        _rho_final,
        _rho_disp_final,
        black_mask_full,
        _scale,
        offnadir_deg,
    ) = generate_image(
        img_path,
        anns_path,
        satellite,
        sat_lat,
        sat_lon,
        sat_alt,
        tgt_lat,
        tgt_lon,
        tgt_alt,
        dt,
        sensor_phisat,
        wave_properties,
        bools,
        seed_dem,
    )

    print("\n--- Geometry used ---")
    print(f"dt           : {dt.isoformat()}")
    print(f"sat_lat/lon  : {sat_lat:.6f}, {sat_lon:.6f}")
    print(f"sat_alt (m)  : {sat_alt:.1f}")
    print(f"tgt_lat/lon  : {tgt_lat:.6f}, {tgt_lon:.6f}")
    print(f"tgt_alt (m)  : {tgt_alt:.1f}")
    print(f"TLE epoch    : {tle_epoch}")
    print(f"Off Nadir    : {offnadir_deg:.2f} deg")
    print(f"GSD          : {sensor_phisat['GSD']:.3f} m/px")
    print(f"Crop         : {crop_sz}x{crop_sz} at x0={x0}, y0={y0}")
    print(f"RGB bands    : (R,G,B) = {rgb_bands}")

    if radiance_final is None:
        raise RuntimeError("radiance_final is None (generate_radiation=False or SPD failure).")

    # Read PHI-SAT crop in the same window
    orig = read_radiance_rgb_from_tiff_crop(paths.bands_tiff, rgb_bands, x0, y0, crop_sz)

    # Mask
    gen_mask = black_mask_full.astype(bool) if black_mask_full is not None else None

    print_stats("PHI-SAT crop (as stored in BC TIFF)", orig, mask=None)

    # Convert generated units to best match PHI-SAT convention
    gen = radiance_final.astype(np.float32)
    gen_conv, info = convert_generated_to_phisat_units(gen, (delta_R, delta_G, delta_B), orig, gen_mask)

    print("\n--- Unit harmonization decision ---")
    print("Chosen:", info["chosen"])
    print("Scores:", info["scores"])
    print("Δλ_um :", info["delta_um"])

    print_stats("GENERATED (raw)", gen, mask=gen_mask)
    print_stats("GENERATED (converted to PHI-SAT-like)", gen_conv, mask=gen_mask)

    # Report ratios at median to show remaining calibration mismatch (not units)
    print("\n--- Median ratios (generated_converted / phisat) ---")
    for i, ch in enumerate(["R", "G", "B"]):
        a = orig[..., i].astype(np.float64)
        b = gen_conv[..., i].astype(np.float64)
        if gen_mask is not None:
            a = a[gen_mask]
            b = b[gen_mask]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
        a = a[ok]
        b = b[ok]
        if a.size < 100:
            print(ch, ": n<100")
            continue
        ratio = np.percentile(b / a, 50)
        print(ch, f": {ratio:.4f}x")


if __name__ == "__main__":
    main()