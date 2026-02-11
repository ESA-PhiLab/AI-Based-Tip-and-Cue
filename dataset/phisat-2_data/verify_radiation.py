from pathlib import Path
import os
import math
import numpy as np
from PIL import Image
import rasterio

from settings import *
from offnadir_imaging.rendering import generate_image
from offnadir_imaging.functions.get_satellite_data import get_band_data

from read_image_L1_full import (
    find_product_files,
    extract_acquisition_times_from_product_path,
    extract_corners_and_center,
    datetime_to_jd,
    fetch_nearest_tle_from_satchecker,
    sat_lla_from_tle_at_time,
    load_band_center_wavelengths,
    choose_rgb_bands_from_wavelengths,
    save_rgb_png,
)


from offnadir_imaging.functions.convert_reference_frames import get_ecef_from_lat_lon
from offnadir_imaging.functions.intermediate_functions import is_dark_from_sun_dir

def refl_to_radiance(L_refl, Ebar, cos_theta_s):
    """refl_to_radiance(L_refl,Ebar,cos_theta_s) -> np.ndarray: Expected radiance from reflectance under Lambertian assumption."""
    return (L_refl * Ebar * cos_theta_s) / np.pi

def cos_theta_s_from_geometry(dt, sat_lat, sat_lon, sat_alt_m, tgt_lat, tgt_lon, tgt_alt_m) -> tuple[float, float]:
    """cos_theta_s_from_geometry(dt,sat_lat,sat_lon,sat_alt_m,tgt_lat,tgt_lon,tgt_alt_m) -> tuple[float,float]: Return (cos_theta_s, sun_elev_deg)."""
    sat_ecef, tgt_ecef, sun_ecef = get_ecef_from_lat_lon(
        float(sat_lat), float(sat_lon), float(sat_alt_m),
        float(tgt_lat), float(tgt_lon), float(tgt_alt_m),
        dt,
        generate_nadir=False,
    )

    _, elev_deg, _ = is_dark_from_sun_dir(
        target_ecef=tgt_ecef,
        sun_ecef=sun_ecef,
        threshold_deg=-90.0,   # don't gate; we just want the angle
        model="wgs84",
        dir_type="target_to_sun",
    )

    cos_theta_s = float(np.sin(np.deg2rad(float(elev_deg))))
    return cos_theta_s, float(elev_deg)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """haversine_m(lat1,lon1,lat2,lon2) -> float: Great-circle distance in meters."""
    R = 6378137.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def gsd_from_gl(extracted: dict, gl_size: int) -> float:
    """gsd_from_gl(extracted,gl_size) -> float: Approx meters/pixel from GL_scene corners."""
    tl = extracted.get("top_left")
    tr = extracted.get("top_right")
    bl = extracted.get("bottom_left")
    if tl is None or tr is None or bl is None:
        raise KeyError("Missing required GL points (top_left/top_right/bottom_left) to estimate GSD.")
    width_m = haversine_m(float(tl["Lat"]), float(tl["Lon"]), float(tr["Lat"]), float(tr["Lon"]))
    height_m = haversine_m(float(tl["Lat"]), float(tl["Lon"]), float(bl["Lat"]), float(bl["Lon"]))
    return float(0.5 * (width_m / float(gl_size) + height_m / float(gl_size)))


def center_crop_uint8(img_rgb: np.ndarray, crop: int) -> tuple[np.ndarray, tuple[int, int]]:
    """center_crop_uint8(img_rgb,crop) -> tuple[np.ndarray,tuple[int,int]]: Center crop HxWx3 to cropxcrop; returns (cropped,(x0,y0))."""
    h, w = img_rgb.shape[:2]
    if crop > h or crop > w:
        raise ValueError(f"Crop {crop} too large for image {w}x{h}")
    x0 = (w - crop) // 2
    y0 = (h - crop) // 2
    return img_rgb[y0:y0 + crop, x0:x0 + crop].copy(), (x0, y0)


def spd_integral(spd_path: str) -> float:
    """spd_integral(spd_path) -> float: Trapezoidal integral of an SPD file (wl, value)."""
    arr = np.loadtxt(spd_path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad SPD format: {spd_path}")
    wl = arr[:, 0]
    y = arr[:, 1]
    ok = np.isfinite(wl) & np.isfinite(y)
    wl = wl[ok]
    y = y[ok]
    if wl.size < 2:
        return 0.0
    return float(np.trapezoid(y, wl))


def reflectance_stats_rgb(arr: np.ndarray, mask: np.ndarray | None = None, name: str = "") -> dict:
    """reflectance_stats_rgb(arr,mask,name) -> dict: Per-channel stats (min/max/mean/p1/p50/p99)."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got {arr.shape}")
    m = mask.astype(bool) if mask is not None else None
    out = {"name": name, "channels": {}}
    for ci, ch in enumerate(["R", "G", "B"]):
        x = arr[..., ci].astype(np.float64)
        x = x[m] if m is not None else x.reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out["channels"][ch] = {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "p1": np.nan, "p50": np.nan, "p99": np.nan}
            continue
        p1, p50, p99 = np.percentile(x, [1, 50, 99])
        out["channels"][ch] = {
            "n": int(x.size),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "p1": float(p1),
            "p50": float(p50),
            "p99": float(p99),
        }
    return out


def print_stats(arr: np.ndarray, mask: np.ndarray | None = None, name: str = "reflectance") -> None:
    """print_stats(arr,mask,name) -> None: Print reflectance stats for an HxWx3 float array."""
    s = reflectance_stats_rgb(arr, mask=mask, name=name)
    print(f"\n--- {s['name']} ---")
    for ch in ["R", "G", "B"]:
        c = s["channels"][ch]
        print(
            f"{ch}: n={c['n']}  min={c['min']:.4f}  max={c['max']:.4f}  mean={c['mean']:.4f}  "
            f"p1={c['p1']:.4f}  p50={c['p50']:.4f}  p99={c['p99']:.4f}"
        )


def print_diff(a: dict, b: dict) -> None:
    """print_diff(a,b) -> None: Print (generated - original) deltas for key stats."""
    keys = ["min", "max", "mean", "p1", "p50", "p99"]
    print("\n--- Difference (generated - original) ---")
    for ch in ["R", "G", "B"]:
        da = a["channels"][ch]
        db = b["channels"][ch]
        parts = [f"Δ{k}={db[k]-da[k]:+.4f}" for k in keys]
        print(f"{ch}: " + "  ".join(parts))


def read_reflectance_rgb_from_tiff_crop(tiff_path: Path, rgb_bands_1based: tuple[int, int, int], x0: int, y0: int, crop: int) -> np.ndarray:
    """read_reflectance_rgb_from_tiff_crop(tiff_path,rgb_bands_1based,x0,y0,crop) -> np.ndarray: Read DN/10000 reflectance RGB from multiband TIFF window."""
    r_b, g_b, b_b = rgb_bands_1based
    win = rasterio.windows.Window(col_off=int(x0), row_off=int(y0), width=int(crop), height=int(crop))
    with rasterio.open(tiff_path) as ds:
        r = ds.read(r_b, window=win, masked=True).astype(np.float32).filled(np.nan) / 10000.0
        g = ds.read(g_b, window=win, masked=True).astype(np.float32).filled(np.nan) / 10000.0
        b = ds.read(b_b, window=win, masked=True).astype(np.float32).filled(np.nan) / 10000.0
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~np.isfinite(rgb)] = 0.0
    return np.clip(rgb, 0.0, 1.5).astype(np.float32)


if __name__ == "__main__":

    from settings import *

    ROOT = Path(__file__).resolve().parents[2]
    os.chdir(ROOT)

    spd_folder = ROOT / "offnadir_imaging" / "spd_files"
    band_data = get_band_data(satellite, str(spd_folder))

    product_name = (
        "phisat-2_data/dataset/"
        "offnadir_ocean2/"
        "PHISAT-2_L1_000001987_20250410143947_20250410143950_B05E6C3E"
    )

    paths = find_product_files(ROOT, product_name)

    anns_path = None
    bools["use_annotations"] = False
    bools["generate_radiation"] = True
    bools["plot_result"] = True

    # wave_properties['specular_weight'] = 1.0
    wave_properties['wind_speed'] = 3.0

    # Time + GL
    t0, t1, tm = extract_acquisition_times_from_product_path(str(paths.product_dir))

    gl_size = 4096
    try:
        with rasterio.open(paths.bands_tiff) as ds:
            gl_size = int(max(ds.width, ds.height))
    except Exception:
        gl_size = 4096

    extracted, (tgt_lat, tgt_lon) = extract_corners_and_center(paths.geoloc_json, size=gl_size)

    tgt_alt = 0.0
    ctr = extracted.get("center")
    if ctr and ctr.get("Alt") is not None:
        try:
            tgt_alt = float(ctr["Alt"])
        except Exception:
            tgt_alt = 0.0

    # Choose PHISAT RGB bands
    rgb_bands = (3, 2, 1)
    if paths.metadata_json is not None:
        try:
            wl = load_band_center_wavelengths(paths.metadata_json)
            if wl:
                with rasterio.open(paths.bands_tiff) as ds:
                    rgb_bands = choose_rgb_bands_from_wavelengths(wl, ds.count)
        except Exception:
            rgb_bands = (3, 2, 1)

    r_b, g_b, b_b = rgb_bands
    phisat_png = paths.product_dir / f"rgb_reflectance_R{r_b}_G{g_b}_B{b_b}.png"
    if not phisat_png.exists():
        save_rgb_png(paths.bands_tiff, rgb_bands, phisat_png)

    # Crop 128x128 from the PNG (for fast rendering)
    crop_sz = 128 * 4
    img_full = np.asarray(Image.open(phisat_png).convert("RGB"))
    img_crop, (x0, y0) = center_crop_uint8(img_full, crop_sz)

    phisat_crop_png = paths.product_dir / f"rgb_reflectance_R{r_b}_G{g_b}_B{b_b}_crop{crop_sz}_x{x0}_y{y0}.png"
    Image.fromarray(img_crop).save(phisat_crop_png)
    img_path = str(phisat_crop_png)

    # Satellite position
    norad_id = 60470
    jd_mid = datetime_to_jd(tm)
    tle1, tle2, tle_epoch = fetch_nearest_tle_from_satchecker(norad_id, jd_mid)
    lla = sat_lla_from_tle_at_time(tle1, tle2, tm)

    sat_lat = float(lla.lat_deg)
    sat_lon = float(lla.lon_deg)
    sat_alt = float(lla.alt_km * 1000.0)
    dt = tm

    # Override sensor for crop
    sensor_phisat = dict(sensor_characteristics)
    sensor_phisat["resolution"] = int(crop_sz)
    sensor_phisat["GSD"] = float(gsd_from_gl(extracted, gl_size=gl_size))

    print("\n--- Geometry used ---")
    print(f"dt           : {dt.isoformat()}")
    print(f"sat_lat/lon  : {sat_lat:.6f}, {sat_lon:.6f}")
    print(f"sat_alt (m)  : {sat_alt:.1f}")
    print(f"tgt_lat/lon  : {tgt_lat:.6f}, {tgt_lon:.6f}")
    print(f"tgt_alt (m)  : {tgt_alt:.1f}")
    print(f"TLE epoch    : {tle_epoch}")

    print("\n--- Inputs ---")
    print(f"bands_tiff   : {paths.bands_tiff}")
    print(f"full_png     : {phisat_png}")
    print(f"crop_png     : {img_path}")
    print(f"rgb_bands    : R,G,B = {rgb_bands}")
    print(f"crop         : {crop_sz}x{crop_sz} at x0={x0}, y0={y0}")
    print(f"GSD          : {sensor_phisat['GSD']:.3f} m/px")

    # Print SPD integrals (sanity check)
    print("\n--- Band SPD integrals ---")
    print("R:", spd_integral(band_data["red"]["spd"]))
    print("G:", spd_integral(band_data["green"]["spd"]))
    print("B:", spd_integral(band_data["blue"]["spd"]))

    # ORIGINAL reflectance from TIFF (DN/10000) for the same crop window
    orig_refl = read_reflectance_rgb_from_tiff_crop(
        tiff_path=paths.bands_tiff,
        rgb_bands_1based=rgb_bands,
        x0=x0,
        y0=y0,
        crop=crop_sz,
    )

    cos_theta_s, sun_elev_deg = cos_theta_s_from_geometry(
        dt=dt,
        sat_lat=sat_lat, sat_lon=sat_lon, sat_alt_m=sat_alt,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt_m=tgt_alt,
    )

    print_stats(orig_refl, name="ORIGINAL reflectance")
    orig_stats = reflectance_stats_rgb(orig_refl, mask=None, name="ORIGINAL (TIFF cropped)")

    img_path = "C:/Users/nadine/Downloads/oceanbg3.png"

    # Render
    (
        texture_disp, radiance_no_glint, radiance_disp_no_glint, rho_no_glint, rho_disp_no_glint, radiance_final, radiance_disp_final, rho_final, rho_disp_final, black_mask_full, scale, offnadir_deg

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

    print(f"\noffnadir angle = {offnadir_deg:.2f} deg")

    if rho_final is None:
        raise RuntimeError("rho_glint is None (SPD generation failed or generate_radiation=False).")

    gen_mask = black_mask_full.astype(bool) if black_mask_full is not None else None
    print_stats(rho_final.astype(np.float32), mask=gen_mask, name="GENERATED (TOA reflectance, masked)")
    gen_stats = reflectance_stats_rgb(rho_final.astype(np.float32), mask=gen_mask, name="GENERATED (TOA reflectance, masked)")

    print_diff(orig_stats, gen_stats)
