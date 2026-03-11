#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import rasterio

import matplotlib
import matplotlib.pyplot as plt
from openpyxl import Workbook

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from settings import *
from offnadir_imaging.rendering import generate_image
from offnadir_imaging.functions.get_satellite_data import get_band_data
from offnadir_imaging.functions.convert_reference_frames import get_ecef_from_lat_lon
from offnadir_imaging.functions.intermediate_functions import is_dark_from_sun_dir

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

@dataclass(frozen=True)
class SweepPlotEntry:
    dataset: str
    input_img: str
    rows: list[Any]
    best_wind: float
    orig_stats: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    wind: float
    score: float
    offnadir_deg: float
    gen_stats: dict[str, Any]


@dataclass(frozen=True)
class BestResultEntry:
    dataset: str
    input_img: str
    best: RunResult
    true_stats: dict[str, Any]
    input_stats: dict[str, Any]

def plot_band_combined(
    entries: list[SweepPlotEntry],
    dataset: str,
    channel: str,
    stat_key: str,
    out_path: Path,
    legend_fontsize: int = 12,
    axis_fontsize: int = 14,
) -> None:
    """plot_band_combined(entries,dataset,channel,stat_key,out_path,legend_fontsize,axis_fontsize) -> None"""

    if not entries:
        return

    label_map = {
        "Auckland_SRW_QB2_PS_20060812_B32_002794_O_nadir": "Auckland 2006",
        "Auckland_SRW_WV2_PS_20110827_B26_002042_O_nadir": "Auckland 2011",
        "Maui_HB_WV3_PS_20150109_B16_000885_O_nadir": "Maui 2015",
        "PelagosIm2_FW_WV3_PS_20160619_B1_000136_O_nadir": "Pelagos 2016",
        "Valdes_SRW_WV2_PS_20160923_B109_001361_O_nadir": "Valdes 2014",
    }

    shade_maps = {
        "R": plt.cm.Reds(np.linspace(0.45, 0.9, max(len(entries), 2))),
        "G": plt.cm.Greens(np.linspace(0.45, 0.9, max(len(entries), 2))),
        "B": plt.cm.Blues(np.linspace(0.45, 0.9, max(len(entries), 2))),
    }

    colors = shade_maps[channel]

    plt.figure()

    for i, entry in enumerate(entries):

        winds = np.array([r.wind_speed for r in entry.rows], dtype=np.float64)
        gen_vals = np.array([getattr(r, f"{channel}_{stat_key}") for r in entry.rows], dtype=np.float64)

        orig_val = float(entry.orig_stats["channels"][channel][stat_key])
        color = colors[i]

        label = label_map.get(entry.input_img, entry.input_img)

        plt.plot(
            winds,
            gen_vals,
            linewidth=2,
            label=label,
            color=color,
        )

        plt.scatter(
            [entry.best_wind],
            [orig_val],
            s=90,
            marker="o",
            color=color,
        )

    plt.xlabel("Wind speed (m/s)", fontsize=axis_fontsize)
    plt.ylabel("Reflectance", fontsize=axis_fontsize)

    plt.title(
        f"{dataset} {channel} reflectance vs wind speed ({stat_key})",
        fontsize=axis_fontsize + 2,
    )

    plt.grid(True)

    plt.legend(fontsize=legend_fontsize)

    plt.ylim(0.0, 0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close()


def extract_id(name: str) -> str:
    """extract_id(name) -> str: Return last underscore-separated token from filename string."""
    return name.split("_")[-1]


def dataset_tag_from_product_name(product_name: str) -> str:
    """dataset_tag_from_product_name(product_name) -> str: Return C3E/A81 from product name suffix."""
    product_id = extract_id(product_name)
    if product_id.endswith("C3E"):
        return "C3E"
    if product_id.endswith("A81"):
        return "A81"
    return product_id


def compute_channel_mean_rows(rows: list[list[Any]]) -> list[Any]:
    """compute_channel_mean_rows(rows) -> list[Any]: Mean row for stats columns, optional blanks for non-applicable columns."""
    if not rows:
        return ["", "MEAN", "", "", "", "", "", "", "", "", "", "", "", ""]

    n_vals = [float(r[4]) for r in rows]
    min_vals = [float(r[5]) for r in rows]
    max_vals = [float(r[6]) for r in rows]
    mean_vals = [float(r[7]) for r in rows]
    p1_vals = [float(r[8]) for r in rows]
    p50_vals = [float(r[9]) for r in rows]
    p99_vals = [float(r[10]) for r in rows]

    def mean_optional(col_idx: int) -> float | str:
        vals = [r[col_idx] for r in rows if r[col_idx] not in ("", None)]
        return float(np.mean([float(v) for v in vals])) if vals else ""

    return [
        "",
        "MEAN",
        "",
        "",
        int(round(np.mean(n_vals))),
        float(np.mean(min_vals)),
        float(np.mean(max_vals)),
        float(np.mean(mean_vals)),
        float(np.mean(p1_vals)),
        float(np.mean(p50_vals)),
        float(np.mean(p99_vals)),
        mean_optional(11),
        mean_optional(12),
        mean_optional(13),
    ]


def write_grouped_dataset_excel(xlsx_path: Path, dataset: str, results: list[BestResultEntry]) -> None:
    """write_grouped_dataset_excel(xlsx_path,dataset,results) -> None: Write grouped Excel with generated rows, means, true rows, true means, input PNG rows, and input PNG means."""
    wb = Workbook()
    ws = wb.active
    ws.title = "best_results"

    ws.append([
        "dataset",
        "input_img",
        "source",
        "channel",
        "n",
        "min",
        "max",
        "mean",
        "p1",
        "p50",
        "p99",
        "best_wind_speed",
        "best_score",
        "offnadir_deg",
    ])

    for ch in ["R", "G", "B"]:
        gen_rows: list[list[Any]] = []
        true_rows: list[list[Any]] = []
        input_rows: list[list[Any]] = []

        for item in results:
            c_gen = item.best.gen_stats["channels"][ch]
            gen_rows.append([
                dataset,
                item.input_img,
                "generated_best",
                ch,
                int(c_gen["n"]),
                float(c_gen["min"]),
                float(c_gen["max"]),
                float(c_gen["mean"]),
                float(c_gen["p1"]),
                float(c_gen["p50"]),
                float(c_gen["p99"]),
                float(item.best.wind),
                float(item.best.score),
                float(item.best.offnadir_deg),
            ])

            c_true = item.true_stats["channels"][ch]
            true_rows.append([
                dataset,
                item.input_img,
                "phisat_true",
                ch,
                int(c_true["n"]),
                float(c_true["min"]),
                float(c_true["max"]),
                float(c_true["mean"]),
                float(c_true["p1"]),
                float(c_true["p50"]),
                float(c_true["p99"]),
                float(item.best.wind),
                float(item.best.score),
                float(item.best.offnadir_deg),
            ])

            c_input = item.input_stats["channels"][ch]
            input_rows.append([
                dataset,
                item.input_img,
                "input_png",
                ch,
                int(c_input["n"]),
                float(c_input["min"]),
                float(c_input["max"]),
                float(c_input["mean"]),
                float(c_input["p1"]),
                float(c_input["p50"]),
                float(c_input["p99"]),
                "",
                "",
                "",
            ])

        for row in gen_rows:
            ws.append(row)

        mean_gen = compute_channel_mean_rows(gen_rows)
        ws.append([
            dataset,
            mean_gen[1],
            "generated_best",
            ch,
            mean_gen[4],
            mean_gen[5],
            mean_gen[6],
            mean_gen[7],
            mean_gen[8],
            mean_gen[9],
            mean_gen[10],
            mean_gen[11],
            mean_gen[12],
            mean_gen[13],
        ])

        for row in true_rows:
            ws.append(row)

        mean_true = compute_channel_mean_rows(true_rows)
        ws.append([
            dataset,
            mean_true[1],
            "phisat_true",
            ch,
            mean_true[4],
            mean_true[5],
            mean_true[6],
            mean_true[7],
            mean_true[8],
            mean_true[9],
            mean_true[10],
            mean_true[11],
            mean_true[12],
            mean_true[13],
        ])

        for row in input_rows:
            ws.append(row)

        mean_input = compute_channel_mean_rows(input_rows)
        ws.append([
            dataset,
            mean_input[1],
            "input_png",
            ch,
            mean_input[4],
            mean_input[5],
            mean_input[6],
            mean_input[7],
            mean_input[8],
            mean_input[9],
            mean_input[10],
            mean_input[11],
            mean_input[12],
            mean_input[13],
        ])

        ws.append([""] * 14)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)


def plot_reflectance_vs_wind(
    rows: list[Any],
    orig_stats: dict[str, Any],
    best_wind: float,
    out_path: Path,
    stat_key: str,
    axis_fontsize: int = 14,
    legend_fontsize: int = 12,
) -> None:
    """plot_reflectance_vs_wind(rows,orig_stats,best_wind,out_path,stat_key,axis_fontsize,legend_fontsize) -> None"""

    winds = np.array([r.wind_speed for r in rows], dtype=np.float64)

    gen_R = np.array([getattr(r, f"R_{stat_key}") for r in rows], dtype=np.float64)
    gen_G = np.array([getattr(r, f"G_{stat_key}") for r in rows], dtype=np.float64)
    gen_B = np.array([getattr(r, f"B_{stat_key}") for r in rows], dtype=np.float64)

    orig_R = float(orig_stats["channels"]["R"][stat_key])
    orig_G = float(orig_stats["channels"]["G"][stat_key])
    orig_B = float(orig_stats["channels"]["B"][stat_key])

    plt.figure()

    plt.plot(winds, gen_R, color="red", linewidth=2, label=f"Generated R ({stat_key})")
    plt.plot(winds, gen_G, color="green", linewidth=2, label=f"Generated G ({stat_key})")
    plt.plot(winds, gen_B, color="blue", linewidth=2, label=f"Generated B ({stat_key})")

    plt.scatter([best_wind], [orig_R], color="red", s=80, marker="o", label=f"Original R ({stat_key})")
    plt.scatter([best_wind], [orig_G], color="green", s=80, marker="o", label=f"Original G ({stat_key})")
    plt.scatter([best_wind], [orig_B], color="blue", s=80, marker="o", label=f"Original B ({stat_key})")

    plt.xlabel("Wind speed (m/s)", fontsize=axis_fontsize)
    plt.ylabel("Reflectance", fontsize=axis_fontsize)

    plt.title(
        f"Reflectance vs wind speed ({stat_key})",
        fontsize=axis_fontsize + 2,
    )

    plt.grid(True)

    plt.legend(fontsize=legend_fontsize)

    plt.ylim(0.0, 0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """haversine_m(lat1,lon1,lat2,lon2) -> float: Great-circle distance in meters."""
    R = 6378137.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def gsd_from_gl(extracted: dict[str, Any], gl_size: int) -> float:
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
    """center_crop_uint8(img_rgb,crop) -> tuple[np.ndarray,tuple[int,int]]: Center crop HxWx3 to cropxcrop."""
    h, w = img_rgb.shape[:2]
    if crop > h or crop > w:
        raise ValueError(f"Crop {crop} too large for image {w}x{h}")
    x0 = (w - crop) // 2
    y0 = (h - crop) // 2
    return img_rgb[y0:y0 + crop, x0:x0 + crop].copy(), (x0, y0)


def reflectance_stats_rgb(arr: np.ndarray, mask: np.ndarray | None = None, name: str = "") -> dict[str, Any]:
    """reflectance_stats_rgb(arr,mask,name) -> dict[str,Any]: Per-channel stats min/max/mean/p1/p50/p99."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got {arr.shape}")
    m = mask.astype(bool) if mask is not None else None
    out: dict[str, Any] = {"name": name, "channels": {}}
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
        print(f"{ch}: n={c['n']}  min={c['min']:.4f}  max={c['max']:.4f}  mean={c['mean']:.4f}  p1={c['p1']:.4f}  p50={c['p50']:.4f}  p99={c['p99']:.4f}")


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


def cos_theta_s_from_geometry(dt, sat_lat: float, sat_lon: float, sat_alt_m: float, tgt_lat: float, tgt_lon: float, tgt_alt_m: float) -> tuple[float, float]:
    """cos_theta_s_from_geometry(dt,sat_lat,sat_lon,sat_alt_m,tgt_lat,tgt_lon,tgt_alt_m) -> tuple[float,float]: Solar elevation and cos(theta_s)."""
    sat_ecef, tgt_ecef, sun_ecef = get_ecef_from_lat_lon(
        float(sat_lat), float(sat_lon), float(sat_alt_m),
        float(tgt_lat), float(tgt_lon), float(tgt_alt_m),
        dt,
        generate_nadir=False,
    )
    _, elev_deg, _ = is_dark_from_sun_dir(
        target_ecef=tgt_ecef,
        sun_ecef=sun_ecef,
        threshold_deg=-90.0,
        model="wgs84",
        dir_type="target_to_sun",
    )
    cos_theta_s = float(np.sin(np.deg2rad(float(elev_deg))))
    return cos_theta_s, float(elev_deg)


def objective_log_ratio(orig_stats: dict[str, Any], gen_stats: dict[str, Any], keys: tuple[str, ...] = ("p50", "p99"), eps: float = 1e-9) -> float:
    """objective_log_ratio(orig_stats,gen_stats,keys,eps) -> float: Sum |log(gen/orig)| over channels and keys."""
    s = 0.0
    for ch in ["R", "G", "B"]:
        o = orig_stats["channels"][ch]
        g = gen_stats["channels"][ch]
        for k in keys:
            ov = float(o[k])
            gv = float(g[k])
            if not np.isfinite(ov) or not np.isfinite(gv):
                continue
            ov = max(eps, ov)
            gv = max(eps, gv)
            s += abs(math.log(gv / ov))
    return float(s)


def score_breakdown_logratio(orig_stats: dict[str, Any], gen_stats: dict[str, Any], keys: tuple[str, ...] = ("p50", "p99"), eps: float = 1e-9) -> dict[str, float]:
    """score_breakdown_logratio(orig_stats,gen_stats,keys,eps) -> dict[str,float]: Per-channel sum |log(gen/orig)|."""
    out = {}
    for ch in ["R", "G", "B"]:
        s = 0.0
        o = orig_stats["channels"][ch]
        g = gen_stats["channels"][ch]
        for k in keys:
            ov = max(eps, float(o[k]))
            gv = max(eps, float(g[k]))
            if np.isfinite(ov) and np.isfinite(gv):
                s += abs(math.log(gv / ov))
        out[ch] = float(s)
    out["total"] = float(out["R"] + out["G"] + out["B"])
    return out


def objective_rmse(orig_stats: dict[str, Any], gen_stats: dict[str, Any], keys: tuple[str, ...] = ("mean", "p50", "p99")) -> float:
    """objective_rmse(orig_stats,gen_stats,keys) -> float: RMSE over channels and keys."""
    vals = []
    for ch in ["R", "G", "B"]:
        o = orig_stats["channels"][ch]
        g = gen_stats["channels"][ch]
        for k in keys:
            ov = float(o[k])
            gv = float(g[k])
            if np.isfinite(ov) and np.isfinite(gv):
                vals.append((gv - ov) ** 2)
    if not vals:
        return float("inf")
    return float(math.sqrt(sum(vals) / len(vals)))


def frange(start: float, stop: float, step: float) -> list[float]:
    """frange(start,stop,step) -> list[float]: Inclusive float range."""
    if step <= 0:
        raise ValueError("step must be > 0")
    out = []
    x = float(start)
    while x <= float(stop) + 1e-12:
        out.append(round(x, 6))
        x += float(step)
    return out


if __name__ == "__main__":
    bools["plot_3d"] = False

    product_names = [
       # "PHISAT-2_L1_000001987_20250410143947_20250410143950_B05E6C3E",
        "PHISAT-2_L1_000002103_20250423144634_20250423144637_B69C2A81",
    ]

    png_images = [
        #"Auckland_SRW_QB2_PS_20060812_B32_002794_O_nadir",
        #"Auckland_SRW_WV2_PS_20110827_B26_002042_O_nadir",
        #"Maui_HB_WV3_PS_20150109_B16_000885_O_nadir",
        "PelagosIm2_FW_WV3_PS_20160619_B1_000136_O_nadir",
        "Valdes_SRW_WV2_PS_20160923_B109_001361_O_nadir",
    ]

    WIND_MIN = 2.0
    WIND_MAX = 12.0
    WIND_STEP = 4.0 #0.25

    LEGEND_FONTSIZE = 14
    AXIS_FONTSIZE = 16

    ROOT = Path(__file__).resolve().parents[2]
    os.chdir(ROOT)

    dataset_results: dict[str, list[BestResultEntry]] = {"C3E": [], "A81": []}
    dataset_plot_entries: dict[str, list[SweepPlotEntry]] = {"C3E": [], "A81": []}

    for prod in product_names:
        PRODUCT_NAME = "phisat-2_data/dataset/offnadir_ocean2/" + str(prod)
        dataset = dataset_tag_from_product_name(PRODUCT_NAME)

        for img0 in png_images:
            print(f"START TASK FOR {img0}")

            img_handle = img0
            img_name = img_handle + ".PNG"

            IMG_PATH = os.path.join("dataset", "phisat-2_data", img_name)

            extension = img_handle
            img_id = extract_id(PRODUCT_NAME)

            NORAD_ID = 60470
            CROP_MULT = 4
            METRIC = "both"

            out_dir = ROOT / "rgb_outputs" / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"wind_sweep_results_{img_id}_{extension}.csv"

            paths = find_product_files(ROOT, PRODUCT_NAME)

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

            crop_sz = 128 * int(CROP_MULT)
            img_full = np.asarray(Image.open(phisat_png).convert("RGB"))
            img_crop, (x0, y0) = center_crop_uint8(img_full, crop_sz)

            orig_refl = read_reflectance_rgb_from_tiff_crop(
                tiff_path=paths.bands_tiff,
                rgb_bands_1based=rgb_bands,
                x0=x0,
                y0=y0,
                crop=crop_sz,
            )
            orig_stats = reflectance_stats_rgb(orig_refl, mask=None, name="ORIGINAL (TIFF cropped)")
            print_stats(orig_refl, name="ORIGINAL reflectance (TIFF crop)")

            input_img_arr = np.asarray(Image.open(IMG_PATH).convert("RGB"), dtype=np.float32) / 255.0
            input_img_stats = reflectance_stats_rgb(input_img_arr, mask=None, name="INPUT PNG")

            jd_mid = datetime_to_jd(tm)
            tle1, tle2, tle_epoch = fetch_nearest_tle_from_satchecker(NORAD_ID, jd_mid)
            lla = sat_lla_from_tle_at_time(tle1, tle2, tm)

            sat_lat = float(lla.lat_deg)
            sat_lon = float(lla.lon_deg)
            sat_alt = float(lla.alt_km * 1000.0)
            dt = tm

            cos_theta_s, sun_elev_deg = cos_theta_s_from_geometry(
                dt=dt,
                sat_lat=sat_lat,
                sat_lon=sat_lon,
                sat_alt_m=sat_alt,
                tgt_lat=tgt_lat,
                tgt_lon=tgt_lon,
                tgt_alt_m=tgt_alt,
            )

            sensor_phisat = dict(sensor_characteristics)
            sensor_phisat["resolution"] = int(crop_sz)
            sensor_phisat["GSD"] = float(gsd_from_gl(extracted, gl_size=gl_size))

            anns_path = None
            bools_local = dict(bools)
            bools_local["use_annotations"] = False
            bools_local["generate_radiation"] = True
            bools_local["plot_result"] = False

            wave_base = dict(wave_properties)

            spd_folder = ROOT / "offnadir_imaging" / "spd_files"
            _ = get_band_data(satellite, str(spd_folder))

            winds = frange(WIND_MIN, WIND_MAX, WIND_STEP)

            @dataclass(frozen=True)
            class Row:
                wind_speed: float
                score: float
                offnadir_deg: float
                R_mean: float
                R_p50: float
                R_p99: float
                G_mean: float
                G_p50: float
                G_p99: float
                B_mean: float
                B_p50: float
                B_p99: float

            rows: list[Row] = []
            best: RunResult | None = None

            for wind in winds:
                wave_cur = dict(wave_base)
                wave_cur["wind_speed"] = float(wind)

                (
                    texture_disp,
                    radiance_no_glint,
                    radiance_disp_no_glint,
                    rho_no_glint,
                    rho_disp_no_glint,
                    radiance_final,
                    radiance_disp_final,
                    rho_final,
                    rho_disp_final,
                    black_mask_full,
                    scale,
                    offnadir_deg,
                ) = generate_image(
                    IMG_PATH,
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
                    wave_cur,
                    bools_local,
                    seed_dem,
                )

                if rho_final is None:
                    print(f"[wind={wind:.3f}] rho_final is None -> skipping")
                    continue

                gen_mask = black_mask_full.astype(bool) if black_mask_full is not None else None
                gen_stats = reflectance_stats_rgb(rho_final.astype(np.float32), mask=gen_mask, name=f"GEN wind={wind:.3f}")

                score = 0.0
                if METRIC == "logratio" or METRIC == "both":
                    score_log = objective_log_ratio(orig_stats, gen_stats, keys=("mean", "p50"))
                    print(score_log)
                    score += score_log

                if METRIC == "rmse" or METRIC == "both":
                    score_rmse = objective_rmse(orig_stats, gen_stats, keys=("mean", "p50"))
                    print(score_rmse)
                    score += score_rmse * 50.0

                bd = score_breakdown_logratio(orig_stats, gen_stats, keys=("mean", "p50"))
                print("score breakdown (|log(gen/orig)|, p50+p99):", bd)

                rr = RunResult(wind=float(wind), score=float(score), offnadir_deg=float(offnadir_deg), gen_stats=gen_stats)
                if best is None or rr.score < best.score:
                    best = rr

                ch = gen_stats["channels"]
                rows.append(
                    Row(
                        wind_speed=float(wind),
                        score=float(score),
                        offnadir_deg=float(offnadir_deg),
                        R_mean=float(ch["R"]["mean"]),
                        R_p50=float(ch["R"]["p50"]),
                        R_p99=float(ch["R"]["p99"]),
                        G_mean=float(ch["G"]["mean"]),
                        G_p50=float(ch["G"]["p50"]),
                        G_p99=float(ch["G"]["p99"]),
                        B_mean=float(ch["B"]["mean"]),
                        B_p50=float(ch["B"]["p50"]),
                        B_p99=float(ch["B"]["p99"]),
                    )
                )

                print(f"[wind={wind:.3f}] score={score:.6f}  offnadir={offnadir_deg:.2f} deg")

            if best is None:
                raise RuntimeError("No successful runs; check renderer outputs and paths.")

            print("\n=== BEST MATCH ===")
            print(f"dataset    : {dataset}")
            print(f"input_img  : {img_handle}")
            print(f"metric     : {METRIC}")
            print(f"wind_speed : {best.wind:.6f} m/s")
            print(f"score      : {best.score:.6f}")
            print(f"offnadir   : {best.offnadir_deg:.2f} deg")

            with out_csv.open("w", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["wind_speed", "score", "offnadir_deg", "R_mean", "R_p50", "R_p99", "G_mean", "G_p50", "G_p99", "B_mean", "B_p50", "B_p99"])
                for r in rows:
                    wcsv.writerow([
                        f"{r.wind_speed:.6f}",
                        f"{r.score:.12f}",
                        f"{r.offnadir_deg:.6f}",
                        f"{r.R_mean:.8f}",
                        f"{r.R_p50:.8f}",
                        f"{r.R_p99:.8f}",
                        f"{r.G_mean:.8f}",
                        f"{r.G_p50:.8f}",
                        f"{r.G_p99:.8f}",
                        f"{r.B_mean:.8f}",
                        f"{r.B_p50:.8f}",
                        f"{r.B_p99:.8f}",
                    ])

            print(f"\nWrote sweep table: {out_csv.resolve()}")

            plot_reflectance_vs_wind(rows, orig_stats, best.wind, out_dir / f"reflectance_vs_wind_p50_{img_id}_{extension}.png", stat_key="p50", axis_fontsize=AXIS_FONTSIZE, legend_fontsize=LEGEND_FONTSIZE)
            plot_reflectance_vs_wind(rows, orig_stats, best.wind, out_dir / f"reflectance_vs_wind_p99_{img_id}_{extension}.png", stat_key="p99", axis_fontsize=AXIS_FONTSIZE, legend_fontsize=LEGEND_FONTSIZE)
            plot_reflectance_vs_wind(rows, orig_stats, best.wind, out_dir / f"reflectance_vs_wind_mean_{img_id}_{extension}.png", stat_key="mean", axis_fontsize=AXIS_FONTSIZE, legend_fontsize=LEGEND_FONTSIZE)
            print(f"Saved plots to: {out_dir.resolve()}")

            dataset_results.setdefault(dataset, []).append(
                BestResultEntry(
                    dataset=dataset,
                    input_img=img_handle,
                    best=best,
                    true_stats=orig_stats,
                    input_stats=input_img_stats,
                )
            )

            dataset_plot_entries.setdefault(dataset, []).append(
                SweepPlotEntry(
                    dataset=dataset,
                    input_img=img_handle,
                    rows=rows,
                    best_wind=best.wind,
                    orig_stats=orig_stats,
                )
            )

    for dataset, results in dataset_results.items():
        if not results:
            continue
        out_xlsx = ROOT / "rgb_outputs" / f"best_wind_summary_{dataset}.xlsx"
        write_grouped_dataset_excel(out_xlsx, dataset, results)
        print(f"Wrote grouped Excel for {dataset}: {out_xlsx.resolve()}")

    for dataset, entries in dataset_plot_entries.items():
        if not entries:
            continue

        plot_dir = ROOT / "rgb_outputs" / dataset

        for channel in ["R", "G", "B"]:
            plot_band_combined(
                entries=entries,
                dataset=dataset,
                channel=channel,
                stat_key="mean",
                out_path=plot_dir / f"reflectance_vs_wind_all_{channel}_{dataset}.png",
                legend_fontsize=LEGEND_FONTSIZE,
                axis_fontsize=AXIS_FONTSIZE,
            )

        print(f"Saved combined channel plots for {dataset}: {plot_dir.resolve()}")