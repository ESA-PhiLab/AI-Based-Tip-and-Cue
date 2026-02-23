#!/usr/bin/env python3
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
    """extract_acquisition_times_from_product_path(any_path_in_product) -> (start, end, midpoint) UTC datetimes."""
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
    """find_product_files(base_dir, product_name) -> ProductPaths."""
    product_dir = (base_dir / "dataset" / product_name).resolve()
    if not product_dir.exists():
        raise FileNotFoundError(f"Product folder not found: {product_dir}")

    bands_dir = product_dir / "bands"
    geoloc_dir = product_dir / "geolocation"

    tiffs = sorted(bands_dir.glob("session_*_BC.tif*"))
    if len(tiffs) != 1:
        raise RuntimeError("Expected exactly one session_*_BC.tif file.")
    bands_tiff = tiffs[0]

    gl_scene = geoloc_dir / "GL_scene_0.json"

    meta_candidates = sorted(product_dir.glob("**/*metadata*.json"))
    metadata_json = meta_candidates[0] if meta_candidates else None

    return ProductPaths(product_dir, bands_dir, geoloc_dir, bands_tiff, gl_scene, metadata_json)


# ----------------------------
# RGB utilities
# ----------------------------

def stretch01_2d(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """stretch01_2d(band) -> np.ndarray in [0,1] using percentile stretch."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if hi <= lo:
        return np.zeros_like(band, dtype=np.float32)
    out = (band - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def make_rgb_linear01_from_tiff(tiff_path: Path, rgb_bands_1based: tuple[int, int, int]) -> np.ndarray:
    """make_rgb_linear01_from_tiff(tiff_path, rgb_bands) -> float RGB [0,1]."""
    r_b, g_b, b_b = rgb_bands_1based
    with rasterio.open(tiff_path) as ds:
        r = ds.read(r_b, masked=True).astype("float32").filled(np.nan) / 10000.0
        g = ds.read(g_b, masked=True).astype("float32").filled(np.nan) / 10000.0
        b = ds.read(b_b, masked=True).astype("float32").filled(np.nan) / 10000.0

    return np.dstack([stretch01_2d(r), stretch01_2d(g), stretch01_2d(b)])


def linear_to_DN255(img_linear: np.ndarray) -> np.ndarray:
    """linear_to_DN255(img_linear) -> uint8 RGB using gamma 1/2.2."""
    img = np.clip(img_linear, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    img_dn = np.clip(img * 255.0, 0.0, 255.0)
    return img_dn.astype(np.uint8)


def linear_to_uint8(img_linear: np.ndarray) -> np.ndarray:
    """linear_to_uint8(img_linear) -> uint8 RGB without gamma."""
    return np.clip(img_linear * 255.0, 0, 255).astype(np.uint8)


# ----------------------------
# Plot + save
# ----------------------------

def plot_and_save_all(tiff_path: Path, rgb_bands: tuple[int, int, int], out_dir: Path, extension: str) -> None:
    """plot_and_save_all(...) -> None: Save linear, gamma, and side-by-side figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb01 = make_rgb_linear01_from_tiff(tiff_path, rgb_bands)
    rgb_linear = linear_to_uint8(rgb01)
    rgb_gamma = linear_to_DN255(rgb01)

    r_b, g_b, b_b = rgb_bands

    # Save raw PNGs
    plt.imsave((out_dir / f"rgb_linear_{extension}.png").as_posix(), rgb_linear)
    plt.imsave((out_dir / f"rgb_gamma_{extension}.png").as_posix(), rgb_gamma)

    # Save linear figure
    fig1 = plt.figure(figsize=(6, 6))
    plt.imshow(rgb_linear)
    plt.title(f"RGB Linear (R,G,B={r_b},{g_b},{b_b})")
    plt.axis("off")
    fig1.savefig((out_dir / f"figure_linear_{extension}.png").as_posix(), bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # Save gamma figure
    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(rgb_gamma)
    plt.title(f"RGB Gamma 1/2.2 (R,G,B={r_b},{g_b},{b_b})")
    plt.axis("off")
    fig2.savefig((out_dir / f"figure_gamma_{extension}.png").as_posix(), bbox_inches="tight", dpi=300)
    plt.close(fig2)

    # Save side-by-side figure
    fig3, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(rgb_linear)
    ax[0].set_title("Linear")
    ax[0].axis("off")
    ax[1].imshow(rgb_gamma)
    ax[1].set_title("Gamma 1/2.2")
    ax[1].axis("off")
    fig3.savefig((out_dir / f"figure_comparison_{extension}.png").as_posix(), bbox_inches="tight", dpi=300)
    plt.close(fig3)

    print("Saved:")
    print(f"  rgb_linear_{extension}.png")
    print(f"  rgb_gamma_{extension}.png")
    print(f"  figure_linear_{extension}.png")
    print(f"  figure_gamma_{extension}.png")
    print(f"  figure_comparison_{extension}.png")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    base_dir = Path.cwd()
    product_name = "offnadir_ocean2/PHISAT-2_L1_000001987_20250410143947_20250410143950_B05E6C3E"

    paths = find_product_files(base_dir, product_name)

    rgb_bands =  (4,3,2)

    # Save in script directory (where this .py file lives)
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "rgb_outputs"

    print(f"Saving outputs to: {output_dir.resolve()}")

    plot_and_save_all(paths.bands_tiff, rgb_bands, output_dir)