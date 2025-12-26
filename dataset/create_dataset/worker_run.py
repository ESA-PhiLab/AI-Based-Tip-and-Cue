# worker_run.py
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import numpy as np
import matplotlib.pyplot as plt

from create_patch import generate_patch
from save_patch import save_patch
from translate_patch import translate_offnadir, add_sunglint


def cleanup() -> None:
    """cleanup() -> None: Close figures to avoid hanging/leaks."""
    try:
        plt.close("all")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    """parse_args() -> argparse.Namespace: Parse CLI args."""
    p = argparse.ArgumentParser()

    # Core inputs
    p.add_argument("--img_file", type=str, required=True)
    p.add_argument("--patch_seed", type=int, required=True)
    p.add_argument("--dem_seed", type=int, required=True)
    p.add_argument("--show_plot", type=int, default=0)
    p.add_argument("--render_resolution", type=int, default=124)

    # Off-nadir pose/time
    p.add_argument("--sat_lat", type=float, required=True)
    p.add_argument("--sat_lon", type=float, required=True)
    p.add_argument("--sat_alt", type=float, required=True)

    p.add_argument("--tgt_lat", type=float, required=True)
    p.add_argument("--tgt_lon", type=float, required=True)
    p.add_argument("--tgt_alt", type=float, required=True)

    p.add_argument("--datetime_utc", type=str, default="")

    # generate_patch parameters (treated as general settings)
    p.add_argument("--mode_single", type=str, default="full")
    p.add_argument("--mode_multiple_allow_partial", type=int, default=0)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--nowhale_max_fraction", type=float, default=0.10)
    p.add_argument("--whale_min_fraction", type=float, default=0.99)
    p.add_argument("--half_fraction_low", type=float, default=0.20)
    p.add_argument("--half_fraction_high", type=float, default=0.80)
    p.add_argument("--mask_alpha", type=int, default=80)

    # Meta out (absolute path passed by main)
    p.add_argument("--meta_out", type=str, required=True)

    return p.parse_args()


def classify_label(fracs: list[float],
                   whale_min_fraction: float,
                   half_fraction_range: tuple[float, float],
                   nowhale_max_fraction: float) -> str:
    """classify_label(fracs,whale_min_fraction,half_fraction_range,nowhale_max_fraction) -> str: Return whale/whale_half/ocean."""
    if not fracs:
        return "ocean"
    fmax = float(max(fracs))
    if fmax <= float(nowhale_max_fraction):
        return "ocean"
    if any(float(f) >= float(whale_min_fraction) for f in fracs):
        return "whale"
    lo, hi = float(half_fraction_range[0]), float(half_fraction_range[1])
    if any(lo <= float(f) <= hi for f in fracs):
        return "whale_half"
    return "whale"


def ann_to_row_dict(ann: dict) -> dict:
    """ann_to_row_dict(ann) -> dict: Normalize annotation dict to row-friendly fields."""
    known = {"id", "image_id", "category_id", "bbox", "segmentation", "area", "iscrowd"}
    other = {k: v for k, v in ann.items() if k not in known}
    return {
        "annotation_id": ann.get("id", None),
        "image_id": ann.get("image_id", None),
        "category_id": ann.get("category_id", None),
        "bbox": ann.get("bbox", None),
        "segmentation": ann.get("segmentation", None),
        "area": ann.get("area", None),
        "iscrowd": ann.get("iscrowd", None),
        "other": other if other else None,
    }


def write_meta(meta_out: str,
               patch_bundle: dict,
               label_simple: str,
               half_fraction_range: tuple[float, float],
               whale_min_fraction: float,
               nowhale_max_fraction: float) -> None:
    """write_meta(meta_out,patch_bundle,label_simple,half_fraction_range,whale_min_fraction,nowhale_max_fraction) -> None: Write per-run patch outputs for main_create.py."""
    anns_patch = patch_bundle.get("anns_patch", patch_bundle.get("anns", []))
    anns_patch_rows = [ann_to_row_dict(a) for a in anns_patch] if isinstance(anns_patch, list) else []

    meta = {
        "patch_name": patch_bundle.get("patch_name", ""),
        "label_simple": label_simple,
        "top_left": list(patch_bundle.get("top_left", (None, None))),
        "fracs": patch_bundle.get("fracs", []),
        "offset_xy": list(patch_bundle.get("offset_xy", (None, None))),
        "anns_patch": anns_patch_rows,
        "label_thresholds": {
            "nowhale_max_fraction": float(nowhale_max_fraction),
            "whale_min_fraction": float(whale_min_fraction),
            "half_fraction_range": [float(half_fraction_range[0]), float(half_fraction_range[1])],
        },
    }

    out_path = Path(meta_out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    """main() -> int: Run one isolated pipeline."""
    args = parse_args()
    show_plot = bool(int(args.show_plot))

    dt = (
        datetime.fromisoformat(args.datetime_utc.replace("Z", "+00:00"))
        if args.datetime_utc
        else datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)
    )

    patch_rng = np.random.default_rng(int(args.patch_seed))

    half_range = (float(args.half_fraction_low), float(args.half_fraction_high))

    patch_bundle = generate_patch(
        mode_single=str(args.mode_single),
        mode_multiple_allow_partial=bool(int(args.mode_multiple_allow_partial)),
        window_size=int(args.window_size),
        img_file=str(args.img_file),
        rng=patch_rng,
        nowhale_max_fraction=float(args.nowhale_max_fraction),
        whale_min_fraction=float(args.whale_min_fraction),
        half_fraction_range=half_range,
        mask_alpha=int(args.mask_alpha),
        plot_patch=show_plot,
    )

    # Save nadir first (assigns patch_name inside save_patch and creates anns_patch)
    save_patch("nadir", patch_bundle)

    label_simple = classify_label(
        fracs=list(patch_bundle.get("fracs", [])) if isinstance(patch_bundle.get("fracs", []), list) else [],
        whale_min_fraction=float(args.whale_min_fraction),
        half_fraction_range=half_range,
        nowhale_max_fraction=float(args.nowhale_max_fraction),
    )

    # Write meta immediately after nadir save
    write_meta(
        args.meta_out,
        patch_bundle,
        label_simple=label_simple,
        half_fraction_range=half_range,
        whale_min_fraction=float(args.whale_min_fraction),
        nowhale_max_fraction=float(args.nowhale_max_fraction),
    )

    offnadir_bundle = translate_offnadir(
        patch_bundle,
        render_resolution=int(args.render_resolution),
        sat_lat=args.sat_lat, sat_lon=args.sat_lon, sat_alt=args.sat_alt,
        tgt_lat=args.tgt_lat, tgt_lon=args.tgt_lon, tgt_alt=args.tgt_alt,
        dem_seed=int(args.dem_seed),
        show_plot=show_plot,
        datetime_utc=dt,
    )
    save_patch("offnadir", offnadir_bundle)

    sunglint_bundle = add_sunglint(offnadir_bundle, show_plot=False)
    save_patch("sunglint", sunglint_bundle)

    cleanup()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        cleanup()
        print(f"[worker_run.py] ERROR: {e}", file=sys.stderr)
        raise
