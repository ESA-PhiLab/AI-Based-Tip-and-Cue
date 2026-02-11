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
from translate_patch import translate_image, mirror_rotate_patchlocal_bundle


def cleanup() -> None:
    """cleanup() -> None: Close figures to avoid hanging/leaks."""
    try:
        plt.close("all")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    """parse_args() -> argparse.Namespace: Parse CLI args."""
    p = argparse.ArgumentParser()

    p.add_argument("--stage", type=str, choices=["nadir", "offnadir"], required=True)

    p.add_argument("--img_file", type=str, required=True)
    p.add_argument("--patch_seed", type=int, required=True)
    p.add_argument("--dem_seed", type=int, required=True)
    p.add_argument("--show_plot", type=int, default=0)

    p.add_argument("--sat_lat", type=float, required=True)
    p.add_argument("--sat_lon", type=float, required=True)
    p.add_argument("--sat_alt", type=float, required=True)

    p.add_argument("--tgt_lat", type=float, required=True)
    p.add_argument("--tgt_lon", type=float, required=True)
    p.add_argument("--tgt_alt", type=float, required=True)

    p.add_argument("--datetime_utc", type=str, default="")

    p.add_argument("--mode_single", type=str, default="full")
    p.add_argument("--mode_multiple_allow_partial", type=int, default=0)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--rotation_angle_deg", type=float, default=0.0)
    p.add_argument("--mirror_bool", type=int, default=0)

    p.add_argument("--nowhale_max_fraction", type=float, default=0.10)
    p.add_argument("--whale_min_fraction", type=float, default=0.99)
    p.add_argument("--half_fraction_low", type=float, default=0.20)
    p.add_argument("--half_fraction_high", type=float, default=0.80)
    p.add_argument("--mask_alpha", type=int, default=80)

    p.add_argument("--sensor_json", type=str, required=True)
    p.add_argument("--wave_json", type=str, required=True)

    p.add_argument("--meta_out", type=str, required=True)
    p.add_argument("--patch_name", type=str, default="")  # only used for offnadir stage

    return p.parse_args()


def array_stats(x: object) -> dict:
    """array_stats(x) -> dict: Return min/max/mean over finite, non-black pixels (black=0 or [0,0,0]); NaN if empty."""
    a = np.asarray(x)
    if a.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}

    a = a.astype(np.float64, copy=False)

    finite = np.isfinite(a)

    # Non-black mask: exclude pixels that are exactly 0 (or [0,0,0] for RGB-like arrays)
    if a.ndim >= 3:
        # treat last axis as channels
        nonblack = np.any(a != 0.0, axis=-1)
        finite_pix = np.all(finite, axis=-1)   # pixel is finite if all channels finite
        keep = nonblack & finite_pix
        v = a[keep]  # flattened channels of kept pixels
    else:
        keep = finite & (a != 0.0)
        v = a[keep]

    if v.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}

    return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


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

def anns_to_rows(anns: object) -> list[dict]:
    """anns_to_rows(anns) -> list[dict]: Convert annotation list into row dicts."""
    if not isinstance(anns, list):
        return []
    out = []
    for a in anns:
        if isinstance(a, dict):
            out.append(ann_to_row_dict(a))
    return out



def write_meta(meta_out: str,
               patch_bundle: dict,
               label_simple: str,
               half_fraction_range: tuple[float, float],
               whale_min_fraction: float,
               nowhale_max_fraction: float,
               rotation_angle_deg: float,
               mirror_bool: bool) -> None:
    """write_meta(meta_out,patch_bundle,label_simple,half_fraction_range,whale_min_fraction,nowhale_max_fraction,rotation_angle_deg) -> None: Write per-run patch outputs."""
    anns_patch = patch_bundle.get("anns_patch", patch_bundle.get("anns", []))
    anns_patch_rows = [ann_to_row_dict(a) for a in anns_patch] if isinstance(anns_patch, list) else []

    meta = {
        "patch_name": patch_bundle.get("patch_name", ""),
        "label_simple": label_simple,
        "category_id": patch_bundle.get("category_id", None),
        "top_left": list(patch_bundle.get("top_left", (None, None))),
        "fracs": patch_bundle.get("fracs", []),
        "offset_xy": list(patch_bundle.get("offset_xy", (None, None))),
        "rotation_angle_deg": float(rotation_angle_deg),
        "mirror_bool": bool(mirror_bool),
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

def update_meta_file(meta_out: str, updates: dict) -> None:
    """update_meta_file(meta_out,updates) -> None: Load meta_out JSON, update keys, write back."""
    p = Path(meta_out)
    if p.is_file():
        meta = json.loads(p.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.update(updates)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_patch_bundle_from_patchraw(img_file: str, patch_name: str) -> dict:
    """load_patch_bundle_from_patchraw(img_file,patch_name) -> dict: Build patch_bundle from patch_raw COCO json."""
    main_path = Path(__file__).resolve().parents[2]
    coco_path = main_path / "dataset" / "create_dataset" / "patch_raw_255" / "final_annotations.json"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing patch_raw COCO json: {coco_path}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    subdir = Path(img_file).parent.as_posix()
    ext = Path(img_file).suffix
    subdir = Path(img_file).parent.as_posix()
    ext = Path(img_file).suffix

    def matches(im: dict) -> bool:
        fn = str(im.get("file_name", ""))
        p = Path(fn)
        if p.suffix != ext:
            return False
        if subdir and p.parent.as_posix() != subdir:
            return False
        return p.stem == patch_name or p.stem.startswith(patch_name + "_")

    img_rec = next((im for im in images if isinstance(im, dict) and matches(im)), None)
    if img_rec is None:
        raise FileNotFoundError(f"patch_raw image record not found for patch_name={patch_name} (subdir={subdir}, ext={ext})")


    image_id = img_rec.get("id")
    anns_patch = [a for a in anns if a.get("image_id") == image_id]

    label_simple = img_rec.get("label_simple", None)

    return {
        "img_file": img_file,
        "patch_name": patch_name,
        "label_simple": label_simple,  # add this
        "img_info": dict(img_rec),
        "anns_patch": anns_patch,
    }


def main() -> int:
    """main() -> int: Run one isolated pipeline stage (nadir or offnadir)."""
    args = parse_args()
    show_plot = bool(int(args.show_plot))
    mirror_bool = bool(int(args.mirror_bool))

    sensor_characteristics = json.loads(args.sensor_json)
    wave_properties = json.loads(args.wave_json)

    dt = (
        datetime.fromisoformat(args.datetime_utc.replace("Z", "+00:00"))
        if args.datetime_utc
        else datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)
    )

    half_range = (float(args.half_fraction_low), float(args.half_fraction_high))

    if args.stage == "nadir":

        print("\nGenerate patch")
        patch_rng = np.random.default_rng(int(args.patch_seed))

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

        label_simple = classify_label(
            fracs=list(patch_bundle.get("fracs", [])) if isinstance(patch_bundle.get("fracs", []), list) else [],
            whale_min_fraction=float(args.whale_min_fraction),
            half_fraction_range=half_range,
            nowhale_max_fraction=float(args.nowhale_max_fraction),
        )
        patch_bundle["label_simple"] = label_simple

        save_patch("patch_raw_255", patch_bundle)

        b_rot = mirror_rotate_patchlocal_bundle(patch_bundle, args.rotation_angle_deg, mirror_bool=mirror_bool)
        save_patch("patch_raw_rot_255", b_rot)

        write_meta(
            args.meta_out,
            patch_bundle,
            label_simple=label_simple,
            half_fraction_range=half_range,
            whale_min_fraction=float(args.whale_min_fraction),
            nowhale_max_fraction=float(args.nowhale_max_fraction),
            rotation_angle_deg=float(args.rotation_angle_deg),
            mirror_bool=mirror_bool
        )


        patch_bundle["label_simple"] = label_simple

        write_meta(
            args.meta_out,
            patch_bundle,
            label_simple=label_simple,
            half_fraction_range=half_range,
            whale_min_fraction=float(args.whale_min_fraction),
            nowhale_max_fraction=float(args.nowhale_max_fraction),
            rotation_angle_deg=float(args.rotation_angle_deg),
            mirror_bool=mirror_bool
        )

        print("\nGenerate nadir image")
        nadir_bundle = translate_image(
            patch_bundle,
            sensor_characteristics=sensor_characteristics,
            wave_properties=wave_properties,
            sat_lat=args.sat_lat, sat_lon=args.sat_lon, sat_alt=args.sat_alt,
            tgt_lat=args.tgt_lat, tgt_lon=args.tgt_lon, tgt_alt=args.tgt_alt,
            dem_seed=int(args.dem_seed),
            show_plot=show_plot,
            datetime_utc=dt,
            generate_nadir=True,
            rotation_angle_deg=args.rotation_angle_deg,
            mirror_bool=mirror_bool
        )

        update_meta_file(args.meta_out, {
            "rad_stats_nadir": array_stats(nadir_bundle.get("radiance", np.array([]))),
            "refl_stats_nadir": array_stats(nadir_bundle.get("reflectance", np.array([]))),
        })

        update_meta_file(args.meta_out, {
            "anns_nadir": anns_to_rows(nadir_bundle.get("anns_patch", [])),
        })


        # save texture
        b_tex = dict(nadir_bundle)
        b_tex["patch"] = nadir_bundle["texture_u8"]
        save_patch("texture_nadir_255", b_tex)

        # save radiance (float .npy)
        b_rad = dict(nadir_bundle)
        b_rad["patch"] = nadir_bundle["radiance"]
        save_patch("radiance_nadir_npy", b_rad)

        b_rad["patch"] = nadir_bundle["radiance_u8"]
        save_patch("radiance_nadir_255", b_rad)

        # save reflectance (float .npy)
        b_ref = dict(nadir_bundle)
        b_ref["patch"] = nadir_bundle["reflectance"]
        save_patch("reflection_nadir_npy", b_ref)

        b_ref["patch"] = nadir_bundle["reflectance_u8"]
        save_patch("reflection_nadir_255", b_ref)

        cleanup()
        return 0

    # OFFNADIR stage
    if not args.patch_name:
        raise ValueError("--patch_name is required for stage=offnadir")

    patch_bundle = load_patch_bundle_from_patchraw(img_file=args.img_file, patch_name=args.patch_name)

    print("\nGenerate off-nadir image")
    off_bundle = translate_image(
        patch_bundle,
        sensor_characteristics=sensor_characteristics,
        wave_properties=wave_properties,
        sat_lat=args.sat_lat, sat_lon=args.sat_lon, sat_alt=args.sat_alt,
        tgt_lat=args.tgt_lat, tgt_lon=args.tgt_lon, tgt_alt=args.tgt_alt,
        dem_seed=int(args.dem_seed),
        show_plot=show_plot,
        datetime_utc=dt,
        generate_nadir=False,
        rotation_angle_deg=args.rotation_angle_deg,
        mirror_bool=mirror_bool
    )

    update_meta_file(args.meta_out, {
        "rad_stats_offnadir": array_stats(off_bundle.get("radiance", np.array([]))),
        "refl_stats_offnadir": array_stats(off_bundle.get("reflectance", np.array([]))),
    })

    update_meta_file(args.meta_out, {
        "offnadir_deg": float(off_bundle.get("offnadir_deg", 0.0)) if off_bundle.get("offnadir_deg", None) is not None else None,
        "rotation_angle_deg": float(args.rotation_angle_deg),
        "anns_offnadir": anns_to_rows(off_bundle.get("anns_patch", [])),
    })

    b_tex = dict(off_bundle)
    b_tex["offnadir_deg"] = off_bundle.get("offnadir_deg", None)
    b_tex["patch"] = off_bundle["texture_u8"]
    save_patch("texture_offnadir_255", b_tex)

    b_rad = dict(off_bundle)
    b_rad["offnadir_deg"] = off_bundle.get("offnadir_deg", None)
    b_rad["patch"] = off_bundle["radiance"]
    save_patch("radiance_offnadir_npy", b_rad)

    b_rad["patch"] = off_bundle["radiance_u8"]
    save_patch("radiance_offnadir_255", b_rad)

    b_ref = dict(off_bundle)
    b_ref["offnadir_deg"] = off_bundle.get("offnadir_deg", None)
    b_ref["patch"] = off_bundle["reflectance"]
    save_patch("reflection_offnadir_npy", b_ref)

    b_ref = dict(off_bundle)
    b_ref["patch"] = off_bundle["reflectance_u8"]
    save_patch("reflection_offnadir_255", b_ref)

    cleanup()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        cleanup()
        print(f"[worker_run.py] ERROR: {e}", file=sys.stderr)
        raise
