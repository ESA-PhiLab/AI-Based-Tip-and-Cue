# worker_create_one.py
import argparse
import sys

import mitsuba as mi

MI_VARIANT = "llvm_ad_rgb"
mi.set_variant(MI_VARIANT)

from create_patch import generate_patch
from save_patch import save_patch
from translate_patch import translate_offnadir, add_sunglint

import numpy as np


def parse_args() -> argparse.Namespace:
    """parse_args() -> argparse.Namespace: Read CLI args for one patch generation run."""
    p = argparse.ArgumentParser()
    p.add_argument("--img_file", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--show_plot", action="store_true")
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--nowhale_max_fraction", type=float, default=0.10)
    p.add_argument("--whale_min_fraction", type=float, default=0.99)
    p.add_argument("--half_lo", type=float, default=0.20)
    p.add_argument("--half_hi", type=float, default=0.80)
    p.add_argument("--mask_alpha", type=int, default=80)

    p.add_argument("--render_resolution", type=int, default=124)
    p.add_argument("--sat_lat", type=float, default=58.0)
    p.add_argument("--sat_lon", type=float, default=-5.0)
    p.add_argument("--sat_alt", type=float, default=617000.0)
    p.add_argument("--tgt_lat", type=float, default=53.0)
    p.add_argument("--tgt_lon", type=float, default=0.0)
    p.add_argument("--tgt_alt", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    """main() -> int: Generate+save nadir, offnadir, sunglint for exactly one patch."""
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    patch_bundle = generate_patch(
        mode_single="full",
        mode_multiple_allow_partial=False,
        window_size=args.window_size,
        img_file=args.img_file,
        rng=rng,
        nowhale_max_fraction=args.nowhale_max_fraction,
        whale_min_fraction=args.whale_min_fraction,
        half_fraction_range=(args.half_lo, args.half_hi),
        mask_alpha=args.mask_alpha,
        plot_patch=bool(args.show_plot),
    )

    save_patch("nadir", patch_bundle)

    offnadir_bundle = translate_offnadir(
        patch_bundle,
        render_resolution=args.render_resolution,
        sat_lat=args.sat_lat, sat_lon=args.sat_lon, sat_alt=args.sat_alt,
        tgt_lat=args.tgt_lat, tgt_lon=args.tgt_lon, tgt_alt=args.tgt_alt,
        show_plot=bool(args.show_plot),
    )
    save_patch("offnadir", offnadir_bundle)

    sunglint_bundle = add_sunglint(offnadir_bundle, show_plot=False)
    save_patch("sunglint", sunglint_bundle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
