# worker_run.py
import argparse
import sys
from datetime import datetime, timezone

import mitsuba as mi
mi.set_variant("llvm_ad_rgb")

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
    p.add_argument("--img_file", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dem_seed", type=int, required=True)
    p.add_argument("--show_plot", type=int, default=0)

    p.add_argument("--render_resolution", type=int, default=124)

    p.add_argument("--sat_lat", type=float, required=True)
    p.add_argument("--sat_lon", type=float, required=True)
    p.add_argument("--sat_alt", type=float, required=True)

    p.add_argument("--tgt_lat", type=float, required=True)
    p.add_argument("--tgt_lon", type=float, required=True)
    p.add_argument("--tgt_alt", type=float, required=True)

    p.add_argument("--datetime_utc", type=str, default="")
    return p.parse_args()


def main() -> int:
    """main() -> int: Run one isolated pipeline."""
    args = parse_args()
    show_plot = bool(int(args.show_plot))

    dt = (
        datetime.fromisoformat(args.datetime_utc.replace("Z", "+00:00"))
        if args.datetime_utc
        else datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)
    )

    rng = np.random.default_rng(int(args.seed))

    patch_bundle = generate_patch(
        mode_single="full",
        mode_multiple_allow_partial=False,
        window_size=64,
        img_file=args.img_file,
        rng=rng,
        nowhale_max_fraction=0.10,
        whale_min_fraction=0.99,
        half_fraction_range=(0.20, 0.80),
        mask_alpha=80,
        plot_patch=show_plot,
    )

    save_patch("nadir", patch_bundle)

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
