#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image


def _iter_coco_image_paths(coco_path: Path, img_root: Path, locations: set[str] | None) -> List[Path]:
    """Return existing image file paths from COCO filtered by locations."""
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images", []) or []

    out: List[Path] = []

    for im in images:
        fn = (im.get("file_name") or "").replace("\\", "/").lstrip("./").lstrip("/")
        if not fn:
            continue

        if locations:
            parts = set(Path(fn).parts)
            if not any(loc in parts or fn.startswith(loc + "/") for loc in locations):
                continue

        p = (img_root / fn).resolve()
        if p.exists():
            out.append(p)

    return out


def _process_image(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute per-image RGB channel sums and squared sums."""
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return np.zeros(3), np.zeros(3), 0

    flat = arr.reshape(-1, 3)

    sum_c = flat.sum(axis=0)
    sumsq_c = (flat ** 2).sum(axis=0)
    pixels = flat.shape[0]

    return sum_c, sumsq_c, pixels


def compute_mean_std_rgb_fast(
    image_paths: List[Path],
    workers: int,
    progress_step_pct: int = 1
) -> Tuple[List[float], List[float], int, int]:
    """Compute exact RGB mean/std in [0,1] using multiprocessing."""

    total = len(image_paths)

    step = max(1, int(total * progress_step_pct / 100))
    next_print = step

    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)

    n_pixels = 0
    n_images = 0

    chunksize = max(1, total // (workers * 8))

    with Pool(workers) as pool:

        for idx, (s, ss, pix) in enumerate(
            pool.imap_unordered(_process_image, image_paths, chunksize=chunksize),
            start=1
        ):

            if pix == 0:
                continue

            sum_c += s
            sumsq_c += ss
            n_pixels += pix
            n_images += 1

            if idx >= next_print or idx == total:
                pct = int(round(idx / total * 100))
                print(f"[compute_dataset_mean_std] progress: {pct}% ({idx}/{total})")
                next_print += step

    if n_pixels == 0:
        raise RuntimeError("No pixels processed.")

    mean = sum_c / n_pixels
    var = sumsq_c / n_pixels - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    return mean.tolist(), std.tolist(), n_images, n_pixels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-channel RGB mean/std over a COCO dataset (multiprocessing)."
    )

    p.add_argument("--img_root", type=str, required=True)
    p.add_argument("--coco", type=str, required=True)
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--workers", type=int, default=cpu_count())
    p.add_argument("--progress_step_pct", type=int, default=1)
    p.add_argument(
        "--locations",
        type=str,
        default="",
        help="Comma-separated list of locations to include (e.g. Maui2015,Valdes2016)"
    )

    return p.parse_args()


def main() -> None:

    args = parse_args()

    img_root = Path(args.img_root).expanduser().resolve()
    coco_path = Path(args.coco).expanduser().resolve()

    if not coco_path.exists():
        raise SystemExit(f"[compute_dataset_mean_std] ERROR: coco not found: {coco_path}")

    if not img_root.exists():
        raise SystemExit(f"[compute_dataset_mean_std] ERROR: img_root not found: {img_root}")

    locations = {x.strip() for x in args.locations.split(",") if x.strip()} if args.locations else None

    print(f"Generate MEAN/STD for {locations}")

    paths = _iter_coco_image_paths(coco_path, img_root, locations)

    if not paths:
        raise SystemExit(
            "[compute_dataset_mean_std] ERROR: no image paths resolved."
        )

    print(f"[compute_dataset_mean_std] images found: {len(paths)}")
    print(f"[compute_dataset_mean_std] using {args.workers} workers")

    mean, std, n_images, n_pixels = compute_mean_std_rgb_fast(
        paths,
        workers=args.workers,
        progress_step_pct=max(1, int(args.progress_step_pct)),
    )

    out: Dict[str, object] = {
        "img_root": str(img_root),
        "coco": str(coco_path),
        "n_images_processed": int(n_images),
        "n_pixels_total": int(n_pixels),
        "mean_rgb_01": mean,
        "std_rgb_01": std,
        "note": "Exact RGB mean/std over all pixels using multiprocessing.",
    }

    print("[compute_dataset_mean_std] RESULT")
    print(json.dumps(out, indent=2))

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[compute_dataset_mean_std] wrote {out_path}")


if __name__ == "__main__":
    main()