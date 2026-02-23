#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from train_utils import (
    read_json,
    write_json,
    write_text,
    resolve_path,
    run_and_tee,
    ensure_env_cuda_visible_devices,
    pick_checkpoint,
    cleanup_numbered_checkpoints,
)


def _require_coco_dict(obj: Any, path: str) -> dict[str, Any]:
    """_require_coco_dict(obj, path) -> dict[str,Any]: Validate COCO dict structure."""
    if not isinstance(obj, dict):
        raise SystemExit(
            f"ERROR: COCO annotation must be a dict, but {path} is {type(obj)}. "
            f"You may have passed predictions_coco.json (a list)."
        )
    if "images" not in obj or "annotations" not in obj:
        raise SystemExit(f"ERROR: {path} does not look like COCO (missing images/annotations).")
    return obj


def _location_from_file_name(file_name: str) -> str:
    """_location_from_file_name(file_name) -> str: Location = first path segment."""
    s = str(file_name or "").replace("\\", "/")
    if "/" in s:
        return s.split("/", 1)[0]
    return "unknown"


def _split_holdout_by_location(
    coco: dict[str, Any],
    val_frac: float,
    seed: int,
    min_val_per_location: int,
) -> tuple[set[int], set[int]]:
    """_split_holdout_by_location(coco, val_frac, seed, min_val_per_location) -> (train_ids, val_ids)."""
    rng = random.Random(int(seed))

    images = coco.get("images", []) or []
    if not isinstance(images, list) or len(images) == 0:
        raise SystemExit("ERROR: COCO has no images.")

    by_loc: dict[str, list[dict[str, Any]]] = {}
    for im in images:
        loc = _location_from_file_name(im.get("file_name", ""))
        by_loc.setdefault(loc, []).append(im)

    val_img_ids: set[int] = set()
    all_img_ids: list[int] = [int(im["id"]) for im in images if "id" in im]

    for loc, ims in by_loc.items():
        ids = [int(im["id"]) for im in ims if "id" in im]
        rng.shuffle(ids)
        take = min(len(ids), max(int(min_val_per_location), int(round(len(ids) * float(val_frac)))))
        for x in ids[:take]:
            val_img_ids.add(int(x))

    target_val = max(1, int(round(len(all_img_ids) * float(val_frac))))
    if len(val_img_ids) < target_val:
        remaining = [i for i in all_img_ids if i not in val_img_ids]
        rng.shuffle(remaining)
        for x in remaining:
            val_img_ids.add(int(x))
            if len(val_img_ids) >= target_val:
                break

    train_img_ids = set(all_img_ids) - set(val_img_ids)
    if len(train_img_ids) == 0:
        raise SystemExit("ERROR: training set became empty. Lower val_frac or min_val_per_location.")

    return train_img_ids, val_img_ids


def _filter_coco_by_image_ids(coco: dict[str, Any], keep_img_ids: set[int]) -> dict[str, Any]:
    """_filter_coco_by_image_ids(coco, keep_img_ids) -> dict[str,Any]: Filter COCO images/annotations by image_id."""
    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []

    keep_img_ids = set(int(x) for x in keep_img_ids)

    images2 = [im for im in images if int(im.get("id")) in keep_img_ids]
    anns2 = [a for a in anns if int(a.get("image_id")) in keep_img_ids]

    coco2 = dict(coco)
    coco2["images"] = images2
    coco2["annotations"] = anns2
    return coco2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--coco_trainval", required=True)
    ap.add_argument("--coco_test", default="")
    ap.add_argument("--img_root_test", default="")
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--seed", default="123")
    ap.add_argument("--val_frac", default="0.15")
    ap.add_argument("--min_val_per_location", default="1")
    ap.add_argument("--overwrite", default="0")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    base_config_abs = resolve_path(args.base_config)
    coco_trainval_abs = resolve_path(args.coco_trainval)
    coco_test_abs = resolve_path(args.coco_test) if str(args.coco_test).strip() else ""
    pretrained_abs = resolve_path(args.pretrained) if str(args.pretrained).strip() else ""

    img_root = Path(args.img_root).expanduser()
    img_root_test = Path(args.img_root_test).expanduser() if str(args.img_root_test).strip() else img_root

    ensure_env_cuda_visible_devices(args.gpus)

    if str(args.overwrite).strip() == "1" and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    coco = _require_coco_dict(read_json(coco_trainval_abs), coco_trainval_abs)

    train_ids, val_ids = _split_holdout_by_location(
        coco=coco,
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        min_val_per_location=int(args.min_val_per_location),
    )

    train_coco = _filter_coco_by_image_ids(coco, train_ids)
    val_coco = _filter_coco_by_image_ids(coco, val_ids)

    train_ann_path = out_dir / "splits" / "instances_train.json"
    val_ann_path = out_dir / "splits" / "instances_val.json"
    write_json(str(train_ann_path), train_coco)
    write_json(str(val_ann_path), val_coco)

    cfg_dir = out_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "train_config.yml"

    # Keep DEIM default solver behavior by not overriding solver params here.
    # Only override dataset paths.
    cfg_obj = {
        "__base__": [str(Path(base_config_abs).resolve())],
        "output_dir": str(out_dir),
        "coco_train": str(train_ann_path),
        "coco_val": str(val_ann_path),
        "img_root": str(img_root),
        "pretrained": (str(Path(pretrained_abs).resolve()) if pretrained_abs else None),
    }
    cfg_path.write_text(json.dumps(cfg_obj, indent=2), encoding="utf-8")

    cmd = [sys.executable, "train.py", "-c", str(cfg_path)]
    env = os.environ.copy()
    log_path = out_dir / "train_stdout.log"

    # Remove stale stage files if present
    for stale in ["best_stg1.pth", "best_stg2.pth", "last.pth"]:
        p = out_dir / stale
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    rc = run_and_tee(cmd, env=env, cwd=str(repo_root), log_path=str(log_path))
    if rc != 0:
        raise SystemExit(f"Final training failed: exit {rc}")

    ckpt = pick_checkpoint(out_dir)

    # Record which checkpoint variant was selected (stg1/stg2/last/other)
    variant = "other"
    n = ckpt.name
    if n == "best_stg1.pth":
        variant = "stg1"
    elif n == "best_stg2.pth":
        variant = "stg2"
    elif n == "last.pth":
        variant = "last"
    elif n.lower().startswith("best"):
        variant = "best"
    write_text(out_dir / "selected_final_model.txt", variant + "\n")

    final_meta: dict[str, Any] = {
        "out_dir": str(out_dir),
        "final_checkpoint": str(ckpt.resolve()),
        "train_ann": str(train_ann_path.resolve()),
        "val_ann": str(val_ann_path.resolve()),
        "coco_test": (str(Path(coco_test_abs).resolve()) if coco_test_abs else None),
        "img_root": str(img_root),
        "img_root_test": str(img_root_test),
        "base_config": str(Path(base_config_abs).resolve()),
        "pretrained": (str(Path(pretrained_abs).resolve()) if pretrained_abs else None),
        "val_frac": float(args.val_frac),
        "seed": int(args.seed),
        "min_val_per_location": int(args.min_val_per_location),
    }
    write_json(str(out_dir / "final_meta.json"), final_meta)

    cleanup_numbered_checkpoints(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())