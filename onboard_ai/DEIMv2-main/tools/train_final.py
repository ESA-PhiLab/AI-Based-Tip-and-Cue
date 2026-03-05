#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any


import shutil

from compute_dataset_mean_std import _iter_coco_image_paths, compute_mean_std_rgb_fast

from train_utils import (
    read_json,
    write_json,
    write_text,
    resolve_path,
    run_and_tee,
    pick_checkpoint,
    cleanup_numbered_checkpoints, export_tb_plots
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
    coco: dict[str, Any], val_frac: float, seed: int, min_val_per_location: int
) -> tuple[set[int], set[int]]:
    """_split_holdout_by_location(coco, val_frac, seed, min_val_per_location) -> (train_ids, val_ids): Per-location stratified split (whale vs empty)."""
    rng = random.Random(int(seed))

    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    if not isinstance(images, list) or len(images) == 0:
        raise SystemExit("ERROR: COCO has no images.")

    whale_img_ids: set[int] = set()
    for a in anns:
        try:
            whale_img_ids.add(int(a.get("image_id")))
        except Exception:
            continue

    by_loc: dict[str, list[dict[str, Any]]] = {}
    all_img_ids: list[int] = []
    for im in images:
        if "id" not in im:
            continue
        iid = int(im["id"])
        all_img_ids.append(iid)
        loc = _location_from_file_name(im.get("file_name", ""))
        by_loc.setdefault(loc, []).append(im)

    val_img_ids: set[int] = set()

    for _, ims in by_loc.items():
        whale_ids: list[int] = []
        empty_ids: list[int] = []
        for im in ims:
            iid = int(im["id"])
            if iid in whale_img_ids:
                whale_ids.append(iid)
            else:
                empty_ids.append(iid)

        rng.shuffle(whale_ids)
        rng.shuffle(empty_ids)

        def _take_n(n_total: int) -> int:
            if n_total <= 0:
                return 0
            return min(n_total, max(int(min_val_per_location), int(round(n_total * float(val_frac)))))

        take_whale = _take_n(len(whale_ids)) if whale_ids else 0
        take_empty = _take_n(len(empty_ids)) if empty_ids else 0

        for x in whale_ids[:take_whale]:
            val_img_ids.add(int(x))
        for x in empty_ids[:take_empty]:
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


def write_ann_override_yml(path: Path, img_root: Path, train_ann: str, val_ann: str) -> None:
    """write_ann_override_yml(path, img_root, train_ann, val_ann) -> None: Write dataset override yml for DEIM."""
    yml = "\n".join(
        [
            "val_dataloader:",
            "  dataset:",
            f"    img_folder: {img_root}",
            f"    ann_file: {val_ann}",
            "",
            "train_dataloader:",
            "  dataset:",
            f"    img_folder: {img_root}",
            f"    ann_file: {train_ann}",
            "",
        ]
    )
    write_text(str(path), yml)


def write_include_config_yml(path: Path, base_config: str, override_yml: str, output_dir: str) -> None:
    """write_include_config_yml(path, base_config, override_yml, output_dir) -> None: Include base+override and set output_dir."""
    yml = "\n".join(
        [
            "__include__: [",
            f'  "{base_config}",',
            f'  "{override_yml}",',
            "]",
            "",
            f"output_dir: {output_dir}",
            "",
        ]
    )
    write_text(str(path), yml)


def _fmt_float_list(xs: list[float]) -> str:
    """_fmt_float_list(xs) -> str: Format float list for YAML inline arrays."""
    return "[" + ", ".join(f"{float(x):.16g}" for x in xs) + "]"


def _patch_normalize_lines(yml_text: str, mean: list[float], std: list[float]) -> tuple[str, int]:
    """_patch_normalize_lines(yml_text, mean, std) -> (new_text, n): Replace all inline Normalize mean/std blocks."""
    mean_s = _fmt_float_list(mean)
    std_s = _fmt_float_list(std)

    pat = re.compile(
        r"(\{type:\s*Normalize\s*,\s*mean:\s*)\[[^\]]*\](\s*,\s*std:\s*)\[[^\]]*\](\s*\})"
    )

    def _repl(m: re.Match) -> str:
        return f"{m.group(1)}{mean_s}{m.group(2)}{std_s}{m.group(3)}"

    new_text, n = pat.subn(_repl, yml_text)
    return new_text, int(n)


def _compute_mean_std_from_coco(img_root: Path, coco_path: Path, workers: int, progress_step_pct: int) -> tuple[list[float], list[float], int, int]:
    """_compute_mean_std_from_coco(img_root, coco_path, workers, progress_step_pct) -> (mean, std, n_images, n_pixels): Exact RGB mean/std."""
    paths = _iter_coco_image_paths(coco_path=coco_path, img_root=img_root, locations=None)
    if not paths:
        raise RuntimeError(f"No images resolved from COCO: {coco_path}")

    mean, std, n_images, n_pixels = compute_mean_std_rgb_fast(
        image_paths=paths,
        workers=int(workers),
        progress_step_pct=max(1, int(progress_step_pct)),
    )
    return mean, std, int(n_images), int(n_pixels)


def _write_base_config_with_norm(base_config_in: Path, base_config_out: Path, mean: list[float], std: list[float]) -> int:
    """_write_base_config_with_norm(base_config_in, base_config_out, mean, std) -> int: Patch Normalize mean/std in base config."""
    txt = base_config_in.read_text(encoding="utf-8")
    new_txt, n = _patch_normalize_lines(txt, mean=mean, std=std)
    if n < 1:
        raise RuntimeError(
            f"Did not find any inline Normalize blocks to patch in: {base_config_in}\n"
            "Expected lines like: - {type: Normalize, mean: [...], std: [...]}."
        )
    base_config_out.parent.mkdir(parents=True, exist_ok=True)
    base_config_out.write_text(new_txt, encoding="utf-8")
    return int(n)


def main() -> int:
    """main() -> int: Split train/val, write configs, launch torchrun training for final model."""
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

    # Final model RGB normalization computed on TRAIN split (after the train/val split)
    ap.add_argument("--final_norm_stats", default="1")  # 1|0
    ap.add_argument("--stats_workers", default=str(cpu_count()))
    ap.add_argument("--stats_progress_step_pct", default="5")
    ap.add_argument("--overwrite", default="0")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--master_port", default="29500")
    ap.add_argument("--eval_name", default="eval_data")
    ap.add_argument("--eval_gpus", default="")
    ap.add_argument("--eval_nproc", default="1")
    ap.add_argument("--eval_master_port", default="29501")
    ap.add_argument("--overwrite_eval", default="0")
    ap.add_argument("--val_test_final", default="0")
    ap.add_argument("--label_offset", default="0")
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    base_config_abs = resolve_path(repo_root, args.base_config)

    # Fail hard unless config lives inside results/ (run-frozen config)
    bc_norm = str(Path(base_config_abs).resolve()).replace("\\", "/")
    if "/results/" not in bc_norm:
        raise SystemExit(
            f"[train_final] ERROR: base_config must be the run-local frozen config under results/, got:\n{bc_norm}"
        )


    coco_trainval_abs = resolve_path(repo_root, args.coco_trainval)
    coco_test_abs = resolve_path(repo_root, args.coco_test) if str(args.coco_test).strip() else ""
    pretrained_abs = resolve_path(repo_root, args.pretrained) if str(args.pretrained).strip() else ""

    img_root = Path(args.img_root).expanduser()
    img_root_test = Path(args.img_root_test).expanduser() if str(args.img_root_test).strip() else img_root

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

    override_yml = cfg_dir / "split_override.yml"
    cfg_path = cfg_dir / "train_config.yml"

    write_ann_override_yml(override_yml, img_root, str(train_ann_path), str(val_ann_path))

    # ----- Final mean/std computed on TRAIN split (instances_train.json) -----
    base_cfg_for_final = Path(base_config_abs).resolve()
    stats_json_path = cfg_dir / "dataset_rgb_mean_std_train.json"
    base_cfg_patched = cfg_dir / "base_config_with_train_norm.yml"

    final_mean: list[float] | None = None
    final_std: list[float] | None = None
    n_images = 0
    n_pixels = 0
    n_norm_patched = 0

    if str(getattr(args, "final_norm_stats", "1")).strip() == "1":
        mean, std, n_images, n_pixels = _compute_mean_std_from_coco(
            img_root=img_root,
            coco_path=train_ann_path,
            workers=int(getattr(args, "stats_workers", cpu_count())),
            progress_step_pct=int(getattr(args, "stats_progress_step_pct", 5)),
        )
        final_mean, final_std = mean, std
        n_norm_patched = _write_base_config_with_norm(
            base_config_in=base_cfg_for_final,
            base_config_out=base_cfg_patched,
            mean=mean,
            std=std,
        )

        write_json(
            str(stats_json_path),
            {
                "train_ann": str(train_ann_path),
                "img_root": str(img_root),
                "n_images_processed": int(n_images),
                "n_pixels_total": int(n_pixels),
                "mean_rgb_01": mean,
                "std_rgb_01": std,
                "patched_normalize_blocks": int(n_norm_patched),
                "note": "Exact RGB mean/std over FINAL TRAIN split (instances_train.json). Applied to all inline Normalize blocks (train+val).",
            },
        )
        base_cfg_for_include = base_cfg_patched
    else:
        base_cfg_for_include = base_cfg_for_final

    write_include_config_yml(cfg_path, str(base_cfg_for_include), str(override_yml), str(out_dir))

    # -----------------------
    # FIX: propagate GPU selection into the env used for torchrun
    # -----------------------
    env = os.environ.copy()
    if not str(env.get("CUDA_VISIBLE_DEVICES", "")).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()

    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(int(args.master_port))

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(int(args.nproc)),
        "--master_port",
        str(int(args.master_port)),
        "train.py",
        "-c",
        str(cfg_path),
    ]

    if pretrained_abs:
        cmd += ["-t", str(Path(pretrained_abs).resolve())]
    if args.use_amp:
        cmd.append("--use-amp")

    log_path = out_dir / "train_stdout.log"

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

    export_tb_plots(repo_root=repo_root, run_dir=out_dir, out_subdir="tb_exports")

    ckpt = pick_checkpoint(out_dir)

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
        "base_config_original": str(Path(base_config_abs).resolve()),
        "base_config_with_train_norm": (str(base_cfg_patched.resolve()) if base_cfg_patched.exists() else None),
        "train_norm_stats_json": (str(stats_json_path.resolve()) if stats_json_path.exists() else None),
        "train_mean_rgb_01": final_mean,
        "train_std_rgb_01": final_std,
    }
    write_json(str(out_dir / "final_meta.json"), final_meta)

    cleanup_numbered_checkpoints(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())