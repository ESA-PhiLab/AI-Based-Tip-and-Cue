#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import eval_utils
from compute_dataset_mean_std import _iter_coco_image_paths, compute_mean_std_rgb_fast
from train_utils import (
    read_json,
    write_json,
    write_text,
    resolve_path,
    run_and_tee,
    ensure_env_cuda_visible_devices,
    pick_checkpoint,
    cleanup_numbered_checkpoints, export_tb_plots
)

@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_locations: list[str]
    val_locations: list[str]

def export_tb_per_fold(repo_root: Path, fold_dir: Path) -> None:
    """export_tb_per_fold(repo_root, fold_dir) -> None: Export TB loss plots for this fold (best-effort)."""
    summary_dir = fold_dir / "summary"
    logdir = summary_dir if summary_dir.exists() else fold_dir
    outdir = fold_dir / "tb_exports"
    try:
        cmd = [sys.executable, "tools/export_tb_curves.py", "--logdir", str(logdir), "--outdir", str(outdir)]
        rc = subprocess.call(cmd, cwd=str(repo_root))
        if rc != 0:
            print(f"[tb_export] WARNING: export_tb_curves.py exited with {rc} for {fold_dir}", flush=True)
    except Exception as e:
        print(f"[tb_export] WARNING: failed exporting TB plots for {fold_dir}: {e}", flush=True)

def _run_eval_one(repo_root: Path, base_config: Path, checkpoint: Path, img_root: Path, ann: Path, out_dir: Path, gpus: str, nproc: str, master_port: int, overwrite: str, label_offset: str) -> int:
    """_run_eval_one(...) -> int: Run tools/eval_one_deimv2.py for a single split and return exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/eval_one_deimv2.py",
        "--base_config",
        str(base_config),
        "--checkpoint",
        str(checkpoint),
        "--img_root",
        str(img_root),
        "--ann",
        str(ann),
        "--out_dir",
        str(out_dir),
        "--gpus",
        str(gpus),
        "--nproc",
        str(nproc),
        "--master_port",
        str(int(master_port)),
        "--overwrite",
        str(overwrite),
        "--dump_predictions",
        "1",
        "--label_offset",
        str(label_offset),
    ]
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    if str(gpus).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpus).strip()
    return subprocess.call(cmd, cwd=str(repo_root), env=env)


def _has_eval_metrics(fold_dir: Path) -> bool:
    """_has_eval_metrics(fold_dir) -> bool: True if eval produced metrics.json."""
    if (fold_dir / "metrics" / "validation" / "metrics.json").exists():
        return True
    if (fold_dir / "metrics" / "test" / "metrics.json").exists():
        return True
    if list(fold_dir.glob("eval_val/**/metrics.json")):
        return True
    if list(fold_dir.glob("eval_test/**/metrics.json")):
        return True
    return False


def _find_resume_ckpt(fold_dir: Path) -> Path | None:
    """_find_resume_ckpt(fold_dir) -> Path|None: Select checkpoint for resume."""
    ckpt_dir = fold_dir / "checkpoints"

    candidates: list[Path] = []
    candidates += list(fold_dir.glob("*.pth"))
    if ckpt_dir.exists():
        candidates += list(ckpt_dir.glob("*.pth"))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    by_name = {p.name: p for p in candidates}

    for key in ["last.pth", "best_stg2.pth", "best_stg2_local.pth", "best_stg1.pth", "best.pth"]:
        if key in by_name:
            return by_name[key]

    numbered = [p for p in candidates if p.name.startswith("checkpoint") and p.name.endswith(".pth")]
    if numbered:
        def _step(p: Path) -> int:
            digits = "".join(ch for ch in p.stem if ch.isdigit())
            return int(digits) if digits else -1
        numbered.sort(key=_step)
        return numbered[-1]

    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _fold_is_done(fold_dir: Path, eval_after_each_fold: str) -> bool:
    """_fold_is_done(fold_dir, eval_after_each_fold) -> bool: True if fold completed."""
    if (fold_dir / "DONE").exists():
        return True

    if not (fold_dir / "fold_meta.json").exists():
        return False

    if _find_resume_ckpt(fold_dir) is None:
        return False

    if str(eval_after_each_fold).strip() == "1":
        return _has_eval_metrics(fold_dir)

    return True

def make_random_folds(locations: list[str], k: int, seed: int, val_size: int) -> list[FoldSpec]:
    """make_random_folds(locations, k, seed, val_size) -> list[FoldSpec]: Balanced unique folds (greedy) so each location appears ~equally often in val."""
    rng = random.Random(int(seed))
    locs = list(dict.fromkeys([str(x) for x in locations if str(x).strip() != ""]))

    val_size = int(val_size)
    k = int(k)

    if val_size < 1:
        raise SystemExit(f"ERROR: --val_size must be >= 1, got {val_size}.")
    if len(locs) <= val_size:
        raise SystemExit(f"Need at least 1 training location: got {len(locs)} for val_size={val_size}")

    all_vals = list(combinations(locs, val_size))
    if k > len(all_vals):
        raise SystemExit(
            f"Requested k={k} folds, but only {len(all_vals)} unique val combinations exist "
            f"for n={len(locs)} locations and val_size={val_size}."
        )

    counts: dict[str, int] = {l: 0 for l in locs}
    remaining = list(all_vals)
    chosen: list[tuple[str, ...]] = []

    def score_after_add(comb: tuple[str, ...]) -> tuple[int, int, float]:
        tmp = counts.copy()
        for l in comb:
            tmp[l] += 1
        vals = list(tmp.values())
        spread = max(vals) - min(vals)
        maxv = max(vals)
        mean = sum(vals) / float(len(vals))
        var = sum((v - mean) ** 2 for v in vals) / float(len(vals))
        return spread, maxv, var

    for _ in range(k):
        best_score: tuple[int, int, float] | None = None
        best: list[tuple[str, ...]] = []

        # Greedy selection: minimize imbalance (spread), then max count, then variance
        for comb in remaining:
            sc = score_after_add(comb)
            if best_score is None or sc < best_score:
                best_score = sc
                best = [comb]
            elif sc == best_score:
                best.append(comb)

        pick = rng.choice(best)
        chosen.append(pick)
        remaining.remove(pick)
        for l in pick:
            counts[l] += 1

    folds: list[FoldSpec] = []
    for fi, comb in enumerate(chosen):
        val = list(comb)
        train = [x for x in locs if x not in val]
        folds.append(FoldSpec(fold_id=fi, train_locations=train, val_locations=val))

    # ---- Debug report: validation usage per location ----
    print("\nBalanced CV validation usage:")
    for loc in sorted(counts):
        print(f"  {loc}: {counts[loc]}")

    total_val_slots = k * val_size
    expected = total_val_slots / float(len(locs))
    print(f"\nTotal validation slots: {total_val_slots}")
    print(f"Locations: {len(locs)}")
    print(f"Expected appearances per location (ideal): {expected:.2f}")
    print(f"Min count: {min(counts.values())}, Max count: {max(counts.values())}\n")

    return folds


def make_all_combinations_folds(locations: list[str], val_size: int) -> list[FoldSpec]:
    """make_all_combinations_folds(locations, val_size) -> list[FoldSpec]: All combinations for val set."""
    locs = list(dict.fromkeys([str(x) for x in locations if str(x).strip() != ""]))
    if len(locs) <= int(val_size):
        raise SystemExit(f"Need at least 1 training location: got {len(locs)} for val_size={val_size}")

    folds: list[FoldSpec] = []
    for fi, comb in enumerate(combinations(locs, int(val_size))):
        val = list(comb)
        train = [x for x in locs if x not in val]
        folds.append(FoldSpec(fold_id=fi, train_locations=train, val_locations=val))
    return folds


def extract_locations_from_csv_arg(s: str) -> list[str]:
    """extract_locations_from_csv_arg(s) -> list[str]: Split comma-separated list."""
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def _image_matches_location(file_name: str, loc: str) -> bool:
    """_image_matches_location(file_name, loc) -> bool: True if loc matches a path segment."""
    fn = str(file_name or "").replace("\\", "/").lstrip("./").lstrip("/")
    parts = set(Path(fn).parts)
    if loc in parts:
        return True
    if f"/{loc}/" in fn:
        return True
    if fn.startswith(loc + "/"):
        return True
    return False


def _require_coco_dict(obj: Any, path: str) -> dict[str, Any]:
    """_require_coco_dict(obj, path) -> dict[str,Any]: Validate COCO dict structure (not a list)."""
    if not isinstance(obj, dict):
        raise SystemExit(
            f"ERROR: COCO annotation must be a dict, but {path} is {type(obj)}. "
            f"This usually happens if you accidentally passed predictions_coco.json (a list) as --coco_val."
        )
    if "images" not in obj or "annotations" not in obj:
        raise SystemExit(
            f"ERROR: {path} does not look like COCO (missing 'images'/'annotations'). Keys={list(obj.keys())[:20]}"
        )
    return obj


def _filter_coco_by_locations(coco: dict[str, Any], keep_locations: list[str]) -> dict[str, Any]:
    """_filter_coco_by_locations(coco, keep_locations) -> dict[str,Any]: Filter images/annotations by locations."""
    keep_locs = [str(x) for x in keep_locations if str(x).strip() != ""]
    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []

    keep_img_ids: set[int] = set()
    keep_images: list[dict[str, Any]] = []
    for im in images:
        fn = im.get("file_name", "")
        if any(_image_matches_location(fn, loc) for loc in keep_locs):
            iid = int(im.get("id"))
            keep_img_ids.add(iid)
            keep_images.append(im)

    keep_anns: list[dict[str, Any]] = []
    for a in anns:
        iid = int(a.get("image_id"))
        if iid in keep_img_ids:
            keep_anns.append(a)

    out: dict[str, Any] = {}
    for k in ["info", "licenses", "categories"]:
        if k in coco:
            out[k] = coco[k]
    out["images"] = keep_images
    out["annotations"] = keep_anns
    return out


def write_coco_split_json(out_path: Path, coco_full: dict[str, Any], keep_locations: list[str]) -> None:
    """write_coco_split_json(out_path, coco_full, keep_locations) -> None: Write filtered COCO json."""
    out = _filter_coco_by_locations(coco_full, keep_locations)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def write_ann_override_yml(path: Path, img_root: Path, train_ann: str, val_ann: str) -> None:
    """write_ann_override_yml(path, img_root, train_ann, val_ann) -> None: Dataset override yml."""
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
    """write_include_config_yml(path, base_config, override_yml, output_dir) -> None: Include base+override."""
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

    # Matches: {type: Normalize, mean: [...], std: [...]}
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


def _write_base_config_with_fold_norm(base_config_in: Path, base_config_out: Path, mean: list[float], std: list[float]) -> int:
    """_write_base_config_with_fold_norm(base_config_in, base_config_out, mean, std) -> int: Patch Normalize mean/std in base config."""
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

def _detect_all_locations(img_root_train: Path, coco_full: dict[str, Any]) -> list[str]:
    """_detect_all_locations(img_root_train, coco_full) -> list[str]: Detect all locations from folders (preferred) or COCO file_name."""
    locs_from_dirs: list[str] = []
    try:
        for p in sorted(img_root_train.iterdir()):
            if p.is_dir() and p.name.strip() and not p.name.startswith("."):
                locs_from_dirs.append(p.name)
    except Exception:
        locs_from_dirs = []

    if locs_from_dirs:
        return locs_from_dirs

    locs: set[str] = set()
    for im in coco_full.get("images", []) or []:
        fn = str(im.get("file_name", "") or "").replace("\\", "/").lstrip("./").lstrip("/")
        if not fn:
            continue
        parts = [p for p in fn.split("/") if p.strip()]
        if parts:
            locs.add(parts[0])
    return sorted(locs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--img_root_test", default="")
    ap.add_argument("--coco_val", required=True)
    ap.add_argument("--coco_test", default="")
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--select_metric", default="AP_precision_iou_0.50_area_all_maxdets_100")

    # Old behavior: if --test_locations is empty, locations are auto-detected.
    # New: --holdout_test_locations is ALWAYS the fixed 2-location TEST holdout.
    ap.add_argument("--test_locations", default="")
    ap.add_argument("--holdout_test_locations", default="")

    ap.add_argument("--mode", default="random")  # random|all
    ap.add_argument("--k", default="4")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--val_size", default="2")

    # Per-fold RGB normalization from TRAIN split (instances_train.json)
    ap.add_argument("--per_fold_norm_stats", default="1")  # 1|0
    ap.add_argument("--stats_workers", default=str(cpu_count()))
    ap.add_argument("--stats_progress_step_pct", default="5")

    ap.add_argument("--gpus", default="0")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--master_port", default="7777")

    ap.add_argument("--eval_after_each_fold", default="0")
    ap.add_argument("--eval_name", default="eval_data")
    ap.add_argument("--eval_gpus", default="")
    ap.add_argument("--eval_nproc", default="1")
    ap.add_argument("--eval_master_port", default="7788")
    ap.add_argument("--overwrite_eval", default="0")

    ap.add_argument("--label_offset", default="0")
    ap.add_argument("--use_amp", action="store_true")
    sys.argv = [a.replace("\r", "") for a in sys.argv]
    args, unknown = ap.parse_known_args()

    unknown_clean = [u for u in unknown if str(u).replace("\r", "").strip() != ""]
    if unknown_clean:
        ap.error("unrecognized arguments: " + " ".join(unknown_clean))

    repo_root = Path(__file__).resolve().parents[1]

    img_root_train = Path(resolve_path(repo_root, args.img_root)).resolve()
    img_root_test = Path(resolve_path(repo_root, args.img_root_test)).resolve() if str(args.img_root_test).strip() else None

    coco_val_abs = resolve_path(repo_root, args.coco_val)
    coco_test_abs = resolve_path(repo_root, args.coco_test) if str(args.coco_test).strip() else ""

    base_config_abs = resolve_path(repo_root, args.base_config)
    init_ckpt_abs = resolve_path(repo_root, args.pretrained) if str(args.pretrained).strip() else ""

    results_dir = Path(args.results_dir).resolve()
    cv_root = results_dir / "cross_validation"
    cv_root.mkdir(parents=True, exist_ok=True)

    coco_full = _require_coco_dict(read_json(coco_val_abs), coco_val_abs)

    # 1) all locations: explicit list if provided, else auto-detect (old-style)
    all_locations = extract_locations_from_csv_arg(args.test_locations) if str(args.test_locations).strip() else []
    if not all_locations:
        all_locations = _detect_all_locations(img_root_train, coco_full)

    if not all_locations:
        raise SystemExit(
            "ERROR: Could not determine ALL locations. Provide --test_locations as CSV, "
            "or ensure img_root contains location subfolders, or COCO images[*].file_name starts with 'Location/...'."
        )

    holdout_test = extract_locations_from_csv_arg(args.holdout_test_locations) if str(args.holdout_test_locations).strip() else []
    if len(holdout_test) not in (0, 2):
        raise SystemExit(
            f"ERROR: --holdout_test_locations must contain 0 or exactly 2 locations, got {len(holdout_test)}: {holdout_test}"
        )

    missing = [x for x in holdout_test if x not in all_locations]
    if missing:
        raise SystemExit(f"ERROR: Holdout test locations not found among detected locations: {missing}")

    # 3) CV pool = all minus optional holdout test
    cv_locations = [x for x in all_locations if x not in set(holdout_test)]

    val_size = int(args.val_size)
    if val_size < 1:
        raise SystemExit(f"ERROR: --val_size must be >= 1, got {val_size}.")
    if len(cv_locations) <= val_size:
        raise SystemExit(
            f"ERROR: Not enough locations for CV after removing holdout_test. "
            f"Need len(cv_locations) > val_size, got len(cv_locations)={len(cv_locations)} val_size={val_size}. "
            f"all_locations={all_locations} holdout_test={holdout_test}"
        )

    print(f"[train_crossval] All locations ({len(all_locations)}): {all_locations}", flush=True)
    print(f"[train_crossval] Holdout TEST ({len(holdout_test)}): {holdout_test}", flush=True)
    print(f"[train_crossval] CV pool ({len(cv_locations)}): {cv_locations}", flush=True)

    if str(args.mode).strip().lower() == "all":
        folds = make_all_combinations_folds(cv_locations, val_size)
    else:
        folds = make_random_folds(cv_locations, int(args.k), int(args.seed), val_size)

    run_meta: dict[str, Any] = {
        "results_dir": str(results_dir),
        "cross_validation_dir": str(cv_root),
        "folds": [],
        "base_config": str(Path(base_config_abs).resolve()),
        "pretrained": (str(Path(init_ckpt_abs).resolve()) if init_ckpt_abs else None),
        "img_root": str(img_root_train),
        "img_root_test": (str(img_root_test) if img_root_test else None),
        "coco_val": str(Path(coco_val_abs).resolve()),
        "coco_test": (str(Path(coco_test_abs).resolve()) if coco_test_abs else None),
        "mode": str(args.mode),
        "k": int(args.k),
        "seed": int(args.seed),
        "val_size": val_size,
        "all_locations": all_locations,
        "holdout_test_locations": holdout_test,
        "cv_locations": cv_locations,
    }

    train_env = ensure_env_cuda_visible_devices(str(args.gpus))
    master_port = int(args.master_port)
    nproc = int(args.nproc)

    for fold in folds:
        if len(fold.val_locations) != int(val_size):
            raise SystemExit(
                f"ERROR: Fold split mismatch. Expected val_size={int(val_size)}. "
                f"Got train={len(fold.train_locations)} val={len(fold.val_locations)} "
                f"train={fold.train_locations} val={fold.val_locations}"
            )
        if len(fold.train_locations) < 1:
            raise SystemExit(
                f"ERROR: Fold split mismatch. Training set became empty. "
                f"train={fold.train_locations} val={fold.val_locations}"
            )

        fold_dir = cv_root / f"fold{fold.fold_id + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        splits_dir = fold_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        train_ann_path = (splits_dir / "instances_train.json").resolve()
        val_ann_path = (splits_dir / "instances_val.json").resolve()

        write_coco_split_json(train_ann_path, coco_full, fold.train_locations)
        write_coco_split_json(val_ann_path, coco_full, fold.val_locations)

        cfg_dir = fold_dir / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        override_yml = cfg_dir / "split_override.yml"
        run_cfg = cfg_dir / "train_config.yml"

        write_ann_override_yml(override_yml, img_root_train, str(train_ann_path), str(val_ann_path))

        # ----- Per-fold mean/std computed on TRAIN split (=> train locations of this fold) -----
        base_cfg_for_fold = Path(base_config_abs).resolve()
        stats_json_path = cfg_dir / "dataset_rgb_mean_std_train.json"
        base_cfg_patched = cfg_dir / "base_config_with_fold_norm.yml"

        fold_mean: list[float] | None = None
        fold_std: list[float] | None = None
        n_images = 0
        n_pixels = 0
        n_norm_patched = 0

        if str(getattr(args, "per_fold_norm_stats", "1")).strip() == "1":
            mean, std, n_images, n_pixels = _compute_mean_std_from_coco(
                img_root=img_root_train,
                coco_path=train_ann_path,
                workers=int(getattr(args, "stats_workers", cpu_count())),
                progress_step_pct=int(getattr(args, "stats_progress_step_pct", 5)),
            )
            fold_mean, fold_std = mean, std
            n_norm_patched = _write_base_config_with_fold_norm(
                base_config_in=base_cfg_for_fold,
                base_config_out=base_cfg_patched,
                mean=mean,
                std=std,
            )

            write_json(
                str(stats_json_path),
                {
                    "fold_id": int(fold.fold_id),
                    "train_locations": fold.train_locations,
                    "train_ann": str(train_ann_path),
                    "img_root": str(img_root_train),
                    "n_images_processed": int(n_images),
                    "n_pixels_total": int(n_pixels),
                    "mean_rgb_01": mean,
                    "std_rgb_01": std,
                    "patched_normalize_blocks": int(n_norm_patched),
                    "note": "Exact RGB mean/std over TRAIN split (instances_train.json). Applied to all inline Normalize blocks (train+val).",
                },
            )

            base_cfg_for_include = base_cfg_patched
        else:
            base_cfg_for_include = base_cfg_for_fold

        write_include_config_yml(run_cfg, str(base_cfg_for_include), str(override_yml), str(fold_dir))

        log_dir = fold_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "train_stdout.log"

        cmd = [
            "torchrun",
            f"--master_port={master_port + fold.fold_id}",
            f"--nproc_per_node={nproc}",
            "train.py",
            "-c",
            str(run_cfg),
        ]
        if init_ckpt_abs:
            cmd += ["-t", init_ckpt_abs]
        if getattr(args, "use_amp", False):
            cmd.append("--use-amp")

        print("\n=== FOLD", fold.fold_id, "===", flush=True)
        print("TRAIN:", fold.train_locations, flush=True)
        print("VAL  :", fold.val_locations, flush=True)

        rc = run_and_tee(cmd, env=train_env, cwd=str(repo_root), log_path=str(log_path))
        if rc != 0:
            raise SystemExit(f"Fold {fold.fold_id} training failed: exit {rc}")


        export_tb_plots(repo_root=repo_root, run_dir=fold_dir, out_subdir="tb_exports")
        fold_ckpt = pick_checkpoint(fold_dir)

        fold_meta: dict[str, Any] = {
            "fold_id": int(fold.fold_id),
            "train_locations": fold.train_locations,
            "val_locations": fold.val_locations,
            "train_ann": str(train_ann_path),
            "val_ann": str(val_ann_path),
            "test_ann": (str(Path(coco_test_abs).resolve()) if (coco_test_abs and holdout_test) else None),
            "log": str(log_path.resolve()),
            "fold_checkpoint": str(fold_ckpt.resolve()),
            "holdout_test_locations": holdout_test,
            "base_config_original": str(Path(base_config_abs).resolve()),
            "base_config_with_fold_norm": (str(base_cfg_patched.resolve()) if base_cfg_patched.exists() else None),
            "fold_norm_stats_json": (str(stats_json_path.resolve()) if stats_json_path.exists() else None),
            "fold_mean_rgb_01": fold_mean,
            "fold_std_rgb_01": fold_std,
        }

        write_json(str(fold_dir / "fold_meta.json"), fold_meta)

        if str(args.eval_after_each_fold).strip() == "1":
            repo_root2 = Path(__file__).resolve().parents[1]
            base_cfg = Path(fold_meta.get("base_config_with_fold_norm") or base_config_abs).resolve()
            ckpt = fold_ckpt.resolve()

            out_val = fold_dir / "eval_val" / str(args.eval_name)
            rc_val = _run_eval_one(
                repo_root=repo_root2,
                base_config=base_cfg,
                checkpoint=ckpt,
                img_root=img_root_train,
                ann=val_ann_path,
                out_dir=out_val,
                gpus=(args.eval_gpus or args.gpus),
                nproc=str(args.eval_nproc),
                master_port=int(args.eval_master_port) + int(fold.fold_id),
                overwrite=str(args.overwrite_eval),
                label_offset=str(args.label_offset),
            )
            if rc_val != 0:
                raise SystemExit(f"Fold {fold.fold_id} validation eval failed: exit {rc_val}")
            eval_utils.copy_eval_artifacts_to_metrics_folder(out_val, fold_dir, "validation", overwrite=str(args.overwrite_eval))

            if coco_test_abs and holdout_test:
                out_test = fold_dir / "eval_test" / str(args.eval_name)
                rc_test = _run_eval_one(
                    repo_root=repo_root2,
                    base_config=base_cfg,
                    checkpoint=ckpt,
                    img_root=img_root_test if img_root_test else img_root_train,
                    ann=Path(coco_test_abs).resolve(),
                    out_dir=out_test,
                    gpus=(args.eval_gpus or args.gpus),
                    nproc=str(args.eval_nproc),
                    master_port=int(args.eval_master_port) + 100 + int(fold.fold_id),
                    overwrite=str(args.overwrite_eval),
                    label_offset=str(args.label_offset),
                )
                if rc_test != 0:
                    raise SystemExit(f"Fold {fold.fold_id} test eval failed: exit {rc_test}")
                eval_utils.copy_eval_artifacts_to_metrics_folder(out_test, fold_dir, "test", overwrite=str(args.overwrite_eval))

        run_meta["folds"].append(
            {
                "fold_id": int(fold.fold_id),
                "fold_dir": str(fold_dir.resolve()),
                "fold_checkpoint": str(fold_ckpt.resolve()),
                "train_locations": fold.train_locations,
                "val_locations": fold.val_locations,
            }
        )

        cleanup_numbered_checkpoints(fold_dir)

    write_json(str(results_dir / "run_meta.json"), run_meta)
    print("\nDONE. Results written to:", results_dir, flush=True)


if __name__ == "__main__":
    main()