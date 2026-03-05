#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import eval_utils


def read_json(path: Path) -> dict[str, Any]:
    """read_json(path) -> dict[str,Any]: Read JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_fold_dirs(results_dir: Path) -> list[Path]:
    """find_fold_dirs(results_dir) -> list[Path]: Return sorted fold directories under cross_validation."""
    cv = results_dir / "cross_validation"
    if not cv.exists():
        return []
    folds = [p for p in cv.iterdir() if p.is_dir() and p.name.startswith("fold")]

    def key(p: Path) -> int:
        s = p.name.replace("fold", "")
        return int(s) if s.isdigit() else 10**9

    return sorted(folds, key=key)


def run_eval_one(
    repo_root: Path,
    base_config: Path,
    checkpoint: Path,
    img_root: Path,
    ann: Path,
    out_dir: Path,
    gpus: str,
    nproc: str,
    master_port: str,
    overwrite: str,
    label_offset: str,
    score_thr: str,
    optimize_score_thr: str,
) -> int:
    """run_eval_one(...) -> int: Run eval_one_deimv2.py with dump_predictions enabled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
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
        str(master_port),
        "--overwrite",
        str(overwrite),
        "--dump_predictions",
        "1",
        "--label_offset",
        str(label_offset),
        "--score_thr",
        str(score_thr),
        "--optimize_score_thr",
        str(optimize_score_thr),
    ]
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    if str(gpus).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpus).strip()
    return subprocess.call(cmd, cwd=str(repo_root), env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--eval_name", default="eval_data")
    ap.add_argument("--split", default="val")  # val|test|both
    ap.add_argument("--gpus", default="")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--master_port", default="7788")
    ap.add_argument("--overwrite", default="0")
    ap.add_argument("--label_offset", default="0")
    ap.add_argument("--include_final", default="0")  # 1|0

    # Forwarded into eval_one_deimv2.py -> extra_detection_metrics.py
    ap.add_argument("--score_thr", default="0.05")
    ap.add_argument("--optimize_score_thr", default="1")  # 1 => optimize image-level F1, 0 => fixed score_thr
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    img_root = Path(args.img_root).resolve()

    folds = find_fold_dirs(results_dir)
    if not folds:
        raise SystemExit(f"[dump_predictions_all.py] ERROR: no folds found under {results_dir}/cross_validation")

    port0 = int(str(args.master_port))
    rc_any = 0

    for i, fold_dir in enumerate(folds):
        meta_path = fold_dir / "fold_meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = read_json(meta_path)

            fold_cfg = meta.get("base_config_with_fold_norm")
            if not fold_cfg:
                raise SystemExit(
                    f"[dump_predictions_all] ERROR: fold meta has no base_config_with_fold_norm: {meta_path}\n"
                    "Refusing to evaluate without the per-fold patched config."
                )
            fold_base_cfg = Path(fold_cfg).resolve()
            if not fold_base_cfg.exists():
                raise SystemExit(f"[dump_predictions_all] ERROR: config missing: {fold_base_cfg}")


        else:
            raise SystemExit(

                f"[dump_predictions_all] ERROR: missing fold_meta.json: {meta_path}\n"

                "Refusing to evaluate without per-fold patched config."

            )


        ckpt_str = str(meta.get("fold_checkpoint", "")).strip()
        ckpt = eval_utils.resolve_checkpoint_path(results_dir, fold_dir, ckpt_str)
        if ckpt is None or not ckpt.exists():
            print(f"[dump_predictions_all.py] SKIP {fold_dir.name}: checkpoint missing/unresolvable: {ckpt_str}", flush=True)
            rc_any = 1
            continue

        val_ann = eval_utils.resolve_val_ann_path(fold_dir, meta)
        test_ann = eval_utils.resolve_test_ann_path(results_dir, fold_dir, meta)

        if args.split in ("val", "both"):
            out_val = fold_dir / "eval_val" / str(args.eval_name)
            if val_ann is None or not val_ann.exists():
                print(f"[dump_predictions_all.py] SKIP {fold_dir.name} val: ann missing/unresolvable", flush=True)
            else:

                rc = run_eval_one(
                    repo_root=repo_root,
                    base_config=fold_base_cfg,
                    checkpoint=ckpt,
                    img_root=img_root,
                    ann=val_ann,
                    out_dir=out_val,
                    gpus=args.gpus,
                    nproc=args.nproc,
                    master_port=str(port0 + i),
                    overwrite=args.overwrite,
                    label_offset=args.label_offset,
                    score_thr=args.score_thr,
                    optimize_score_thr=args.optimize_score_thr,
                )
                if rc != 0:
                    rc_any = rc_any or rc

                eval_utils.copy_eval_artifacts_to_metrics_folder(out_val, fold_dir, "validation", overwrite=str(args.overwrite))

        if args.split in ("test", "both"):
            out_test = fold_dir / "eval_test" / str(args.eval_name)
            if test_ann is None or not test_ann.exists():
                print(f"[dump_predictions_all.py] SKIP {fold_dir.name} test: ann missing/unresolvable", flush=True)
            else:

                rc = run_eval_one(
                    repo_root=repo_root,
                    base_config=fold_base_cfg,
                    checkpoint=ckpt,
                    img_root=img_root,
                    ann=test_ann,
                    out_dir=out_test,
                    gpus=args.gpus,
                    nproc=args.nproc,
                    master_port=str(port0 + 100 + i),
                    overwrite=args.overwrite,
                    label_offset=args.label_offset,
                    score_thr=args.score_thr,
                    optimize_score_thr=args.optimize_score_thr,
                )
                if rc != 0:
                    rc_any = rc_any or rc

                eval_utils.copy_eval_artifacts_to_metrics_folder(out_test, fold_dir, "test", overwrite=str(args.overwrite))

    if str(args.include_final).strip() == "1":
        final_dir = results_dir / "final_location_holdout"
        meta_path = final_dir / "final_meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)

            final_cfg = meta.get("base_config_with_train_norm") or meta.get("base_config_with_train_norm".strip())
            if not final_cfg:
                raise SystemExit(
                    f"[dump_predictions_all] ERROR: final meta has no base_config_with_train_norm: {meta_path}\n"
                    "Refusing to evaluate without the final patched config."
                )
            final_base_cfg = Path(final_cfg).resolve()
            if not final_base_cfg.exists():
                raise SystemExit(f"[dump_predictions_all] ERROR: config missing: {final_base_cfg}")


            ckpt_str = str(meta.get("final_checkpoint", "")).strip()
            ckpt = eval_utils.resolve_checkpoint_path(results_dir, final_dir, ckpt_str)
            if ckpt is not None and ckpt.exists():
                val_ann_s = str(meta.get("val_ann", "")).strip()
                val_ann = Path(val_ann_s).expanduser() if val_ann_s else None
                if val_ann is not None and not val_ann.exists() and val_ann_s:
                    try:
                        name = Path(val_ann_s).name
                        val_ann = eval_utils._first_existing([final_dir / "splits" / name, final_dir / name]) or val_ann
                    except Exception:
                        pass

                if args.split in ("val", "both") and val_ann is not None and val_ann.exists():
                    out_val = final_dir / "eval_val" / str(args.eval_name)
                    rc = run_eval_one(
                        repo_root=repo_root,
                        base_config=final_base_cfg,
                        checkpoint=ckpt,
                        img_root=img_root,
                        ann=val_ann,
                        out_dir=out_val,
                        gpus=args.gpus,
                        nproc=args.nproc,
                        master_port=str(port0 + 500),
                        overwrite=args.overwrite,
                        label_offset=args.label_offset,
                        score_thr=args.score_thr,
                        optimize_score_thr=args.optimize_score_thr,
                    )
                    if rc != 0:
                        rc_any = rc_any or rc
                    eval_utils.copy_eval_artifacts_to_metrics_folder(out_val, final_dir, "validation", overwrite=str(args.overwrite))

                test_ann_s = str(meta.get("coco_test", "")).strip()
                if args.split in ("test", "both") and test_ann_s:
                    test_ann = Path(test_ann_s).expanduser()
                    if test_ann.exists():
                        out_test = final_dir / "eval_test" / str(args.eval_name)
                        rc = run_eval_one(
                            repo_root=repo_root,
                            base_config=final_base_cfg,
                            checkpoint=ckpt,
                            img_root=img_root,
                            ann=test_ann,
                            out_dir=out_test,
                            gpus=args.gpus,
                            nproc=args.nproc,
                            master_port=str(port0 + 600),
                            overwrite=args.overwrite,
                            label_offset=args.label_offset,
                            score_thr=args.score_thr,
                            optimize_score_thr=args.optimize_score_thr,
                        )
                        if rc != 0:
                            rc_any = rc_any or rc
                        eval_utils.copy_eval_artifacts_to_metrics_folder(out_test, final_dir, "test", overwrite=str(args.overwrite))
            else:
                print(f"[dump_predictions_all.py] NOTE: final checkpoint missing/unresolvable: {ckpt_str}", flush=True)
        else:
            print(f"[dump_predictions_all.py] NOTE: no final_meta.json at {meta_path}", flush=True)

    return int(rc_any)


if __name__ == "__main__":
    raise SystemExit(main())