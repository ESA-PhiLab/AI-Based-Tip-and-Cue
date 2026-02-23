#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch
import yaml

import eval_utils


def read_json(path: str) -> Any:
    """read_json(path) -> Any: Read JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """write_json(path, obj) -> None: Write JSON with mkdir."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """write_text(path, text) -> None: Write text with mkdir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_metrics_xlsx(path: Path, metrics: dict[str, float]) -> None:
    """write_metrics_xlsx(path, metrics) -> None: Write single-row metrics.xlsx."""
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    keys = sorted(metrics.keys())
    ws.append(keys)
    ws.append([metrics.get(k) for k in keys])
    wb.save(str(path))


def _cuda_usable() -> bool:
    """_cuda_usable() -> bool: True if CUDA is usable by this torch build."""
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.empty((1,), device="cuda")
        return True
    except Exception:
        return False


def _ensure_dir(p: Path) -> None:
    """_ensure_dir(p) -> None: mkdir -p."""
    p.mkdir(parents=True, exist_ok=True)


def _write_yaml(path: Path, obj: Any) -> None:
    """_write_yaml(path, obj) -> None: Write YAML."""
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _make_eval_config(base_config: Path, out_dir: Path, img_root: Path, ann: Path, *, eval_num_workers: int) -> Path:
    """_make_eval_config(base_config, out_dir, img_root, ann, eval_num_workers) -> Path: Create eval_config.yml under out_dir/config."""
    cfg_dir = out_dir / "config"
    _ensure_dir(cfg_dir)

    # Override val dataloader to avoid FD/shm pressure on some systems.
    dataset_override = {
        "val_dataloader": {
            "dataset": {
                "img_folder": str(img_root),
                "ann_file": str(ann),
            },
            "num_workers": int(eval_num_workers),
            "persistent_workers": False,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        }
    }
    dataset_override_path = cfg_dir / "dataset_override.yml"
    _write_yaml(dataset_override_path, dataset_override)

    eval_cfg = {
        "__include__": [str(base_config), str(dataset_override_path)],
        "output_dir": str(out_dir),
        "resume": None,
        "test_only": True,
        "print_method": "builtin",
        "print_rank": 0,
    }
    eval_cfg_path = cfg_dir / "eval_config.yml"
    _write_yaml(eval_cfg_path, eval_cfg)
    return eval_cfg_path


def coco_num_images(ann_path: str) -> int:
    """coco_num_images(ann_path) -> int: Return #images in COCO annotation."""
    try:
        obj = read_json(ann_path)
        if isinstance(obj, dict):
            return int(len(obj.get("images", []) or []))
    except Exception:
        pass
    return 0


def parse_final_coco_metrics(log_path: Path) -> dict[str, float]:
    """parse_final_coco_metrics(log_path) -> dict[str,float]: Parse COCO AP/AR lines from eval stdout log."""
    txt = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    coco_line = re.compile(
        r"Average\s+(Precision|Recall)\s+\((AP|AR)\)\s*@\[\s*IoU=([0-9\.]+)(?::([0-9\.]+))?\s*\|\s*area=\s*([a-zA-Z]+)\s*\|\s*maxDets=([0-9]+)\s*\]\s*=\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    )
    out: dict[str, float] = {}
    for line in txt:
        m = coco_line.search(line)
        if not m:
            continue
        ap_ar = m.group(2).upper()
        iou_lo = m.group(3)
        iou_hi = m.group(4)
        area = m.group(5).lower()
        maxd = m.group(6)
        try:
            val = float(m.group(7))
        except Exception:
            continue
        iou_key = f"{iou_lo}:{iou_hi}" if iou_hi else iou_lo
        key = f"{ap_ar}_{'precision' if ap_ar == 'AP' else 'recall'}_iou_{iou_key}_area_{area}_maxdets_{maxd}"
        out[key] = val
    return out


def _set_torch_mp_sharing_strategy() -> None:
    """_set_torch_mp_sharing_strategy() -> None: Prefer file_system sharing to reduce FD/shm pressure."""
    try:
        import torch.multiprocessing as mp  # type: ignore

        mp.set_sharing_strategy("file_system")
    except Exception:
        pass


def _ensure_predictions_file(pred_path: Path) -> bool:
    """_ensure_predictions_file(pred_path) -> bool: Ensure predictions_coco.json exists; salvage rank-sharded dumps if needed."""
    if pred_path.exists():
        return True

    cands = sorted(pred_path.parent.glob(pred_path.name + ".rank*.json"))
    if not cands:
        return False

    if len(cands) == 1:
        try:
            shutil.copyfile(cands[0], pred_path)
            print(f"[eval_one_deimv2.py] NOTE: Salvaged predictions from {cands[0].name} -> {pred_path.name}", flush=True)
            return True
        except Exception:
            return False

    merged: list[Any] = []
    for p in cands:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, list):
                merged.extend(obj)
        except Exception:
            continue

    if not merged:
        return False

    try:
        pred_path.write_text(json.dumps(merged), encoding="utf-8")
        print(
            f"[eval_one_deimv2.py] NOTE: Merged {len(cands)} rank dumps -> {pred_path.name} (n={len(merged)})",
            flush=True,
        )
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--master_port", default="7788")
    ap.add_argument("--overwrite", default="0")
    ap.add_argument("--dump_predictions", default="0")
    ap.add_argument("--label_offset", default="0")
    ap.add_argument("--extra_iou_thr", default="0.5")
    ap.add_argument("--eval_num_workers", default="0")
    args = ap.parse_args()

    base_config = Path(args.base_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    img_root = Path(args.img_root).resolve()
    ann = Path(args.ann).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not base_config.exists():
        print(f"[eval_one_deimv2.py] ERROR: base_config not found: {base_config}")
        return 2
    if not checkpoint.exists():
        print(f"[eval_one_deimv2.py] ERROR: checkpoint not found: {checkpoint}")
        return 2
    if not ann.exists():
        print(f"[eval_one_deimv2.py] ERROR: ann not found: {ann}")
        return 2
    if not img_root.exists():
        print(f"[eval_one_deimv2.py] ERROR: img_root not found: {img_root}")
        return 2

    if str(args.overwrite).strip() == "1" and out_dir.exists():
        shutil.rmtree(out_dir)
    _ensure_dir(out_dir)

    write_text(out_dir / "selected_checkpoint.txt", str(checkpoint) + "\n")

    _set_torch_mp_sharing_strategy()

    try:
        eval_num_workers = max(0, int(str(args.eval_num_workers).strip() or "0"))
    except Exception:
        eval_num_workers = 0

    device = "cuda" if _cuda_usable() else "cpu"
    eval_cfg_path = _make_eval_config(base_config, out_dir, img_root, ann, eval_num_workers=eval_num_workers)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if str(args.gpus).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()

    # Ensure a unique rendezvous even if DEIM/train.py initializes torch.distributed internally.
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(int(args.master_port))
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")

    pred_path = out_dir / "predictions_coco.json"
    if str(args.dump_predictions).strip() == "1":
        env["DEIM_DUMP_PREDICTIONS"] = str(pred_path.resolve())
    env["DEIM_LABEL_OFFSET"] = str(args.label_offset).strip()

    cmd = [
        sys.executable,
        "train.py",
        "-c",
        str(eval_cfg_path),
        "--test-only",
        "-r",
        str(checkpoint),
        "--device",
        device,
    ]
    print(" ".join(cmd), flush=True)

    log_path = out_dir / "eval_stdout.log"
    n_images = coco_num_images(str(ann))

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(repo_root), env=env)
        rc = p.wait()
    t1 = time.time()

    if rc != 0:
        print(f"[eval_one_deimv2.py] ERROR: Eval failed (exit code {rc}). See {log_path}")
        return rc

    if str(args.dump_predictions).strip() == "1":
        if not _ensure_predictions_file(pred_path):
            print(
                "[eval_one_deimv2.py] WARNING: dump_predictions=1 but predictions_coco.json was not produced (no rank dump found either).",
                flush=True,
            )

    latency_total_ms = (t1 - t0) * 1000.0
    latency_ms_per_image = (latency_total_ms / n_images) if n_images > 0 else None
    write_json(
        str(out_dir / "latency.json"),
        {
            "total_ms": float(latency_total_ms),
            "num_images": int(n_images),
            "ms_per_image": (float(latency_ms_per_image) if latency_ms_per_image is not None else None),
            "note": "Wall-clock time for full eval run divided by #images (includes dataloading + model + postprocess).",
        },
    )

    metrics = parse_final_coco_metrics(log_path)
    if not metrics:
        print(
            f"[eval_one_deimv2.py] WARNING: could not parse COCO metrics from {log_path}. Eval stdout format may have changed.",
            flush=True,
        )

    write_json(str(out_dir / "metrics.json"), metrics)
    write_metrics_xlsx(out_dir / "metrics.xlsx", metrics)

    if pred_path.exists():
        rc2 = subprocess.call(
            [sys.executable, "tools/coco_eval_predictions.py", "--gt", str(ann), "--pred", str(pred_path), "--out_dir", str(out_dir)],
            cwd=str(repo_root),
        )
        if rc2 != 0:
            print("[eval_one_deimv2.py] WARNING: coco_eval_predictions.py failed", flush=True)

        rc3 = subprocess.call(
            [sys.executable, "tools/extra_detection_metrics.py", "--gt", str(ann), "--pred", str(pred_path), "--out_dir", str(out_dir), "--iou_thr", str(float(args.extra_iou_thr))],
            cwd=str(repo_root),
        )
        if rc3 != 0:
            print("[eval_one_deimv2.py] WARNING: extra_detection_metrics.py failed", flush=True)
    else:
        print("[eval_one_deimv2.py] NOTE: predictions_coco.json not found; extra metrics are skipped (base metrics.json is still written).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())