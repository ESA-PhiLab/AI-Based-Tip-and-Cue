#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.utils import get_column_letter

import re
from datetime import datetime


_AVG_RE = re.compile(r"\bAveraged stats:\s.*?\bloss:\s[-+0-9.eE]+\s\(([-+0-9.eE]+)\)")
_EPOCH_RE = re.compile(r"^Epoch:\s*\[(\d+)\]")

def read_json(path: str) -> Any:
    """read_json(path) -> Any: Read JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """write_json(path, obj) -> None: Write JSON with mkdir."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(x: Any) -> float | None:
    """safe_float(x) -> float|None: Parse float safely."""
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None

def _safe_float(x: Any) -> float | None:
    """_safe_float(x) -> float|None: Backward-compatible alias for safe_float."""
    return safe_float(x)

def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    """mean_std(vals) -> (mean,std): Mean/std for list; returns (None,None) if empty."""
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    m = sum(vals) / float(len(vals))
    v = sum((x - m) ** 2 for x in vals) / float(max(1, len(vals) - 1))
    return float(m), float(math.sqrt(v))


def ensure_dir(p: Path) -> None:
    """ensure_dir(p) -> None: mkdir(parents=True, exist_ok=True)."""
    p.mkdir(parents=True, exist_ok=True)


def _first_existing(paths: list[Path]) -> Path | None:
    """_first_existing(paths) -> Path|None: First existing path."""
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def strip_numbered_suffix(p: Path) -> Path:
    """strip_numbered_suffix(p) -> Path: Convert DEIMxxx_4 to DEIMxxx if present."""
    m = re.match(r"^(.*)_\d+$", p.name)
    if not m:
        return p
    return p.with_name(m.group(1))


def resolve_results_dir(results_dir: Path) -> Path:
    """resolve_results_dir(results_dir) -> Path: Handle training-machine vs eval-machine naming."""
    if results_dir.exists():
        return results_dir
    alt = strip_numbered_suffix(results_dir)
    if alt.exists():
        return alt
    return results_dir


def read_meta(fold_dir: Path) -> dict[str, Any]:
    """read_meta(fold_dir) -> dict[str,Any]: Read fold_meta.json or final_meta.json if present."""
    cand = fold_dir / "fold_meta.json"
    if cand.exists():
        try:
            j = read_json(str(cand))
            if isinstance(j, dict):
                return j
        except Exception:
            return {}
    cand2 = fold_dir / "final_meta.json"
    if cand2.exists():
        try:
            j = read_json(str(cand2))
            if isinstance(j, dict):
                return j
        except Exception:
            return {}
    return {}


def resolve_val_ann_path(fold_dir: Path, meta: dict[str, Any]) -> Path:
    """resolve_val_ann_path(fold_dir, meta) -> Path: Find validation annotations."""
    # Prefer explicit meta
    for k in ["val_ann", "coco_val", "ann_val"]:
        v = str(meta.get(k) or "").strip()
        if v:
            p = Path(v).expanduser()
            if p.exists():
                return p
            p2 = fold_dir / p.name
            if p2.exists():
                return p2

    # Common locations
    cands = [
        fold_dir / "splits" / "instances_val.json",
        fold_dir / "instances_val.json",
        fold_dir / "val.json",
        fold_dir / "annotations" / "instances_val.json",
        fold_dir / "annotations" / "val.json",
    ]
    p = _first_existing(cands)
    if p is None:
        raise FileNotFoundError(f"Could not resolve val annotation json under {fold_dir}")
    return p

def resolve_val_ann_path(*args) -> Path:
    """resolve_val_ann_path(results_dir?, fold_dir, meta) -> Path: Find validation annotations (supports old/new call signatures)."""
    if len(args) == 2:
        fold_dir, meta = args
    elif len(args) == 3:
        _, fold_dir, meta = args
    else:
        raise TypeError(f"resolve_val_ann_path() expected 2 or 3 args, got {len(args)}")

    fold_dir = Path(fold_dir)
    meta = meta if isinstance(meta, dict) else {}

    for k in ["val_ann", "coco_val", "ann_val"]:
        v = str(meta.get(k) or "").strip()
        if v:
            p = Path(v).expanduser()
            if p.exists():
                return p
            p2 = fold_dir / p.name
            if p2.exists():
                return p2

    cands = [
        fold_dir / "splits" / "instances_val.json",
        fold_dir / "instances_val.json",
        fold_dir / "val.json",
        fold_dir / "annotations" / "instances_val.json",
        fold_dir / "annotations" / "val.json",
    ]
    p = _first_existing(cands)
    if p is None:
        raise FileNotFoundError(f"Could not resolve val annotation json under {fold_dir}")
    return p


def resolve_test_ann_path(*args) -> Path:
    """resolve_test_ann_path(results_dir?, fold_dir, meta) -> Path: Find test annotations (supports old/new call signatures)."""
    if len(args) == 2:
        fold_dir, meta = args
    elif len(args) == 3:
        _, fold_dir, meta = args
    else:
        raise TypeError(f"resolve_test_ann_path() expected 2 or 3 args, got {len(args)}")

    fold_dir = Path(fold_dir)
    meta = meta if isinstance(meta, dict) else {}

    for k in ["test_ann", "coco_test", "ann_test"]:
        v = str(meta.get(k) or "").strip()
        if v:
            p = Path(v).expanduser()
            if p.exists():
                return p
            p2 = fold_dir / p.name
            if p2.exists():
                return p2

    cands = [
        fold_dir / "splits" / "instances_test.json",
        fold_dir / "instances_test.json",
        fold_dir / "test.json",
        fold_dir / "annotations" / "instances_test.json",
        fold_dir / "annotations" / "test.json",
    ]
    p = _first_existing(cands)
    if p is None:
        raise FileNotFoundError(f"Could not resolve test annotation json under {fold_dir}")
    return p


def resolve_img_root(meta: dict[str, Any], default: Path) -> Path:
    """resolve_img_root(meta, default) -> Path: Resolve img_root with fallback."""
    v = str(meta.get("img_root") or meta.get("image_root") or "").strip()
    if v:
        p = Path(v).expanduser()
        if p.exists():
            return p
    return default


def resolve_img_root_test(meta: dict[str, Any], default: Path) -> Path:
    """resolve_img_root_test(meta, default) -> Path: Resolve img_root_test with fallback."""
    v = str(meta.get("img_root_test") or meta.get("image_root_test") or "").strip()
    if v:
        p = Path(v).expanduser()
        if p.exists():
            return p
    return default


def resolve_checkpoint(meta: dict[str, Any], fold_dir: Path) -> Path | None:
    """resolve_checkpoint(meta, fold_dir) -> Path|None: Resolve checkpoint path."""
    for k in ["best_checkpoint", "checkpoint", "final_checkpoint", "ckpt"]:
        v = str(meta.get(k) or "").strip()
        if v:
            p = Path(v).expanduser()
            if p.exists():
                return p
            p2 = fold_dir / p.name
            if p2.exists():
                return p2
            p3 = fold_dir / "checkpoints" / p.name
            if p3.exists():
                return p3

    for name in ["best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth"]:
        p = fold_dir / name
        if p.exists():
            return p
        p2 = fold_dir / "checkpoints" / name
        if p2.exists():
            return p2

    return None


def _make_xlsx(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """_make_xlsx(path, header, rows) -> None: Write simple XLSX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "sheet1"
    ws.append(header)
    for r in rows:
        ws.append(list(r))

    for col in range(1, len(header) + 1):
        ws.column_dimensions[get_column_letter(col)].width = max(12, min(45, len(str(header[col - 1])) + 2))
    wb.save(str(path))


def collect_eval_metrics(eval_dir: Path) -> dict[str, Any]:
    """collect_eval_metrics(eval_dir) -> dict[str,Any]: Load metrics(+extra) and latency if present."""
    out: dict[str, Any] = {}
    m = eval_dir / "metrics.json"
    mx = eval_dir / "metrics_extra.json"
    lat = eval_dir / "latency.json"

    if m.exists():
        try:
            mj = read_json(str(m))
            if isinstance(mj, dict):
                out.update(mj)
        except Exception:
            pass

    if mx.exists():
        try:
            mxj = read_json(str(mx))
            if isinstance(mxj, dict):
                for k, v in mxj.items():
                    out[f"extra_{k}"] = v
        except Exception:
            pass

    if lat.exists():
        try:
            lj = read_json(str(lat))
            if isinstance(lj, dict):
                if "ms_per_image" in lj:
                    out["latency_ms_per_image"] = lj.get("ms_per_image")
                if "total_ms" in lj:
                    out["latency_total_ms"] = lj.get("total_ms")
                if "num_images" in lj:
                    out["latency_num_images"] = lj.get("num_images")
        except Exception:
            pass

    return out


def plot_barplot(out_path: Path, title: str, rows: list[dict[str, Any]], key: str) -> None:
    """plot_barplot(out_path, title, rows, key) -> None: Bar plot per fold for key."""
    labels: list[str] = []
    ys: list[float] = []
    for r in rows:
        labels.append(str(r.get("fold", "")))
        v = safe_float(r.get(key))
        ys.append(float(v) if v is not None else float("nan"))

    if not labels:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 0.6), 4))
    plt.bar(labels, ys)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def plot_boxplot(out_path: Path, title: str, rows: list[dict[str, Any]], key: str) -> None:
    """plot_boxplot(out_path, title, rows, key) -> None: Box plot per fold for key."""
    labels: list[str] = []
    vals: list[list[float]] = []
    for r in rows:
        labels.append(str(r.get("fold", "")))
        v = safe_float(r.get(key))
        if v is None:
            vals.append([float("nan")])
        else:
            vals.append([float(v)])

    if not labels:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 0.4), 4))
    plt.boxplot(vals, labels=labels, vert=True)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def copy_eval_artifacts_to_metrics_folder(eval_dir: Path, fold_dir: Path, split_name: str, overwrite: str = "0") -> dict[str, Any]:
    """copy_eval_artifacts_to_metrics_folder(eval_dir, fold_dir, split_name, overwrite) -> dict[str,Any]: Copy key eval files into fold_dir/metrics/{split_name}."""
    dst = fold_dir / "metrics" / split_name
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for fn in [
        "metrics.json",
        "metrics_detailed.json",
        "metrics_extra.json",
        "eval_stdout.log",
        "latency.json",
        "metrics.xlsx",
        "predictions_coco.json",
    ]:
        src = eval_dir / fn
        if src.exists():
            d = dst / fn
            if overwrite == "1" or not d.exists():
                try:
                    shutil.copyfile(src, d)
                    copied.append(fn)
                except Exception:
                    pass

    # Also copy rank-sharded dumps if they exist (distributed eval).
    for rp in sorted(eval_dir.glob("predictions_coco.json.rank*.json")):
        d = dst / rp.name
        if overwrite == "1" or not d.exists():
            try:
                shutil.copyfile(rp, d)
                copied.append(rp.name)
            except Exception:
                pass

    if fold_dir.name == "final_location_holdout":
        src_pred = eval_dir / "predictions_coco.json"
        if src_pred.exists():
            # Always export split-specific final prediction files.
            if split_name == "validation":
                dst_pred = fold_dir / "final_predictions_val.json"
                if overwrite == "1" or not dst_pred.exists():
                    try:
                        shutil.copyfile(src_pred, dst_pred)
                        copied.append(dst_pred.name)
                    except Exception:
                        pass

            if split_name == "test":
                dst_pred = fold_dir / "final_predictions_test.json"
                if overwrite == "1" or not dst_pred.exists():
                    try:
                        shutil.copyfile(src_pred, dst_pred)
                        copied.append(dst_pred.name)
                    except Exception:
                        pass

            # Always maintain final_predictions.json as:
            # test if available, else validation if available, else current src_pred.
            test_p = fold_dir / "final_predictions_test.json"
            val_p = fold_dir / "final_predictions_val.json"

            if test_p.exists():
                alias_src = test_p
            elif val_p.exists():
                alias_src = val_p
            else:
                alias_src = src_pred

            alias = fold_dir / "final_predictions.json"
            if overwrite == "1" or not alias.exists():
                try:
                    shutil.copyfile(alias_src, alias)
                    copied.append(alias.name)
                except Exception:
                    pass

    plots_src = eval_dir / "plots"
    if plots_src.exists():
        plots_dst = dst / "plots"
        plots_dst.mkdir(parents=True, exist_ok=True)
        for p in plots_src.glob("*.png"):
            d = plots_dst / p.name
            if overwrite == "1" or not d.exists():
                try:
                    shutil.copyfile(p, d)
                    copied.append(f"plots/{p.name}")
                except Exception:
                    pass

    return {"ok": True, "dst": str(dst), "copied": copied}

def resolve_checkpoint_path(results_dir: Path, fold_dir: Path, ckpt_str: str) -> Path | None:
    """resolve_checkpoint_path(results_dir, fold_dir, ckpt_str) -> Path|None: Backward-compatible checkpoint resolver."""
    s = str(ckpt_str or "").strip()
    if not s:
        return None

    p = Path(s).expanduser()
    if p.exists():
        return p

    # Try relative to fold_dir
    cand = fold_dir / p.name
    if cand.exists():
        return cand

    # Try fold checkpoints/
    cand2 = fold_dir / "checkpoints" / p.name
    if cand2.exists():
        return cand2

    # Try results_dir/final_location_holdout (used on eval machines)
    cand3 = results_dir / "final_location_holdout" / p.name
    if cand3.exists():
        return cand3

    cand4 = results_dir / "final_location_holdout" / "checkpoints" / p.name
    if cand4.exists():
        return cand4

    return None

def safe_slug(s: str) -> str:
    """safe_slug(s) -> str: Filesystem-safe slug."""
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "metric"


def _try_parse_json_line(line: str) -> dict[str, Any] | None:
    """_try_parse_json_line(line) -> dict|None: Parse JSON dict from a single line if possible."""
    line = line.strip()
    if not line or not line.startswith("{") or not line.endswith("}"):
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _coco_stats_to_named_metrics(stats: list[float]) -> dict[str, float]:
    """_coco_stats_to_named_metrics(stats) -> dict[str,float]: Convert COCOeval stats[0..11] into named AP/AR keys."""
    if not isinstance(stats, (list, tuple)) or len(stats) < 12:
        return {}
    s = [float(x) for x in stats[:12]]

    # Standard COCO ordering:
    # 0 AP @[.50:.95] all
    # 1 AP @[.50] all
    # 2 AP @[.75] all
    # 3 AP @[.50:.95] small
    # 4 AP @[.50:.95] medium
    # 5 AP @[.50:.95] large
    # 6 AR @[.50:.95] all maxDets=1
    # 7 AR @[.50:.95] all maxDets=10
    # 8 AR @[.50:.95] all maxDets=100
    # 9 AR @[.50:.95] small maxDets=100
    # 10 AR @[.50:.95] medium maxDets=100
    # 11 AR @[.50:.95] large maxDets=100
    return {
        "AP_precision_iou_0.50:0.95_area_all_maxdets_100": s[0],
        "AP_precision_iou_0.50_area_all_maxdets_100": s[1],
        "AP_precision_iou_0.75_area_all_maxdets_100": s[2],
        "AP_precision_iou_0.50:0.95_area_small_maxdets_100": s[3],
        "AP_precision_iou_0.50:0.95_area_medium_maxdets_100": s[4],
        "AP_precision_iou_0.50:0.95_area_large_maxdets_100": s[5],
        "AR_recall_iou_0.50:0.95_area_all_maxdets_1": s[6],
        "AR_recall_iou_0.50:0.95_area_all_maxdets_10": s[7],
        "AR_recall_iou_0.50:0.95_area_all_maxdets_100": s[8],
        "AR_recall_iou_0.50:0.95_area_small_maxdets_100": s[9],
        "AR_recall_iou_0.50:0.95_area_medium_maxdets_100": s[10],
        "AR_recall_iou_0.50:0.95_area_large_maxdets_100": s[11],
    }


def parse_train_coco_metrics_from_log(log_path: Path) -> list[dict[str, Any]]:
    """parse_train_coco_metrics_from_log(log_path) -> list[dict]: Extract per-epoch COCO AP/AR rows from log.txt or stdout log."""
    log_path = Path(log_path)
    fold_dir = log_path
    # If log is in fold/logs/train_stdout.log, fold_dir is two levels up.
    if log_path.parent.name == "logs":
        fold_dir = log_path.parent.parent
    else:
        fold_dir = log_path.parent

    # Prefer fold_dir/log.txt (most reliable JSONL in DEIM)
    log_txt = fold_dir / "log.txt"
    sources: list[Path] = [log_txt] if log_txt.exists() else [log_path]

    rows: dict[int, dict[str, Any]] = {}

    for src in sources:
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            obj = _try_parse_json_line(line)
            if not obj:
                continue
            if "epoch" not in obj:
                continue
            try:
                ep = int(obj.get("epoch"))
            except Exception:
                continue

            r = rows.setdefault(ep, {"epoch": ep})

            # Preferred: explicit COCO stats list key (common in DEIM logs)
            for key in ["test_coco_eval_bbox", "val_coco_eval_bbox", "coco_eval_bbox"]:
                stats = obj.get(key)
                if isinstance(stats, (list, tuple)) and len(stats) >= 12:
                    r.update(_coco_stats_to_named_metrics(list(stats)))
                    break

            # Also accept already-named metric keys if present
            for k, v in obj.items():
                if k.startswith("AP_") or k.startswith("AR_") or k.startswith("AP_precision") or k.startswith("AR_recall"):
                    fv = safe_float(v)
                    if fv is not None:
                        r[k] = fv

    return [rows[k] for k in sorted(rows.keys())]


_LOSS_KV_RE = re.compile(r"\b(loss(?:_[A-Za-z0-9]+)*):\s*[-+0-9.eE]+\s*\(([-+0-9.eE]+)\)")

def parse_loss_curves_from_stdout_log(stdout_log_path: Path) -> list[dict[str, Any]]:
    """parse_loss_curves_from_stdout_log(stdout_log_path) -> list[dict]: Extract per-epoch train/val loss components from plain-text logs."""
    stdout_log_path = Path(stdout_log_path)
    if not stdout_log_path.exists():
        return []

    rows_by_ep: dict[int, dict[str, Any]] = {}
    current_ep: int | None = None
    in_test = False

    for line in stdout_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()

        m_ep = _EPOCH_RE.match(s)
        if m_ep:
            current_ep = int(m_ep.group(1))
            in_test = False
            rows_by_ep.setdefault(current_ep, {"epoch": current_ep})
            continue

        if s.startswith("Test:"):
            in_test = True
            continue

        if "Averaged stats:" not in s or current_ep is None:
            continue

        kvs = _LOSS_KV_RE.findall(s)
        if not kvs:
            continue

        prefix = "val_" if in_test else "train_"
        r = rows_by_ep[current_ep]
        for name, avg_str in kvs:
            fv = safe_float(avg_str)
            if fv is None:
                continue
            r[f"{prefix}{name}"] = float(fv)

    return [rows_by_ep[k] for k in sorted(rows_by_ep.keys())]

def parse_loss_curves_from_jsonl_logtxt(log_path: Path) -> list[dict[str, Any]]:
    """parse_loss_curves_from_jsonl_logtxt(log_path) -> list[dict]: Parse per-epoch train/val component losses from JSONL log.txt."""
    log_path = Path(log_path)

    # If user passed a fold dir, prefer fold/log.txt
    if log_path.is_dir():
        cand = log_path / "log.txt"
        if cand.exists():
            log_path = cand

    if log_path.name != "log.txt":
        # If a stdout log was passed, try sibling log.txt
        cand = log_path.parent / "log.txt"
        if cand.exists():
            log_path = cand

    if not log_path.exists():
        return []

    rows_by_ep: dict[int, dict[str, Any]] = {}
    txt = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def is_loss_key(k: str) -> bool:
        if not k.startswith("loss"):
            return False
        if any(x in k for x in ["dn_", "aux_", "enc_", "_pre", "pre_", "detr"]):
            return False
        return True

    for line in txt:
        obj = _try_parse_json_line(line)
        if not obj:
            continue
        if "epoch" not in obj:
            continue

        ep = safe_float(obj.get("epoch"))
        if ep is None:
            continue
        ep_i = int(ep)

        r = rows_by_ep.setdefault(ep_i, {"epoch": ep_i})

        # Determine whether this JSON line is train or val/test
        # Common pattern: train logs use "loss_*", test logs use "test_loss_*"
        is_val = any(k.startswith("test_") or k.startswith("val_") for k in obj.keys())

        if is_val:
            # Capture test_/val_ prefixed losses
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                if k.startswith("test_") and is_loss_key(k[len("test_"):]):
                    fv = safe_float(v)
                    if fv is not None:
                        r[f"val_{k[len('test_') :]}"] = float(fv)
                elif k.startswith("val_") and is_loss_key(k[len("val_"):]):
                    fv = safe_float(v)
                    if fv is not None:
                        r[f"val_{k[len('val_') :]}"] = float(fv)
        else:
            # Capture train losses (usually unprefixed "loss_*")
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                if is_loss_key(k):
                    fv = safe_float(v)
                    if fv is not None:
                        r[f"train_{k}"] = float(fv)

    return [rows_by_ep[k] for k in sorted(rows_by_ep.keys())]

def parse_loss_curves_from_log(log_path: Path) -> list[dict[str, Any]]:
    """parse_loss_curves_from_log(log_path) -> list[dict]: Parse loss curves from JSONL log.txt or plain-text stdout logs."""
    log_path = Path(log_path)

    rows = parse_loss_curves_from_jsonl_logtxt(log_path)
    if rows:
        return rows

    return parse_loss_curves_from_stdout_log(log_path)


def _loss_component_suffixes(row: dict[str, Any], prefix: str) -> set[str]:
    """_loss_component_suffixes(row, prefix) -> set[str]: Component-loss suffixes for a prefix (excludes total)."""
    total_key = f"{prefix}loss"
    suff: set[str] = set()
    for k in row.keys():
        if k.startswith(prefix) and "loss" in k and k != total_key:
            suff.add(k[len(prefix):])
    return suff


def _compute_total_loss(row: dict[str, Any], prefix: str, allowed_suffixes: set[str] | None = None) -> float | None:
    """_compute_total_loss(row, prefix, allowed_suffixes=None) -> float|None: Sum component losses (ignores total key)."""
    total_key = f"{prefix}loss"
    comps: list[float] = []
    for k, v in row.items():
        if not (k.startswith(prefix) and "loss" in k and k != total_key):
            continue
        suffix = k[len(prefix):]
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        fv = safe_float(v)
        if fv is not None:
            comps.append(float(fv))
    if comps:
        return float(sum(comps))
    return None


def plot_metric_over_epoch(out_path: Path, title: str, rows: list[dict[str, Any]], key: str) -> None:
    """plot_metric_over_epoch(out_path, title, rows, key) -> None: Line plot for a metric vs epoch."""
    xs: list[int] = []
    ys: list[float] = []
    for r in rows:
        ep = r.get("epoch")
        v = r.get(key)
        if ep is None:
            continue
        fv = safe_float(v)
        if fv is None:
            continue
        xs.append(int(ep))
        ys.append(float(fv))

    if not xs:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()

def plot_train_val_loss(out_path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    """plot_train_val_loss(out_path, title, rows) -> None: Plot train vs val loss as identical sum over val components."""
    # Determine which component keys exist in validation across the run
    # We only sum these, for BOTH train and val.
    val_suffixes: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k.startswith("val_loss") and k != "val_loss":
                val_suffixes.add(k[len("val_"):])  # e.g. "loss_bbox"

    # Fallback: if val only logged total (rare for JSONL), try the classic trio
    if not val_suffixes:
        val_suffixes = {"loss_mal", "loss_bbox", "loss_giou"}

    xs: list[int] = []
    tr: list[float] = []
    va: list[float] = []

    for r in rows:
        ep = r.get("epoch")
        if ep is None:
            continue

        train_total = 0.0
        val_total = 0.0
        have_train = False
        have_val = False

        for suf in sorted(val_suffixes):
            tv = safe_float(r.get(f"train_{suf}"))
            vv = safe_float(r.get(f"val_{suf}"))
            if tv is not None:
                train_total += float(tv)
                have_train = True
            if vv is not None:
                val_total += float(vv)
                have_val = True

        if not (have_train and have_val):
            continue

        xs.append(int(ep))
        tr.append(train_total)
        va.append(val_total)

    if not xs:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, tr, label="train")
    plt.plot(xs, va, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss (sum of val components)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()

def write_train_metrics_xlsx(fold_dir: Path, coco_rows: list[dict[str, Any]]) -> None:
    """write_train_metrics_xlsx(fold_dir, coco_rows) -> None: Write train metrics to fold_dir/metrics/train_metrics.xlsx."""
    fold_dir = Path(fold_dir)
    out = fold_dir / "metrics" / "train_metrics.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    seen: set[str] = set()
    for r in coco_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    # epoch first
    keys = ["epoch"] + [k for k in keys if k != "epoch"]

    wb = Workbook()
    ws = wb.active
    ws.title = "train_metrics"
    ws.append(keys)
    for r in coco_rows:
        ws.append([r.get(k, "") for k in keys])

    for ci, c in enumerate(keys, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = min(60, max(10, len(str(c)) + 2))
    wb.save(str(out))


def write_loss_xlsx(fold_dir: Path, loss_rows: list[dict[str, Any]]) -> None:
    """write_loss_xlsx(fold_dir, loss_rows) -> None: Write loss curves to fold_dir/metrics/loss_curves.xlsx."""
    fold_dir = Path(fold_dir)
    out = fold_dir / "metrics" / "loss_curves.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    seen: set[str] = set()
    for r in loss_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    keys = ["epoch"] + [k for k in keys if k != "epoch"]

    wb = Workbook()
    ws = wb.active
    ws.title = "loss_curves"
    ws.append(keys)
    for r in loss_rows:
        ws.append([r.get(k, "") for k in keys])

    for ci, c in enumerate(keys, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = min(60, max(10, len(str(c)) + 2))
    wb.save(str(out))

