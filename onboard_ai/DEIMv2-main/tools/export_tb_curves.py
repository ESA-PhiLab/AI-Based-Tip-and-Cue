#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


Point = Tuple[int, float]


def _event_dirs(root: Path) -> List[Path]:
    """_event_dirs(root) -> list[Path]: Find all directories under root that contain TensorBoard event files."""
    root = root.resolve()
    dirs: set[Path] = set()
    for p in root.rglob("events.out.tfevents.*"):
        if p.is_file():
            dirs.add(p.parent)
    return sorted(dirs)


def _load_scalars_from_dir(logdir: Path) -> Dict[str, List[Point]]:
    """_load_scalars_from_dir(logdir) -> dict[str,list[(step,value)]]: Load all scalar series from one TB logdir."""
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    out: Dict[str, List[Point]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]
    return out


def _merge_series(a: List[Point], b: List[Point]) -> List[Point]:
    """_merge_series(a, b) -> list[(step,value)]: Merge two series by step (b overwrites a on conflicts)."""
    m = {int(s): float(v) for s, v in a}
    for s, v in b:
        m[int(s)] = float(v)
    return sorted(m.items(), key=lambda x: x[0])


def read_all_scalars(root: Path) -> Dict[str, List[Point]]:
    """read_all_scalars(root) -> dict[str,list[(step,value)]]: Read scalars from all event dirs under root."""
    merged: Dict[str, List[Point]] = {}
    for d in _event_dirs(root):
        try:
            series = _load_scalars_from_dir(d)
        except Exception:
            continue
        for tag, pts in series.items():
            if tag in merged:
                merged[tag] = _merge_series(merged[tag], pts)
            else:
                merged[tag] = pts
    return merged


def save_csv(path: Path, series: Dict[str, List[Point]]) -> None:
    """save_csv(path, series) -> None: Save series to CSV with columns name,step,value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("name,step,value\n")
        for name, pts in series.items():
            for step, val in pts:
                f.write(f"{name},{step},{val}\n")


def _plot(ax, pts: List[Point], label: str) -> None:
    """_plot(ax, pts, label) -> None: Plot one series on an axis."""
    xs = [s for s, _ in pts]
    ys = [v for _, v in pts]
    ax.plot(xs, ys, label=label)


def _norm_tag(s: str) -> str:
    """_norm_tag(s) -> str: Normalize tag for matching."""
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def _first_scalar_tag(logdir: Path) -> str | None:
    """_first_scalar_tag(logdir) -> str|None: Return the first scalar tag in this TB directory (or None)."""
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = list(ea.Tags().get("scalars", []))
    return tags[0] if tags else None


def _read_series(logdir: Path, tag: str) -> List[Point]:
    """_read_series(logdir, tag) -> list[(step,value)]: Read one scalar series."""
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]


def _pick_ap_tags(all_tags: List[str]) -> tuple[str | None, str | None, str | None]:
    """_pick_ap_tags(all_tags) -> (ap5095, ap50, ap75): Pick COCO AP tags if present."""
    norm = {t: _norm_tag(t) for t in all_tags}

    ap5095 = None
    ap50 = None
    ap75 = None

    # Prefer explicit 0.50:0.95, then generic AP that isn't 50/75
    for t, n in norm.items():
        if "coco" in n and ("ap50_95" in n or "ap_50_95" in n or "ap_050_095" in n or "ap_0.50:0.95" in n):
            ap5095 = t
            break
    if ap5095 is None:
        for t, n in norm.items():
            if "coco" in n and "ap" in n and "ap50" not in n and "ap_50" not in n and "ap75" not in n and "ap_75" not in n:
                ap5095 = t
                break

    for t, n in norm.items():
        if "coco" in n and ("ap50" in n or "ap_50" in n):
            ap50 = t
            break

    for t, n in norm.items():
        if "coco" in n and ("ap75" in n or "ap_75" in n):
            ap75 = t
            break

    return ap5095, ap50, ap75


def _choose_logroot(logdir: Path) -> Path:
    """_choose_logroot(logdir) -> Path: Prefer logdir/summary if it exists, else logdir."""
    cand = logdir / "summary"
    return cand if cand.exists() and cand.is_dir() else logdir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=Path, required=True, help="Run directory (or summary directory).")
    p.add_argument("--outdir", type=Path, default=Path("tb_exports"), help="Output folder for PNG/CSV.")
    p.add_argument("--print_tags", action="store_true", help="Print all scalar tags found and exit.")
    args = p.parse_args()

    logroot = _choose_logroot(args.logdir)
    all_series = read_all_scalars(logroot)
    all_tags = sorted(all_series.keys())

    if args.print_tags:
        print(json.dumps(all_tags, indent=2))
        return

    extracted: Dict[str, List[Point]] = {}

    # ---- Your exact DEIM fold layout writes losses into:
    # summary/Loss_total_epoch_train/...
    # summary/Loss_total_epoch_val/...
    train_dir = logroot / "Loss_total_epoch_train"
    val_dir = logroot / "Loss_total_epoch_val"

    train_tag = _first_scalar_tag(train_dir) if train_dir.exists() else None
    val_tag = _first_scalar_tag(val_dir) if val_dir.exists() else None

    if train_tag:
        extracted["train_loss_epoch"] = _read_series(train_dir, train_tag)
    if val_tag:
        extracted["val_loss_epoch"] = _read_series(val_dir, val_tag)

    # ---- Optional: AP only if it exists anywhere in the TB logs ----
    ap5095_tag, ap50_tag, ap75_tag = _pick_ap_tags(all_tags)

    if ap5095_tag:
        extracted["val_AP_0.50:0.95"] = all_series.get(ap5095_tag, [])
    if ap50_tag:
        extracted["val_AP_0.50"] = all_series.get(ap50_tag, [])
    if ap75_tag:
        extracted["val_AP_0.75"] = all_series.get(ap75_tag, [])

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_csv(args.outdir / "scalars.csv", extracted)

    if "train_loss_epoch" in extracted and extracted["train_loss_epoch"]:
        fig, ax = plt.subplots()
        _plot(ax, extracted["train_loss_epoch"], "train_total")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Train Total Loss (epoch)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "train_loss_epoch.png", dpi=200)
        plt.close(fig)

    if "val_loss_epoch" in extracted and extracted["val_loss_epoch"]:
        fig, ax = plt.subplots()
        _plot(ax, extracted["val_loss_epoch"], "val_total")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Validation Total Loss (epoch)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "val_loss_total_epoch.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots()
    any_loss = False
    if "train_loss_epoch" in extracted and extracted["train_loss_epoch"]:
        _plot(ax, extracted["train_loss_epoch"], "train")
        any_loss = True
    if "val_loss_epoch" in extracted and extracted["val_loss_epoch"]:
        _plot(ax, extracted["val_loss_epoch"], "val")
        any_loss = True
    if any_loss:
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Train vs Val Total Loss (epoch)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "train_val_loss_epoch.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    any_ap = False
    for key, label in [
        ("val_AP_0.50:0.95", "AP_0.50:0.95"),
        ("val_AP_0.50", "AP_0.50"),
        ("val_AP_0.75", "AP_0.75"),
    ]:
        pts = extracted.get(key, [])
        if pts:
            _plot(ax, pts, label)
            any_ap = True
    if any_ap:
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.set_title("Validation COCO AP (epoch)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / "val_coco_ap_epoch.png", dpi=200)
    plt.close(fig)

    print("Detected:")
    print(f"  train_loss_dir: {train_dir} | tag: {train_tag}")
    print(f"  val_loss_dir  : {val_dir} | tag: {val_tag}")
    print(f"  ap_0.50:0.95_tag: {ap5095_tag}")
    print(f"  ap_0.50_tag    : {ap50_tag}")
    print(f"  ap_0.75_tag    : {ap75_tag}")

    print("\nWrote:")
    print(f"  - {args.outdir / 'scalars.csv'}")
    if (args.outdir / "train_loss_epoch.png").exists():
        print(f"  - {args.outdir / 'train_loss_epoch.png'}")
    if (args.outdir / "val_loss_total_epoch.png").exists():
        print(f"  - {args.outdir / 'val_loss_total_epoch.png'}")
    if (args.outdir / "train_val_loss_epoch.png").exists():
        print(f"  - {args.outdir / 'train_val_loss_epoch.png'}")
    if (args.outdir / "val_coco_ap_epoch.png").exists():
        print(f"  - {args.outdir / 'val_coco_ap_epoch.png'}")


if __name__ == "__main__":
    main()