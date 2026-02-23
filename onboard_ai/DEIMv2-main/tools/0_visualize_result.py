#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image, ImageDraw

# =======================
# DEFAULTS (override via CLI)
# =======================

DEFAULT_RESULTS_FOLDER = "test_model_S"
DEFAULT_MODEL = "final"          # "final" or "fold1"
DEFAULT_SPLIT = "validation"           # "validation" or "test"

SCORE_THRESHOLD = 0.4
MAX_PREDS = 50
IOU_THRESHOLD = 0.5

USE_OVERVIEW_PREDICTIONS = False
OVERVIEW_TAG = "FINAL"

# =======================


def read_json(path: Path) -> Any:
    """read_json(path) -> Any: Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    """xywh_to_xyxy(b) -> (x1,y1,x2,y2)."""
    x, y, w, h = [float(v) for v in b]
    return x, y, x + w, y + h


def iou(a, b) -> float:
    """iou(a, b) -> float: IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def draw_box(draw: ImageDraw.ImageDraw, box, color, width: int) -> None:
    """draw_box(draw, box, color, width) -> None."""
    x1, y1, x2, y2 = box
    for i in range(int(width)):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)


def resolve_repo_root() -> Path:
    """resolve_repo_root() -> Path: Infer repo root as parent of tools/."""
    return Path(__file__).resolve().parents[1]


def resolve_results_dir(repo_root: Path, results_folder: str) -> Path:
    """resolve_results_dir(repo_root, results_folder) -> Path: results/<results_folder>."""
    p = repo_root / "results" / results_folder
    if not p.exists():
        raise RuntimeError(f"RESULTS_DIR not found: {p}")
    return p


def resolve_img_root(repo_root: Path) -> Path:
    """resolve_img_root(repo_root) -> Path: data/0_merged/... under repo root."""
    p = repo_root / "data" / "0_merged" / "reflection_offnadir_glint_255"
    if not p.exists():
        raise RuntimeError(f"IMG_ROOT not found: {p}")
    return p


def resolve_model_dir(results_dir: Path, model: str) -> Path:
    """resolve_model_dir(results_dir, model) -> Path."""
    if model.lower() == "final":
        p = results_dir / "final_location_holdout"
    else:
        p = results_dir / "cross_validation" / model
    if not p.exists():
        raise RuntimeError(f"model dir not found: {p}")
    return p


def resolve_ann_path(results_dir: Path, model_dir: Path, split: str) -> Path:
    """resolve_ann_path(results_dir, model_dir, split) -> Path."""
    if split == "validation":
        p = model_dir / "splits" / "instances_val.json"
        if p.exists():
            return p
        raise RuntimeError(f"validation annotations not found: {p}")

    if split == "test":
        p = results_dir / "test_holdout_only.json"
        if p.exists():
            return p
        raise RuntimeError(f"test annotations not found: {p}")

    raise RuntimeError("split must be validation or test")


def resolve_pred_path(results_dir: Path, model_dir: Path, split: str, model: str) -> Path:
    """resolve_pred_path(results_dir, model_dir, split, model) -> Path."""
    if USE_OVERVIEW_PREDICTIONS:
        p = results_dir / "overview" / split / "predictions" / f"{OVERVIEW_TAG}_predictions_coco.json"
        if p.exists():
            return p
        raise RuntimeError(f"overview predictions not found: {p}")

    p = model_dir / "metrics" / split / "predictions_coco.json"
    if p.exists():
        return p

    eval_dir = model_dir / ("eval_val" if split == "validation" else "eval_test") / "eval_data"
    p2 = eval_dir / "predictions_coco.json"
    if p2.exists():
        return p2

    p3 = results_dir / "overview" / split / "predictions" / f"{('FINAL' if model == 'final' else model)}_predictions_coco.json"
    if p3.exists():
        return p3

    raise RuntimeError("predictions file not found (checked metrics/, eval_data/, overview/)")


def figures_dir(repo_root: Path, results_folder: str, model: str, split: str) -> Path:
    base = repo_root / "figures"
    out = base / results_folder / f"{model}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS_FOLDER, help="results folder, e.g. default_model_S")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help='model dir: "final" or "fold1" etc')
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["validation", "test"], help="split")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    results_dir = resolve_results_dir(repo_root, args.results)
    img_root = resolve_img_root(repo_root)

    model_dir = resolve_model_dir(results_dir, args.model)
    ann_path = resolve_ann_path(results_dir, model_dir, args.split)
    pred_path = resolve_pred_path(results_dir, model_dir, args.split, args.model)

    coco = read_json(ann_path)
    preds = read_json(pred_path)
    if not isinstance(preds, list):
        raise RuntimeError(f"predictions must be a list, got {type(preds)} from {pred_path}")

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not images:
        raise RuntimeError(f'No "images" in annotations file: {ann_path}')

    out_dir = figures_dir(repo_root, args.results, args.model, args.split)

    print("ANN :", ann_path)
    print("PRED:", pred_path)
    print("OUT :", out_dir)
    print(f"Processing {len(images)} images...")

    for idx, im_info in enumerate(images):
        image_id = int(im_info["id"])
        file_name = str(im_info["file_name"])
        img_path = img_root / file_name

        if not img_path.exists():
            print(f"[{idx+1}/{len(images)}] missing image file: {img_path}")
            continue

        gt_boxes = [
            xywh_to_xyxy(a["bbox"])
            for a in annotations
            if int(a.get("image_id")) == image_id
        ]

        pred_boxes = []
        for p in preds:
            if int(p.get("image_id")) == image_id:
                sc = float(p.get("score", 0.0))
                if sc >= float(SCORE_THRESHOLD):
                    pred_boxes.append((sc, xywh_to_xyxy(p["bbox"])))
        pred_boxes.sort(key=lambda t: t[0], reverse=True)
        pred_boxes = pred_boxes[: int(MAX_PREDS)]

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)

        # Draw GT first
        for gb in gt_boxes:
            draw_box(draw, gb, "green", 4)

        # Draw predictions on top (blue TP, red FP)
        matched_gt = set()
        for sc, pb in pred_boxes:
            best_i = -1
            best = 0.0
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                v = iou(pb, gb)
                if v > best:
                    best = v
                    best_i = i

            if best >= float(IOU_THRESHOLD) and best_i >= 0:
                matched_gt.add(best_i)
                draw_box(draw, pb, "blue", 2)
            else:
                draw_box(draw, pb, "red", 2)

        original_name = Path(file_name).name
        out_path = out_dir / original_name
        im.save(out_path)

        print(f"[{idx+1}/{len(images)}] saved: {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()