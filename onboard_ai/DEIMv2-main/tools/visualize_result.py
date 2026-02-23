#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image, ImageDraw

# =======================
# CONFIG
# =======================

MODEL = "final"          # "final" or "fold1"
SPLIT = "validation"     # "validation" or "test"

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
    return Path(__file__).resolve().parents[1]


def resolve_results_dir(repo_root: Path) -> Path:
    p = repo_root / "results" / "default_model_S"
    if not p.exists():
        raise RuntimeError(f"RESULTS_DIR not found: {p}")
    return p


def resolve_img_root(repo_root: Path) -> Path:
    p = repo_root / "data" / "0_merged" / "reflection_offnadir_glint_255"
    if not p.exists():
        raise RuntimeError(f"IMG_ROOT not found: {p}")
    return p


def resolve_model_dir(results_dir: Path, model: str) -> Path:
    if model.lower() == "final":
        p = results_dir / "final_location_holdout"
    else:
        p = results_dir / "cross_validation" / model
    if not p.exists():
        raise RuntimeError(f"model dir not found: {p}")
    return p


def resolve_ann_path(results_dir: Path, model_dir: Path, split: str) -> Path:
    if split == "validation":
        p = model_dir / "splits" / "instances_val.json"
        if p.exists():
            return p
    if split == "test":
        p = results_dir / "test_holdout_only.json"
        if p.exists():
            return p
    raise RuntimeError("Annotation file not found.")


def resolve_pred_path(results_dir: Path, model_dir: Path, split: str) -> Path:
    if USE_OVERVIEW_PREDICTIONS:
        p = results_dir / "overview" / split / "predictions" / f"{OVERVIEW_TAG}_predictions_coco.json"
        if p.exists():
            return p

    p = model_dir / "metrics" / split / "predictions_coco.json"
    if p.exists():
        return p

    eval_dir = model_dir / ("eval_val" if split == "validation" else "eval_test") / "eval_data"
    p2 = eval_dir / "predictions_coco.json"
    if p2.exists():
        return p2

    raise RuntimeError("predictions file not found")


def figures_dir() -> Path:
    base = Path(__file__).resolve().parent / "figures"
    out = base / f"{MODEL}_{SPLIT}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    repo_root = resolve_repo_root()
    results_dir = resolve_results_dir(repo_root)
    img_root = resolve_img_root(repo_root)

    model_dir = resolve_model_dir(results_dir, MODEL)
    ann_path = resolve_ann_path(results_dir, model_dir, SPLIT)
    pred_path = resolve_pred_path(results_dir, model_dir, SPLIT)

    coco = read_json(ann_path)
    preds = read_json(pred_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    id_to_file = {int(im["id"]): im["file_name"] for im in images}

    out_dir = figures_dir()

    print(f"Processing {len(images)} images...")

    for idx, im_info in enumerate(images):
        image_id = int(im_info["id"])
        file_name = im_info["file_name"]
        img_path = img_root / file_name

        if not img_path.exists():
            print(f"[{idx}] Missing image file: {img_path}")
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
                if sc >= SCORE_THRESHOLD:
                    pred_boxes.append((sc, xywh_to_xyxy(p["bbox"])))

        pred_boxes.sort(key=lambda t: t[0], reverse=True)
        pred_boxes = pred_boxes[:MAX_PREDS]

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)

        matched_gt = set()

        # 1) Draw all ground truth first (background layer)
        for gb in gt_boxes:
            draw_box(draw, gb, "green", 4)

        # 2) Then draw predictions on top
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

            if best >= IOU_THRESHOLD and best_i >= 0:
                matched_gt.add(best_i)
                draw_box(draw, pb, "blue", 2)  # TP on top
            else:
                draw_box(draw, pb, "red", 2)  # FP on top

        out_path = out_dir / f"image_{image_id}.png"
        im.save(out_path)

        print(f"[{idx+1}/{len(images)}] saved: {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()