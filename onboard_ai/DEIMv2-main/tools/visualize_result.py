#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image, ImageDraw


# =======================
# CONFIG – EDIT THESE
# =======================

RESULTS_DIR = Path("results/default_model_S")
IMG_ROOT = Path("data/0_merged/reflection_offnadir_glint_255")

MODEL = "final"          # "final" or "fold1"
SPLIT = "validation"     # "validation" or "test"
IMAGE_ID = 12

SCORE_THRESHOLD = 0.3
MAX_PREDS = 50
IOU_THRESHOLD = 0.5

# =======================


def read_json(path: Path) -> Any:
    """read_json(path) -> Any: Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    """xywh_to_xyxy(b) -> (x1,y1,x2,y2)."""
    x, y, w, h = b
    return x, y, x + w, y + h


def iou(boxA, boxB) -> float:
    """iou(boxA, boxB) -> float: IoU of two xyxy boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def resolve_model_dir(results_dir: Path, model: str) -> Path:
    """Resolve final_location_holdout or foldX directory."""
    if model == "final":
        return results_dir / "final_location_holdout"
    return results_dir / "cross_validation" / model


def resolve_pred_path(model_dir: Path, split: str) -> Path:
    """Find predictions_coco.json."""
    p = model_dir / "metrics" / split / "predictions_coco.json"
    if p.exists():
        return p

    # fallback rank0
    p = model_dir / "metrics" / split / "predictions_coco.json.rank0.json"
    if p.exists():
        return p

    raise RuntimeError("predictions file not found")


def resolve_ann_path(results_dir: Path, model_dir: Path, split: str) -> Path:
    """Find correct annotation JSON."""
    if split == "validation":
        return model_dir / "splits" / "instances_val.json"

    if split == "test":
        p = results_dir / "test_holdout_only.json"
        if p.exists():
            return p

    raise RuntimeError("annotation file not found")


def draw_box(draw: ImageDraw.ImageDraw, box, color, width=3):
    """Draw rectangle."""
    x1, y1, x2, y2 = box
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)


def main():
    results_dir = RESULTS_DIR.resolve()
    img_root = IMG_ROOT.resolve()

    model_dir = resolve_model_dir(results_dir, MODEL)
    pred_path = resolve_pred_path(model_dir, SPLIT)
    ann_path = resolve_ann_path(results_dir, model_dir, SPLIT)

    coco = read_json(ann_path)
    preds = read_json(pred_path)

    id_to_file = {int(im["id"]): im["file_name"] for im in coco["images"]}
    file_name = id_to_file.get(int(IMAGE_ID))
    if not file_name:
        raise RuntimeError("image_id not found")

    img_path = img_root / file_name
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Collect GT
    gt_boxes = []
    for a in coco["annotations"]:
        if int(a["image_id"]) == IMAGE_ID:
            gt_boxes.append(xywh_to_xyxy(a["bbox"]))

    # Collect predictions
    pred_boxes = []
    for p in preds:
        if int(p["image_id"]) == IMAGE_ID and p["score"] >= SCORE_THRESHOLD:
            pred_boxes.append((p["score"], xywh_to_xyxy(p["bbox"])))

    pred_boxes.sort(key=lambda x: x[0], reverse=True)
    pred_boxes = pred_boxes[:MAX_PREDS]

    # IoU matching (greedy)
    matched_gt = set()

    for score, pb in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou = v
                best_gt_idx = i

        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_gt_idx)
            draw_box(draw, pb, "blue", width=3)   # TP
        else:
            draw_box(draw, pb, "red", width=2)    # FP

        x1, y1, _, _ = pb
        draw.text((x1 + 2, y1 + 2), f"{score:.2f}", fill="white")

    # Draw GT
    for gb in gt_boxes:
        draw_box(draw, gb, "green", width=4)

    out_path = pred_path.parent / f"overlay_{MODEL}_{SPLIT}_image{IMAGE_ID}.png"
    image.save(out_path)

    print("WROTE:", out_path)


if __name__ == "__main__":
    main()