import json
import os
import numpy as np
from PIL import Image, ImageDraw
from . import image_utils as iu

def load_coco_index(annotations_path: str) -> tuple[dict, dict, dict]:
    """Load COCO JSON and return (by_file, anns_by_image_id, images_by_id)."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    by_file = {im["file_name"].replace("\\", "/"): im["id"] for im in coco["images"]}

    anns_by_image_id = {}
    for ann in coco["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    return by_file, anns_by_image_id, images_by_id


def coco_segmentation_to_mask(height: int, width: int, segmentation: list) -> np.ndarray:
    """Rasterize COCO polygon segmentation into a boolean mask (True=inside polygon)."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # segmentation is usually [ [x1,y1,x2,y2,...], [ ... ], ... ]
    for poly in segmentation:
        if not poly or len(poly) < 6:
            continue
        xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(xy, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8).astype(bool)


def get_whale_mask_for_image(img_rgb_uint8: np.ndarray, img_file_name: str, by_file: dict, anns_by_image_id: dict, images_by_id: dict, whale_category_id: int = 0) -> np.ndarray:
    """Return whale mask for img_file_name using COCO annotations (True=whale)."""
    key = img_file_name.replace("\\", "/")
    if key not in by_file:
        raise KeyError(f"Image file_name not found in annotations: {img_file_name}")

    image_id = by_file[key]
    h = images_by_id[image_id]["height"]
    w = images_by_id[image_id]["width"]

    # Safety: if your loaded image got cropped, you must crop the mask the same way (not handled here)
    if img_rgb_uint8.shape[0] != h or img_rgb_uint8.shape[1] != w:
        raise ValueError(f"Image size mismatch: annotations ({h},{w}) vs loaded ({img_rgb_uint8.shape[0]},{img_rgb_uint8.shape[1]}). If you crop borders, crop the mask identically.")

    whale_mask = np.zeros((h, w), dtype=bool)
    for ann in anns_by_image_id.get(image_id, []):
        if ann.get("iscrowd", 0) != 0:
            continue
        if ann.get("category_id", None) != whale_category_id:
            continue
        whale_mask |= coco_segmentation_to_mask(h, w, ann["segmentation"])

    return whale_mask


def rgb_png_to_reflectance_proxy(img_rgb_uint8: np.ndarray, anchor_mask: np.ndarray, target_reflectance_rgb: tuple[float, float, float] = (0.04, 0.03, 0.02)) -> np.ndarray:
    """Scale linear RGB so median(anchor_mask) matches target_reflectance_rgb; returns HxWx3 float32 in [0,1]."""
    img_lin = iu.DN255_to_linear(img_rgb_uint8).astype(np.float32)  # HxWx3
    mask = anchor_mask.astype(bool)

    if mask.sum() < 50:
        raise ValueError("Anchor mask too small to estimate medians.")

    tgt = np.array(target_reflectance_rgb, dtype=np.float32)
    out = img_lin.copy()

    for c in range(3):
        med = float(np.median(img_lin[:, :, c][mask]))
        s = float(tgt[c] / (med + 1e-12))
        out[:, :, c] = np.clip(img_lin[:, :, c] * s, 0.0, 1.0)

    return out

import numpy as np


def _black_mask_raw(dn255_rgb, tol=0):
    """Return HxW bool: True where all channels <= tol."""
    img = np.asarray(dn255_rgb)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img16 = img.astype(np.int16)  # avoid uint8 comparisons surprises
    t = np.array([tol, tol, tol], dtype=np.int16).reshape(1, 1, 3)
    return np.all(img16 <= t, axis=-1)


def _first_last_true_row(mask_bool):
    """Return (y0,y1) for first/last row that has any True; (None,None) if empty."""
    rows = np.any(mask_bool, axis=1)
    idx = np.flatnonzero(rows)
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def _row_interval(mask_row):
    """Return (xL,xR) of True pixels in a row; (None,None) if empty."""
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def compute_hit_mask_full(dn255_blackproj, tol=0, row_black_frac_keep=0.80, min_row_black_frac_abs=0.02, width_keep_frac=0.40, interval_mode="median"):
    """Build footprint mask from black-projection DN255; keeps only full rows (top/bottom trims), fills rows as one rectangle."""
    mask_raw = _black_mask_raw(dn255_blackproj, tol=tol)
    H, W = mask_raw.shape

    y0, y1 = _first_last_true_row(mask_raw)
    if y0 is None:
        empty = np.zeros((H, W), dtype=bool)
        return empty, mask_raw, None, None, None, None, {"row_black_frac": np.zeros(H), "max_row_black_frac": 0.0, "widths": []}

    # Row black fractions across the initial span
    row_black_frac = mask_raw[y0:y1 + 1].mean(axis=1).astype(np.float32)  # fraction of row that is black
    max_row_black_frac = float(np.max(row_black_frac)) if row_black_frac.size else 0.0

    # Candidate rows must be "black enough" relative to the best row, and also not trivially tiny
    rel_thr = float(row_black_frac_keep) * max_row_black_frac
    abs_thr = float(min_row_black_frac_abs)
    row_is_candidate = (row_black_frac >= rel_thr) & (row_black_frac >= abs_thr)

    # Build intervals/widths from candidate rows
    intervals = {}
    widths = []
    for i, y in enumerate(range(y0, y1 + 1)):
        if not row_is_candidate[i]:
            continue
        xL, xR = _row_interval(mask_raw[y, :])
        if xL is None:
            continue
        w = xR - xL + 1
        intervals[y] = (xL, xR)
        widths.append(w)

    if len(widths) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, mask_raw, None, None, None, None, {"row_black_frac": row_black_frac, "max_row_black_frac": max_row_black_frac, "widths": []}

    med_width = float(np.median(widths))

    # Choose a stable rectangle x-interval from candidate rows
    xLs = np.array([v[0] for v in intervals.values()], dtype=np.float32)
    xRs = np.array([v[1] for v in intervals.values()], dtype=np.float32)
    if interval_mode == "max":
        xL_rect = int(np.min(xLs))
        xR_rect = int(np.max(xRs))
    else:  # "median" (default)
        xL_rect = int(np.median(xLs))
        xR_rect = int(np.median(xRs))

    xL_rect = max(xL_rect, 0)
    xR_rect = min(xR_rect, W - 1)
    if xL_rect > xR_rect:
        xL_rect, xR_rect = 0, W - 1

    # Top trim: drop rows until width is large enough (relative to median width) and row is candidate
    kept_y0 = y0
    while kept_y0 <= y1:
        if kept_y0 in intervals:
            xL, xR = intervals[kept_y0]
            w = (xR - xL + 1)
            if w >= float(width_keep_frac) * med_width:
                break
        kept_y0 += 1

    # Bottom trim
    kept_y1 = y1
    while kept_y1 >= kept_y0:
        if kept_y1 in intervals:
            xL, xR = intervals[kept_y1]
            w = (xR - xL + 1)
            if w >= float(width_keep_frac) * med_width:
                break
        kept_y1 -= 1

    if kept_y0 > kept_y1:
        empty = np.zeros((H, W), dtype=bool)
        return empty, mask_raw, None, None, xL_rect, xR_rect, {"row_black_frac": row_black_frac, "max_row_black_frac": max_row_black_frac, "widths": widths}

    # Build final mask: ONLY full rows, filled with ONE rectangle interval (prevents partial rows)
    mask_full = np.zeros((H, W), dtype=bool)
    for y in range(kept_y0, kept_y1 + 1):
        mask_full[y, xL_rect:xR_rect + 1] = True

    dbg = {
        "row_black_frac": row_black_frac,
        "max_row_black_frac": max_row_black_frac,
        "widths": widths,
        "med_width": med_width,
        "rel_thr": rel_thr,
        "abs_thr": abs_thr,
        "kept_y0": kept_y0,
        "kept_y1": kept_y1,
        "xL_rect": xL_rect,
        "xR_rect": xR_rect,
    }
    return mask_full, mask_raw, kept_y0, kept_y1, xL_rect, xR_rect, dbg
