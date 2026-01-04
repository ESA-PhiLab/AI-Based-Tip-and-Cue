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
    """Return HxW bool where pixel is 'black' if all channels <= tol."""
    img = dn255_rgb.astype(np.int16)
    t = np.array([tol, tol, tol], dtype=np.int16).reshape(1, 1, 3)
    return np.all(img <= t, axis=2)


def _first_last_true_row(mask_bool):
    """Return (y0, y1) first/last row containing any True; (None, None) if empty."""
    rows = np.any(mask_bool, axis=1)
    idx = np.flatnonzero(rows)
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def _row_interval(mask_row):
    """Return (xL, xR) of True pixels; (None, None) if empty."""
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return None, None
    return int(idx[0]), int(idx[-1])


def compute_hit_mask_full(dn255_blackproj, tol=20, min_row_px=50, row_black_frac_keep=0.8, width_keep_frac=0.6):
    """
    Footprint mask from black-projection render (footprint=black).

    Rules:
      - pixel is black if ALL channels <= tol
      - trim ONLY from top/bottom (never remove interior rows)
      - keep a row if (black_fraction_in_row_span >= row_black_frac_keep)
        AND (row_span_width >= width_keep_frac * median_span_width)
      - final mask is filled per row as one contiguous horizontal interval

    Returns:
      mask_full : HxW bool  (filled per row, no holes)
      mask_raw  : HxW bool  (raw black pixels)
      kept_y0, kept_y1 : ints or (None, None)
      med_xL, med_xR   : ints (median interval from strong rows)
      row_black_frac   : H float (fraction black inside row span; nan if no span)
      row_width        : H int (span width; 0 if none)
    """
    mask_raw = _black_mask_raw(dn255_blackproj, tol=tol)
    H, W = mask_raw.shape

    y0, y1 = _first_last_true_row(mask_raw)
    if y0 is None:
        row_black_frac = np.full((H,), np.nan, dtype=np.float32)
        row_width = np.zeros((H,), dtype=np.int32)
        return np.zeros((H, W), dtype=bool), mask_raw, None, None, 0, W - 1, row_black_frac, row_width

    # Per-row spans + black fractions inside span
    xL_row = np.full((H,), -1, dtype=np.int32)
    xR_row = np.full((H,), -1, dtype=np.int32)
    row_width = np.zeros((H,), dtype=np.int32)
    row_black_frac = np.full((H,), np.nan, dtype=np.float32)

    strong_rows = []
    widths = []
    xLs = []
    xRs = []

    for y in range(y0, y1 + 1):
        xL, xR = _row_interval(mask_raw[y, :])
        if xL is None:
            continue

        span_w = xR - xL + 1
        black_in_span = int(mask_raw[y, xL:xR + 1].sum())
        frac = black_in_span / float(span_w)

        xL_row[y] = xL
        xR_row[y] = xR
        row_width[y] = span_w
        row_black_frac[y] = float(frac)

        # "strong" rows define the stable median width/interval
        if black_in_span >= int(min_row_px):
            strong_rows.append(y)
            widths.append(span_w)
            xLs.append(xL)
            xRs.append(xR)

    if len(widths) == 0:
        return np.zeros((H, W), dtype=bool), mask_raw, None, None, 0, W - 1, row_black_frac, row_width

    med_width = float(np.median(widths))
    med_xL = int(np.median(xLs))
    med_xR = int(np.median(xRs))
    med_xL = max(med_xL, 0)
    med_xR = min(med_xR, W - 1)
    if med_xL > med_xR:
        med_xL, med_xR = 0, W - 1

    # --- Trim only top/bottom based on row quality ---
    kept_y0 = y0
    while kept_y0 <= y1:
        if row_width[kept_y0] <= 0 or np.isnan(row_black_frac[kept_y0]):
            kept_y0 += 1
            continue

        ok_frac = row_black_frac[kept_y0] >= float(row_black_frac_keep)
        ok_w = row_width[kept_y0] >= float(width_keep_frac) * med_width
        if ok_frac and ok_w:
            break
        kept_y0 += 1

    kept_y1 = y1
    while kept_y1 >= kept_y0:
        if row_width[kept_y1] <= 0 or np.isnan(row_black_frac[kept_y1]):
            kept_y1 -= 1
            continue

        ok_frac = row_black_frac[kept_y1] >= float(row_black_frac_keep)
        ok_w = row_width[kept_y1] >= float(width_keep_frac) * med_width
        if ok_frac and ok_w:
            break
        kept_y1 -= 1

    if kept_y0 > kept_y1:
        return np.zeros((H, W), dtype=bool), mask_raw, None, None, med_xL, med_xR, row_black_frac, row_width

    # --- Build final filled mask: one contiguous horizontal interval per kept row ---
    mask_full = np.zeros((H, W), dtype=bool)

    for y in range(kept_y0, kept_y1 + 1):
        # Use the row’s own span if it exists; otherwise fallback to median span
        if row_width[y] > 0:
            xL = int(xL_row[y])
            xR = int(xR_row[y])
        else:
            xL = med_xL
            xR = med_xR

        xL = max(xL, 0)
        xR = min(xR, W - 1)
        if xL <= xR:
            mask_full[y, xL:xR + 1] = True

    return mask_full, mask_raw, int(kept_y0), int(kept_y1), int(med_xL), int(med_xR), row_black_frac, row_width
