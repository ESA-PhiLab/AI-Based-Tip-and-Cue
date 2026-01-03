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
