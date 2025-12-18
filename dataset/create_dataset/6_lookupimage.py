import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# =========================
# Path handling (SAME AS ORIGINAL)
# =========================
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)


# =========================
# Config (SAME STYLE AS ORIGINAL)
# =========================
DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "whales_from_space"
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "final_annotations.json"

IMG_FILE = "Ignacio2017/Ignacio_GW_WV3_PS_20170220_B58.PNG"
IMG_FILE = "Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG"

BOOLS = {
    "crop_black_border": True,
}

CROP_THRESHOLD = 1  # matches iu.crop_black_border_image(img_rgb, threshold=1)


# =========================
# COCO helpers
# =========================
def load_json(path: Path) -> dict:
    """load_json(path) -> dict: Read JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def anns_by_image(anns: list) -> dict:
    """anns_by_image(anns) -> dict: Map image_id -> list of annotations."""
    out = {}
    for a in anns:
        out.setdefault(a["image_id"], []).append(a)
    return out


# =========================
# Image utility (standalone version of crop_black_border_image)
# =========================
def crop_black_border_image(img_rgb: np.ndarray, threshold: int) -> np.ndarray:
    """crop_black_border_image(img_rgb,threshold) -> np.ndarray: Crop rows/cols where all pixels <= threshold."""
    a = np.asarray(img_rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"Expected RGB image array (H,W,3), got shape {a.shape}")

    thr = int(threshold)
    mask = np.any(a[:, :, :3] > thr, axis=2)  # (H,W) True where any channel > thr

    if not np.any(mask):
        return a  # nothing non-black detected; return original

    ys = np.where(np.any(mask, axis=1))[0]
    xs = np.where(np.any(mask, axis=0))[0]

    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    return a[y0:y1, x0:x1, :]


# =========================
# Drawing
# =========================
def draw_annotations(img: Image.Image, anns: list) -> Image.Image:
    """draw_annotations(img,anns) -> Image: Draw polygons (green) and bboxes (red)."""
    draw = ImageDraw.Draw(img)

    for ann in anns:
        for seg in ann.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

        if "bbox" in ann and len(ann["bbox"]) == 4:
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)

    return img


# =========================
# Main
# =========================
def main() -> None:
    """main() -> None: Load image, optional crop, overlay COCO annotations, display."""
    coco = load_json(ANNOTATIONS_PATH)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    img_info = next((i for i in images if i["file_name"] == IMG_FILE), None)
    if img_info is None:
        raise FileNotFoundError(f"Image not found in COCO: {IMG_FILE}")

    anns = anns_by_image(annotations).get(img_info["id"], [])

    img_path = BASE_DIR / IMG_FILE
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file missing: {img_path}")

    img_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h0, w0 = img_rgb.shape[:2]

    if BOOLS["crop_black_border"]:
        img_rgb = crop_black_border_image(img_rgb, threshold=CROP_THRESHOLD)

    h1, w1 = img_rgb.shape[:2]

    img_pil = Image.fromarray(img_rgb, mode="RGB")
    img_overlay = draw_annotations(img_pil.copy(), anns)

    plt.figure(figsize=(8, 8))
    plt.imshow(np.asarray(img_overlay))
    plt.axis("off")

    title = f"{IMG_FILE} — {len(anns)} annotations"
    if BOOLS["crop_black_border"]:
        title += f" (cropped {w0}x{h0} -> {w1}x{h1}, thr={CROP_THRESHOLD})"
    plt.title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
