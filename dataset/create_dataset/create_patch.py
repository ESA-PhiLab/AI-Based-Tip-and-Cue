import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# =========================
# Path handling
# =========================
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)


# =========================
# Config
# =========================
DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "whales_from_space"
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "final_annotations.json"

BOOLS = {"crop_black_border": True}
CROP_THRESHOLD = 1


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
# Crop black border (returns offset to keep annotations aligned)
# =========================
def crop_black_border_image(img_rgb: np.ndarray, threshold: int) -> tuple[np.ndarray, tuple[int, int]]:
    """crop_black_border_image(img_rgb,threshold) -> (np.ndarray,(int,int)): Crop black border; return (cropped,(x0,y0)) offset."""
    a = np.asarray(img_rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"Expected RGB image array (H,W,3), got shape {a.shape}")

    thr = int(threshold)
    mask = np.any(a[:, :, :3] > thr, axis=2)
    if not np.any(mask):
        return a, (0, 0)

    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    return a[y0:y1, x0:x1, :], (x0, y0)


# =========================
# Geometry helpers (clip to patch boundary)
# =========================
def _clip_poly_edge(poly: np.ndarray, inside_fn, intersect_fn) -> np.ndarray:
    """_clip_poly_edge(poly,inside_fn,intersect_fn) -> np.ndarray: Clip polygon against one half-plane."""
    if poly.size == 0:
        return poly
    out = []
    prev = poly[-1]
    prev_in = inside_fn(prev)
    for cur in poly:
        cur_in = inside_fn(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect_fn(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect_fn(prev, cur))
        prev, prev_in = cur, cur_in
    return np.array(out, dtype=float) if out else np.zeros((0, 2), dtype=float)


def clip_polygon_to_rect(poly: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """clip_polygon_to_rect(poly,rect) -> np.ndarray: Clip polygon to rect (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = map(float, rect)

    def inside_left(p): return p[0] >= x0
    def inside_right(p): return p[0] <= x1
    def inside_top(p): return p[1] >= y0
    def inside_bottom(p): return p[1] <= y1

    def intersect_vertical(p1, p2, x):
        if p2[0] == p1[0]:
            return np.array([x, p1[1]], dtype=float)
        t = (x - p1[0]) / (p2[0] - p1[0])
        return np.array([x, p1[1] + t * (p2[1] - p1[1])], dtype=float)

    def intersect_horizontal(p1, p2, y):
        if p2[1] == p1[1]:
            return np.array([p1[0], y], dtype=float)
        t = (y - p1[1]) / (p2[1] - p1[1])
        return np.array([p1[0] + t * (p2[0] - p1[0]), y], dtype=float)

    out = poly.astype(float)
    out = _clip_poly_edge(out, inside_left, lambda a, b: intersect_vertical(a, b, x0))
    out = _clip_poly_edge(out, inside_right, lambda a, b: intersect_vertical(a, b, x1))
    out = _clip_poly_edge(out, inside_top, lambda a, b: intersect_horizontal(a, b, y0))
    out = _clip_poly_edge(out, inside_bottom, lambda a, b: intersect_horizontal(a, b, y1))
    return out


def intersect_bbox_with_rect(bbox_xywh: tuple[float, float, float, float], rect: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """intersect_bbox_with_rect(bbox_xywh,rect) -> (x,y,w,h)|None: Intersect bbox (x,y,w,h) with rect (x0,y0,x1,y1)."""
    x, y, w, h = map(float, bbox_xywh)
    x0, y0, x1, y1 = map(float, rect)
    bx0, by0, bx1, by1 = x, y, x + w, y + h
    ix0, iy0 = max(bx0, x0), max(by0, y0)
    ix1, iy1 = min(bx1, x1), min(by1, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0, iy0, ix1 - ix0, iy1 - iy0)


# =========================
# Masks (for scoring patches)
# =========================
def _poly_to_mask(hw: tuple[int, int], seg: list[float], offset_xy: tuple[int, int]) -> np.ndarray:
    """_poly_to_mask(hw,seg,offset_xy) -> np.ndarray: Rasterize one COCO polygon to boolean mask."""
    h, w = int(hw[0]), int(hw[1])
    ox, oy = int(offset_xy[0]), int(offset_xy[1])
    if len(seg) < 6:
        return np.zeros((h, w), dtype=bool)
    pts = [(seg[i] - ox, seg[i + 1] - oy) for i in range(0, len(seg), 2)]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, outline=1, fill=1)
    return np.array(m, dtype=bool)


def build_whale_masks(anns: list, hw: tuple[int, int], offset_xy: tuple[int, int]) -> list[np.ndarray]:
    """build_whale_masks(anns,hw,offset_xy) -> list[np.ndarray]: Union mask per annotation."""
    h, w = int(hw[0]), int(hw[1])
    masks = []
    for ann in anns:
        segs = ann.get("segmentation", [])
        if not segs:
            continue
        union = np.zeros((h, w), dtype=bool)
        for seg in segs:
            if seg:
                union |= _poly_to_mask((h, w), seg, offset_xy)
        if union.any():
            masks.append(union)
    return masks


# =========================
# Drawing: transparent segmentation mask + outline, bbox, patch outline
# =========================
def _draw_bbox_closed(draw: ImageDraw.ImageDraw, bbox_xywh: tuple[float, float, float, float], outline: tuple[int, int, int, int], width: int) -> None:
    """_draw_bbox_closed(draw,bbox_xywh,outline,width) -> None: Draw bbox as closed polyline."""
    x, y, w, h = bbox_xywh
    x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    draw.line(pts, fill=outline, width=int(width))


def _draw_poly_closed(draw: ImageDraw.ImageDraw, poly_xy: np.ndarray, outline: tuple[int, int, int, int], width: int) -> None:
    """_draw_poly_closed(draw,poly_xy,outline,width) -> None: Draw polygon outline as closed polyline."""
    if poly_xy.shape[0] < 3:
        return
    pts = [tuple(map(float, p)) for p in poly_xy.tolist()]
    draw.line(pts + [pts[0]], fill=outline, width=int(width))


def draw_annotations_with_transparent_mask(img_rgb: np.ndarray,
                                           anns: list,
                                           offset_xy: tuple[int, int] = (0, 0),
                                           clip_rect_xyxy: tuple[float, float, float, float] | None = None,
                                           only_ann_indices: list[int] | None = None,
                                           line_width: int = 2,
                                           mask_alpha: int = 80) -> np.ndarray:
    """draw_annotations_with_transparent_mask(...) -> np.ndarray: Transparent seg mask + green outline + red bbox (optionally filtered indices)."""
    ox, oy = int(offset_xy[0]), int(offset_xy[1])

    base = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dmask = ImageDraw.Draw(overlay, "RGBA")
    dbase = ImageDraw.Draw(base, "RGBA")

    if only_ann_indices is None:
        idxs = range(len(anns))
    else:
        idxs = [i for i in only_ann_indices if 0 <= i < len(anns)]

    for ai in idxs:
        ann = anns[ai]

        for seg in ann.get("segmentation", []):
            if not seg or len(seg) < 6:
                continue
            poly = np.array([(seg[i] - ox, seg[i + 1] - oy) for i in range(0, len(seg), 2)], dtype=float)
            if clip_rect_xyxy is not None:
                poly = clip_polygon_to_rect(poly, clip_rect_xyxy)

            if poly.shape[0] >= 3:
                dmask.polygon([tuple(p) for p in poly.tolist()], fill=(0, 255, 0, int(mask_alpha)))
                _draw_poly_closed(dbase, poly, outline=(0, 255, 0, 255), width=line_width)

        if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4:
            bx, by, bw, bh = ann["bbox"]
            bx, by = float(bx - ox), float(by - oy)
            bbox = (bx, by, float(bw), float(bh))
            if clip_rect_xyxy is not None:
                ib = intersect_bbox_with_rect(bbox, clip_rect_xyxy)
                if ib is not None:
                    _draw_bbox_closed(dbase, ib, outline=(255, 0, 0, 255), width=line_width)
            else:
                _draw_bbox_closed(dbase, bbox, outline=(255, 0, 0, 255), width=line_width)

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(composed, dtype=np.uint8)


# =========================
# Mode logic
# =========================
def classify_fracs(fracs: list[float],
                   nowhale_max_fraction: float,
                   whale_min_fraction: float,
                   half_fraction_range: tuple[float, float]) -> tuple[list[int], list[int], list[int]]:
    """classify_fracs(fracs,nowhale_max_fraction,whale_min_fraction,half_fraction_range) -> (full_idxs,half_idxs,partial_idxs)."""
    hlo, hhi = float(half_fraction_range[0]), float(half_fraction_range[1])
    full_idxs = [i for i, f in enumerate(fracs) if f >= whale_min_fraction]
    half_idxs = [i for i, f in enumerate(fracs) if hlo <= f <= hhi]
    partial_idxs = [i for i, f in enumerate(fracs) if (nowhale_max_fraction < f < whale_min_fraction)]
    return full_idxs, half_idxs, partial_idxs


def accept_patch(mode_single: str,
                 mode_multiple_allow_partial: bool,
                 fracs: list[float],
                 nowhale_max_fraction: float,
                 whale_min_fraction: float,
                 half_fraction_range: tuple[float, float]) -> tuple[bool, str]:
    """accept_patch(mode_single,mode_multiple_allow_partial,fracs,nowhale_max_fraction,whale_min_fraction,half_fraction_range) -> (ok,label)."""
    mode_single = str(mode_single).lower().strip()
    valid_modes = {"full", "half", "ocean", "full_half", "all"}
    if mode_single not in valid_modes:
        raise ValueError(f"mode_single must be one of {sorted(valid_modes)}")

    if not fracs:
        if mode_single in {"ocean", "all"}:
            return True, "OCEAN"
        return False, "REJECT_NO_ANN"

    full_idxs, half_idxs, partial_idxs = classify_fracs(fracs, nowhale_max_fraction, whale_min_fraction, half_fraction_range)
    max_frac = float(max(fracs))
    any_whale = max_frac > nowhale_max_fraction

    if mode_single == "ocean":
        ok = (not any_whale)
        return ok, "OCEAN" if ok else "REJECT_NOT_OCEAN"

    if mode_single == "all":
        return True, "ANY"

    if mode_single == "full":
        ok = len(full_idxs) > 0
        if ok and (not mode_multiple_allow_partial) and len(partial_idxs) > 0:
            return False, "REJECT_PARTIAL_PRESENT"
        return ok, "WHALE_FULL" if ok else "REJECT_NO_FULL"

    if mode_single == "half":
        ok = len(half_idxs) > 0
        if ok and (not mode_multiple_allow_partial) and len(partial_idxs) > 0:
            return False, "REJECT_PARTIAL_PRESENT"
        return ok, "WHALE_HALF" if ok else "REJECT_NO_HALF"

    # full_half
    ok = (len(full_idxs) > 0) or (len(half_idxs) > 0)
    if ok and (not mode_multiple_allow_partial) and len(partial_idxs) > 0:
        return False, "REJECT_PARTIAL_PRESENT"
    return ok, "WHALE_FULL_OR_HALF" if ok else "REJECT_NO_FULL_OR_HALF"


# =========================
# Patch generator
# =========================
def generate_patch(mode_single: str,
                   mode_multiple_allow_partial: bool,
                   window_size: int | tuple[int, int],
                   img_file: str,
                   rng: np.random.Generator,
                   whale_min_fraction: float,
                   nowhale_max_fraction: float,
                   half_fraction_range: tuple[float, float] = (0.10, 0.50),
                   annotations_path: Path = ANNOTATIONS_PATH,
                   base_dir: Path = BASE_DIR,
                   crop_black_border: bool = True,
                   crop_threshold: int = 1,
                   max_tries: int = 5000,
                   mask_alpha: int = 80, plot_patch: bool = True) -> tuple[np.ndarray, tuple[int, int], list[float], str]:
    """generate_patch(...) -> (patch,(x,y),fracs,label): Sample patch using mode_single + multiple-whale policy; print all whale fractions; plot full+patch."""
    if not (0.0 <= float(nowhale_max_fraction) < float(whale_min_fraction) <= 1.0):
        raise ValueError("Require 0 <= nowhale_max_fraction < whale_min_fraction <= 1")

    hlo, hhi = float(half_fraction_range[0]), float(half_fraction_range[1])
    if not (0.0 <= hlo < hhi <= 1.0):
        raise ValueError("half_fraction_range must satisfy 0 <= low < high <= 1")

    coco = load_json(Path(annotations_path))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    img_info = next((i for i in images if i.get("file_name") == img_file), None)
    if img_info is None:
        raise FileNotFoundError(f"Image not found in COCO: {img_file}")

    anns = anns_by_image(annotations).get(img_info["id"], [])

    img_path = Path(base_dir) / img_file
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file missing: {img_path}")

    img_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    offset_xy = (0, 0)
    if crop_black_border:
        img_rgb, offset_xy = crop_black_border_image(img_rgb, threshold=int(crop_threshold))

    h, w = img_rgb.shape[:2]

    if isinstance(window_size, int):
        ph, pw = int(window_size), int(window_size)
    else:
        ph, pw = int(window_size[0]), int(window_size[1])

    if ph <= 0 or pw <= 0 or ph > h or pw > w:
        raise ValueError(f"Invalid window_size {window_size} for image {w}x{h}")

    whale_masks = build_whale_masks(anns, (h, w), offset_xy)
    whale_areas = [max(1, int(m.sum())) for m in whale_masks]

    patch = None
    top_left = None
    fracs_out: list[float] = []
    label_out = "UNKNOWN"
    draw_idxs: list[int] = []

    for _ in range(int(max_tries)):
        x = int(rng.integers(0, w - pw + 1))
        y = int(rng.integers(0, h - ph + 1))

        fracs = []
        for m, area in zip(whale_masks, whale_areas):
            inside = int(m[y:y + ph, x:x + pw].sum())
            fracs.append(inside / float(area))

        ok, label = accept_patch(
            mode_single=mode_single,
            mode_multiple_allow_partial=mode_multiple_allow_partial,
            fracs=fracs,
            nowhale_max_fraction=nowhale_max_fraction,
            whale_min_fraction=whale_min_fraction,
            half_fraction_range=half_fraction_range,
        )

        if ok:
            patch = img_rgb[y:y + ph, x:x + pw].copy()
            top_left = (x, y)
            fracs_out = fracs
            label_out = label
            draw_idxs = [i for i, f in enumerate(fracs) if f > nowhale_max_fraction]
            break

    if patch is None or top_left is None:
        raise RuntimeError(f"Failed to sample a valid patch after {max_tries} tries")

    x, y = top_left
    marker_pixel_id = y * w + x
    print(f"{label_out} patch marker at (x={x}, y={y}), pixel_id={marker_pixel_id}, window={pw}x{ph}")
    if fracs_out:
        for i, f in enumerate(fracs_out):
            print(f"  whale[{i}] fraction_inside = {100.0*f:.2f}%")
        print(f"  max = {100.0*max(fracs_out):.2f}% | min = {100.0*min(fracs_out):.2f}%")
    else:
        print("  No whales in this image (no annotations).")

    rect_xyxy = (float(x), float(y), float(x + pw), float(y + ph))

    full_overlay = draw_annotations_with_transparent_mask(
        img_rgb, anns, offset_xy=offset_xy, clip_rect_xyxy=None, only_ann_indices=None, line_width=1, mask_alpha=mask_alpha
    )

    full_img = Image.fromarray(full_overlay, mode="RGB").convert("RGBA")
    dfull = ImageDraw.Draw(full_img, "RGBA")
    _draw_bbox_closed(dfull, (x, y, pw, ph), outline=(0, 0, 0, 255), width=1)
    full_overlay = np.asarray(full_img.convert("RGB"), dtype=np.uint8)

    if draw_idxs:
        overlay_full = draw_annotations_with_transparent_mask(
            img_rgb,
            anns,
            offset_xy=offset_xy,
            clip_rect_xyxy=rect_xyxy,
            only_ann_indices=draw_idxs,
            line_width=1,
            mask_alpha=mask_alpha
        )
        patch_overlay = overlay_full[y:y + ph, x:x + pw].copy()
    else:
        patch_overlay = patch.copy()

    if plot_patch:

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(full_overlay)
        axes[0].set_title("Full image (transparent mask, green outline, red bbox, black patch)")
        axes[0].axis("off")

        axes[1].imshow(patch_overlay)
        axes[1].set_title(f"Patch ({label_out}) (only whales > no-whale threshold drawn)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    patch_bundle = {
        "patch": patch,
        "top_left": (x, y),
        "patch_wh": (pw, ph),
        "label": label_out,
        "img_file": img_file,
        "img_info": img_info,
        "anns": anns,
        "offset_xy": offset_xy,
        "fracs": fracs_out,
        "settings": {
            "mode_single": mode_single,
            "mode_multiple_allow_partial": bool(mode_multiple_allow_partial),
            "window_size": window_size,
            "nowhale_max_fraction": float(nowhale_max_fraction),
            "whale_min_fraction": float(whale_min_fraction),
            "half_fraction_range": tuple(map(float, half_fraction_range)),
            "crop_black_border": bool(crop_black_border),
            "crop_threshold": int(crop_threshold),
            "max_tries": int(max_tries),
            "mask_alpha": int(mask_alpha),
            "plot_patch": bool(plot_patch),
        },
    }

    return patch_bundle


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    img_file = "Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG"
    img_file = "Ignacio2017/Ignacio_GW_WV3_PS_20170220_B58.PNG"

    # mode_single options:
    #   "full"      -> only full whales
    #   "half"      -> only half whales
    #   "ocean"     -> only ocean (no whales)
    #   "full_half" -> full OR half whales
    #   "all"       -> anything
    #
    # mode_multiple_allow_partial:
    #   True  -> if multiple whales, allow other partial whales in the patch
    #   False -> forbid any whale in (nowhale_max_fraction, whale_min_fraction)

    for _ in range(5):
        generate_patch(
            mode_single="full",
            mode_multiple_allow_partial=False,
            window_size=64,
            img_file=img_file,
            rng=rng,
            nowhale_max_fraction=0.10,
            whale_min_fraction=0.99,
            half_fraction_range=(0.20, 0.80),
            mask_alpha=80,
        )
