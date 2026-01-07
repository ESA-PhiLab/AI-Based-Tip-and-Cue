import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import tempfile
import uuid



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
    }

    return patch_bundle

def _resolve_patch_anns_path(patch_img_path: Path) -> Path:
    """_resolve_patch_anns_path(patch_img_path) -> Path: Find patch_raw/.../final_annotations.json next to patch image."""
    subdir = patch_img_path.parent.relative_to(PATCH_DIR)  # e.g. Pelagos2016/
    p = PATCH_DIR / subdir / ANNS_JSON_NAME
    if not p.is_file():
        # fallback: global patch_raw/final_annotations.json (if you store it once)
        p2 = PATCH_DIR / ANNS_JSON_NAME
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"Missing annotations json for patch: tried {p} and {p2}")
    return p

def _normalize_rotation_angle(rotation_angle_deg: float) -> int:
    """_normalize_rotation_angle(rotation_angle_deg) -> int: Normalize to one of {0,90,180,-90}."""
    a = int(round(float(rotation_angle_deg)))
    a = ((a + 180) % 360) - 180  # map to [-180,180)
    if a == -180:
        a = 180
    if a not in (0, 90, 180, -90):
        raise ValueError("rotation_angle_deg must be one of {0, 90, 180, -90}.")
    return a


def _rotate_xy(x: float, y: float, w: int, h: int, angle: int) -> tuple[float, float]:
    """_rotate_xy(x,y,w,h,angle) -> (x2,y2): Rotate a point around image origin for multiples of 90 deg."""
    if angle == 0:
        return x, y
    if angle == 90:      # CCW
        return y, (w - 1) - x
    if angle == -90:     # CW
        return (h - 1) - y, x
    # 180
    return (w - 1) - x, (h - 1) - y


def _rotated_size(w: int, h: int, angle: int) -> tuple[int, int]:
    """_rotated_size(w,h,angle) -> (w2,h2): Output image size after rotation."""
    if angle in (90, -90):
        return h, w
    return w, h


def _rotate_segmentation(segmentation: list, w: int, h: int, angle: int) -> list:
    """_rotate_segmentation(segmentation,w,h,angle) -> list: Rotate COCO polygon segmentation."""
    if not isinstance(segmentation, list):
        return segmentation

    out = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            out.append(poly)
            continue
        flat = []
        for i in range(0, len(poly), 2):
            x, y = float(poly[i]), float(poly[i + 1])
            x2, y2 = _rotate_xy(x, y, w, h, angle)
            flat.extend([x2, y2])
        out.append(flat)
    return out


def _rotate_bbox_xywh(bbox: list, w: int, h: int, angle: int) -> list:
    """_rotate_bbox_xywh(bbox,w,h,angle) -> list: Rotate bbox [x,y,w,h] and return new axis-aligned bbox."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return bbox

    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    corners = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    rc = [_rotate_xy(cx, cy, w, h, angle) for (cx, cy) in corners]

    xs = [p[0] for p in rc]
    ys = [p[1] for p in rc]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]




def _write_temp_rotated_inputs(rot_img_u8: np.ndarray, rot_anns: list, base_coco_path: Path) -> tuple[Path, Path]:
    """_write_temp_rotated_inputs(rot_img_u8,rot_anns,base_coco_path) -> (img_path,anns_path): Write temp rotated image + COCO json."""
    coco_in = json.loads(Path(base_coco_path).read_text(encoding="utf-8"))

    h2, w2 = rot_img_u8.shape[:2]
    tmp_dir = Path(tempfile.gettempdir()) / "ai_tc_rotate"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tag = uuid.uuid4().hex[:10]
    img_name = f"rot_{tag}.png"
    img_path = tmp_dir / img_name
    Image.fromarray(rot_img_u8, mode="RGB").save(img_path)

    coco_out = {
        "info": coco_in.get("info", {}),
        "licenses": coco_in.get("licenses", []),
        "categories": coco_in.get("categories", []),
        "images": [{"id": 1, "file_name": img_name, "width": int(w2), "height": int(h2)}],
        "annotations": [],
    }

    for a in rot_anns:
        if not isinstance(a, dict):
            continue
        a2 = dict(a)
        a2["image_id"] = 1
        coco_out["annotations"].append(a2)

    anns_path = tmp_dir / f"rot_{tag}.json"
    anns_path.write_text(json.dumps(coco_out, indent=2), encoding="utf-8")
    return img_path, anns_path






def mirror_rotate_raw_patch_bundle(patch_bundle: dict, rotation_angle_deg: float, mirror: bool = False) -> dict:
    """rotate_raw_patch_bundle(patch_bundle,rotation_angle_deg,mirror=False) -> dict: Mirror then rotate patch + raw anns."""
    if "patch" not in patch_bundle:
        raise KeyError("patch_bundle['patch'] missing")
    if "anns" not in patch_bundle or not isinstance(patch_bundle["anns"], list):
        raise KeyError("patch_bundle['anns'] missing (expected raw image-space COCO anns)")

    patch = np.asarray(patch_bundle["patch"])
    if patch.ndim != 3 or patch.shape[2] < 3:
        raise ValueError(f"patch_bundle['patch'] must be HxWx3, got shape={patch.shape}")

    mirr_patch, mirr_anns = mirror_image_and_annotations(
        orig_img_u8=patch.astype(np.uint8),
        anns=patch_bundle["anns"],
        mirror=bool(mirror),
        direction="horizontal",
    )

    rot_patch, rot_anns = rotate_image_and_annotations(
        orig_img_u8=mirr_patch.astype(np.uint8),
        anns=mirr_anns,
        rotation_angle_deg=float(rotation_angle_deg),
    )

    out = dict(patch_bundle)
    out["patch"] = rot_patch
    out["anns"] = rot_anns
    out["rotation_angle_deg"] = float(rotation_angle_deg)
    out["mirror"] = bool(mirror)
    out.pop("patch_name", None)
    out.pop("anns_patch", None)
    return out





def rotate_image_and_annotations(orig_img_u8: np.ndarray, anns: list, rotation_angle_deg: float) -> tuple[np.ndarray, list]:
    """rotate_image_and_annotations(orig_img_u8,anns,rotation_angle_deg) -> (img_u8,anns2): Rotate image and COCO anns by 0/90/180/-90."""
    angle = _normalize_rotation_angle(rotation_angle_deg)
    if angle == 0:
        return orig_img_u8, anns

    h, w = orig_img_u8.shape[:2]

    # numpy rotations are CCW for k>0
    if angle == 90:
        img2 = np.rot90(orig_img_u8, k=1)
    elif angle == -90:
        img2 = np.rot90(orig_img_u8, k=3)
    else:  # 180
        img2 = np.rot90(orig_img_u8, k=2)

    anns2 = []
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        a2 = dict(ann)
        if "segmentation" in a2:
            a2["segmentation"] = _rotate_segmentation(a2.get("segmentation", []), w, h, angle)
        if "bbox" in a2:
            a2["bbox"] = _rotate_bbox_xywh(a2.get("bbox", None), w, h, angle)
        anns2.append(a2)

    return img2, anns2

def _mirror_xy(x: float, y: float, w: int, h: int, direction: str) -> tuple[float, float]:
    """_mirror_xy(x,y,w,h,direction) -> (x2,y2): Mirror a point ('horizontal' or 'vertical')."""
    if direction == "horizontal":  # left-right
        return (w - 1) - x, y
    if direction == "vertical":    # up-down
        return x, (h - 1) - y
    raise ValueError("direction must be 'horizontal' or 'vertical'.")


def _mirror_segmentation(segmentation: list, w: int, h: int, direction: str) -> list:
    """_mirror_segmentation(segmentation,w,h,direction) -> list: Mirror COCO polygon segmentation."""
    if not isinstance(segmentation, list):
        return segmentation

    out = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            out.append(poly)
            continue
        flat = []
        for i in range(0, len(poly), 2):
            x, y = float(poly[i]), float(poly[i + 1])
            x2, y2 = _mirror_xy(x, y, w, h, direction)
            flat.extend([x2, y2])
        out.append(flat)
    return out


def _mirror_bbox_xywh(bbox: list, w: int, h: int, direction: str) -> list:
    """_mirror_bbox_xywh(bbox,w,h,direction) -> list: Mirror bbox [x,y,w,h]."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return bbox

    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    if direction == "horizontal":
        x2 = (w - 1) - (x + bw)
        return [float(x2), float(y), float(bw), float(bh)]
    if direction == "vertical":
        y2 = (h - 1) - (y + bh)
        return [float(x), float(y2), float(bw), float(bh)]

    raise ValueError("direction must be 'horizontal' or 'vertical'.")


def mirror_image_and_annotations(orig_img_u8: np.ndarray, anns: list, mirror: bool = False, direction: str = "horizontal") -> tuple[np.ndarray, list]:
    """mirror_image_and_annotations(orig_img_u8,anns,mirror=False,direction='horizontal') -> (img_u8,anns2): Mirror image+COCO anns."""
    if not mirror:
        return orig_img_u8, anns

    img = np.asarray(orig_img_u8)
    if img.ndim != 3:
        raise ValueError(f"orig_img_u8 must be HxWxC, got shape={img.shape}")

    h, w = img.shape[:2]

    if direction == "horizontal":
        img2 = img[:, ::-1, ...]
    elif direction == "vertical":
        img2 = img[::-1, :, ...]
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'.")

    anns2 = []
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        a2 = dict(ann)
        if "segmentation" in a2:
            a2["segmentation"] = _mirror_segmentation(a2.get("segmentation", []), w, h, direction)
        if "bbox" in a2:
            a2["bbox"] = _mirror_bbox_xywh(a2.get("bbox", None), w, h, direction)
        anns2.append(a2)

    return img2, anns2


def make_patch_local_anns(anns: list, top_left: tuple[int, int], patch_wh: tuple[int, int], offset_xy: tuple[int, int]) -> list:
    """make_patch_local_anns(anns,top_left,patch_wh,offset_xy) -> list: Clip anns to patch and shift to patch-local coords."""
    x, y = int(top_left[0]), int(top_left[1])
    pw, ph = int(patch_wh[0]), int(patch_wh[1])
    ox, oy = int(offset_xy[0]), int(offset_xy[1])

    rect_xyxy = (float(x), float(y), float(x + pw), float(y + ph))
    out = []

    for ann in anns:
        if not isinstance(ann, dict):
            continue

        segs_out = []
        for seg in ann.get("segmentation", []):
            if not seg or len(seg) < 6:
                continue

            poly = np.array([(seg[i] - ox, seg[i + 1] - oy) for i in range(0, len(seg), 2)], dtype=float)
            poly = clip_polygon_to_rect(poly, rect_xyxy)
            if poly.shape[0] < 3:
                continue

            poly[:, 0] -= float(x)
            poly[:, 1] -= float(y)

            flat = []
            for px, py in poly.tolist():
                flat.extend([float(px), float(py)])
            if len(flat) >= 6:
                segs_out.append(flat)

        bbox_out = None
        if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4:
            bx, by, bw, bh = ann["bbox"]
            bbox_img = (float(bx - ox), float(by - oy), float(bw), float(bh))
            ib = intersect_bbox_with_rect(bbox_img, rect_xyxy)
            if ib is not None:
                ix, iy, iw, ih = ib
                bbox_out = [float(ix - x), float(iy - y), float(iw), float(ih)]

        if not segs_out and bbox_out is None:
            continue

        a2 = dict(ann)
        a2["segmentation"] = segs_out
        if bbox_out is not None:
            a2["bbox"] = bbox_out
        out.append(a2)

    return out


def plot_patch_after_rotation(patch_bundle: dict) -> None:
    """plot_patch_after_rotation(patch_bundle) -> None: Left panel identical to generate_patch(plot_patch=True); right panel is rotated patch with rotated anns."""
    if "img_file" not in patch_bundle:
        raise KeyError("patch_bundle['img_file'] missing")
    if "top_left" not in patch_bundle or "patch_wh" not in patch_bundle:
        raise KeyError("patch_bundle['top_left'] or ['patch_wh'] missing")
    if "rotation_angle_deg" not in patch_bundle:
        raise KeyError("patch_bundle['rotation_angle_deg'] missing (call rotate_raw_patch_bundle first)")
    if "patch" not in patch_bundle:
        raise KeyError("patch_bundle['patch'] missing")

    img_file = str(patch_bundle["img_file"])
    x, y = map(int, patch_bundle["top_left"])
    pw, ph = map(int, patch_bundle["patch_wh"])
    rot_patch = np.asarray(patch_bundle["patch"], dtype=np.uint8)
    angle = float(patch_bundle["rotation_angle_deg"])
    label_out = str(patch_bundle.get("label", "UNKNOWN"))
    mirror = bool(patch_bundle.get("mirror", False))

    # --- Rebuild LEFT panel exactly like generate_patch(plot_patch=True) ---
    coco = load_json(ANNOTATIONS_PATH)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    img_info = next((i for i in images if i.get("file_name") == img_file), None)
    if img_info is None:
        raise FileNotFoundError(f"Image not found in COCO: {img_file}")

    anns_full = anns_by_image(annotations).get(img_info["id"], [])

    img_path = BASE_DIR / img_file
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file missing: {img_path}")

    img_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    # Must use the same crop behavior as generate_patch
    offset_xy = (0, 0)
    if bool(BOOLS.get("crop_black_border", True)):
        img_rgb, offset_xy = crop_black_border_image(img_rgb, threshold=int(CROP_THRESHOLD))

    # Draw full overlay (all anns) + black patch rectangle
    mask_alpha = 80  # generate_patch default; if you used another value you must pass/store it somewhere
    full_overlay = draw_annotations_with_transparent_mask(
        img_rgb, anns_full, offset_xy=offset_xy, clip_rect_xyxy=None, only_ann_indices=None, line_width=1, mask_alpha=mask_alpha
    )
    full_img = Image.fromarray(full_overlay, mode="RGB").convert("RGBA")
    dfull = ImageDraw.Draw(full_img, "RGBA")
    _draw_bbox_closed(dfull, (x, y, pw, ph), outline=(0, 0, 0, 255), width=1)
    full_overlay = np.asarray(full_img.convert("RGB"), dtype=np.uint8)

    # --- Build RIGHT panel (rotated patch + rotated patch-local annotations) ---
    anns_patch = make_patch_local_anns(anns_full, top_left=(x, y), patch_wh=(pw, ph), offset_xy=offset_xy)

    # Rotate annotations in patch-local frame using a dummy image of the unrotated patch size
    dummy = np.zeros((ph, pw, 3), dtype=np.uint8)

    _mirr_dummy, mirr_anns_patch = mirror_image_and_annotations(
        orig_img_u8=dummy,
        anns=anns_patch,
        mirror=mirror,
        direction="horizontal",
    )

    _img2, rot_anns_patch = rotate_image_and_annotations(_mirr_dummy, mirr_anns_patch, angle)

    rot_patch_overlay = draw_annotations_with_transparent_mask(
        rot_patch, rot_anns_patch, offset_xy=(0, 0), clip_rect_xyxy=None, only_ann_indices=None, line_width=1, mask_alpha=mask_alpha
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(full_overlay)
    axes[0].set_title("Full image (transparent mask, green outline, red bbox, black patch)")
    axes[0].axis("off")

    axes[1].imshow(rot_patch_overlay)
    axes[1].set_title(f"Patch ({label_out}) after rotation")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()






# =========================
# Example usage
# =========================
if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    img_file = "Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG"
    img_file = "Ignacio2017/Ignacio_GW_WV3_PS_20170220_B58.PNG"

    rng_rot = np.random.default_rng(42)

    # mode_single options:
    #   "full"      -> only full whales
    #   "half"      -> only half whales
    #   "ocean"     -> only ocean (no whales)
    #   "full_half" -> full OR half whales
    #   "all"       -> anything

    # mode_multiple_allow_partial:
    #   True  -> if multiple whales, allow other partial whales in the patch
    #   False -> forbid any whale in (nowhale_max_fraction, whale_min_fraction)

    for _ in range(5):
        rotation_angle_deg = float(rng_rot.choice([0, 90, 180, -90]))

        patch_bundle = generate_patch(
            mode_single="full",
            mode_multiple_allow_partial=False,
            window_size=64,
            img_file=img_file,
            rng=rng,
            nowhale_max_fraction=0.10,
            whale_min_fraction=0.99,
            half_fraction_range=(0.20, 0.80),
            mask_alpha=80,
            plot_patch=False,
        )

        mirror = bool(rng_rot.integers(0, 2))  # or set True/False yourself

        patch_bundle_rot = mirror_rotate_raw_patch_bundle(patch_bundle, rotation_angle_deg, mirror=mirror)
        plot_patch_after_rotation(patch_bundle_rot)

