import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# =========================
# Path handling (same style)
# =========================
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

DATASET_PATH = Path("dataset")
CREATE_DATASET_DIR = DATASET_PATH / "create_dataset"
TEMPLATE_JSON_CANDIDATES = [
    CREATE_DATASET_DIR / "nadir" / "final_annotations.json",
    CREATE_DATASET_DIR / "final_annotations.json",
]


# =========================
# COCO helpers
# =========================
def _load_json(path: Path) -> dict:
    """_load_json(path) -> dict: Read JSON utf-8."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    """_save_json(path,data) -> None: Write JSON utf-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_template_coco() -> dict:
    """_load_template_coco() -> dict: Load a COCO template preserving top-level keys."""
    for p in TEMPLATE_JSON_CANDIDATES:
        if p.is_file():
            return _load_json(p)
    return {"images": [], "annotations": [], "categories": []}


def _ensure_coco(split: str) -> tuple[Path, dict]:
    """_ensure_coco(split) -> (Path,dict): Load or create split COCO JSON."""
    out_path = CREATE_DATASET_DIR / split / "final_annotations.json"
    if out_path.is_file():
        return out_path, _load_json(out_path)

    tmpl = _load_template_coco()
    coco = dict(tmpl)
    coco["images"] = []
    coco["annotations"] = []
    return out_path, coco


# =========================
# Geometry helpers (clip + shift)
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


def _clip_polygon_to_rect(poly: np.ndarray, rect_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """_clip_polygon_to_rect(poly,rect_xyxy) -> np.ndarray: Clip polygon to rect (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = map(float, rect_xyxy)

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


def _intersect_bbox_with_rect(bbox_xywh: tuple[float, float, float, float],
                              rect_xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """_intersect_bbox_with_rect(bbox_xywh,rect_xyxy) -> (x,y,w,h)|None: Intersect bbox with rect."""
    x, y, w, h = map(float, bbox_xywh)
    x0, y0, x1, y1 = map(float, rect_xyxy)
    bx0, by0, bx1, by1 = x, y, x + w, y + h
    ix0, iy0 = max(bx0, x0), max(by0, y0)
    ix1, iy1 = min(bx1, x1), min(by1, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0, iy0, ix1 - ix0, iy1 - iy0)


def _flatten_poly(poly_xy: np.ndarray) -> list[float]:
    """_flatten_poly(poly_xy) -> list[float]: Flatten Nx2 to [x0,y0,x1,y1,...]."""
    flat: list[float] = []
    for x, y in poly_xy.tolist():
        flat.extend([float(x), float(y)])
    return flat


def _translate_and_clip_anns_to_patch(anns: list,
                                     top_left_xy: tuple[int, int],
                                     patch_wh: tuple[int, int],
                                     offset_xy: tuple[int, int]) -> list[dict]:
    """_translate_and_clip_anns_to_patch(anns,top_left_xy,patch_wh,offset_xy) -> list[dict]: Clip to patch and shift to patch-local coords."""
    x0, y0 = int(top_left_xy[0]), int(top_left_xy[1])
    pw, ph = int(patch_wh[0]), int(patch_wh[1])
    ox, oy = int(offset_xy[0]), int(offset_xy[1])

    rect_xyxy = (float(x0), float(y0), float(x0 + pw), float(y0 + ph))
    out: list[dict] = []

    for ann in anns:
        segs_in = ann.get("segmentation", [])
        bbox_in = ann.get("bbox", None)

        new_segs: list[list[float]] = []
        any_seg = False
        for seg in segs_in if isinstance(segs_in, list) else []:
            if not seg or len(seg) < 6:
                continue

            # bring to cropped-image coords (subtract crop offset)
            poly = np.array([(seg[i] - ox, seg[i + 1] - oy) for i in range(0, len(seg), 2)], dtype=float)

            # clip in that coord system
            poly = _clip_polygon_to_rect(poly, rect_xyxy)
            if poly.shape[0] < 3:
                continue

            # shift to patch-local
            poly[:, 0] -= float(x0)
            poly[:, 1] -= float(y0)

            flat = _flatten_poly(poly)
            if len(flat) >= 6:
                new_segs.append(flat)
                any_seg = True

        new_bbox = None
        if isinstance(bbox_in, (list, tuple)) and len(bbox_in) == 4:
            bx, by, bw, bh = bbox_in
            bx, by = float(bx - ox), float(by - oy)
            inter = _intersect_bbox_with_rect((bx, by, float(bw), float(bh)), rect_xyxy)
            if inter is not None:
                ix, iy, iw, ih = inter
                new_bbox = [float(ix - x0), float(iy - y0), float(iw), float(ih)]

        if (not any_seg) and (new_bbox is None):
            continue

        a2 = dict(ann)  # preserve ALL keys
        if any_seg:
            a2["segmentation"] = new_segs
        else:
            a2["segmentation"] = []
        if new_bbox is not None:
            a2["bbox"] = new_bbox
        out.append(a2)

    return out


# =========================
# Naming
# =========================
def _split_patch_name(stem: str) -> tuple[str, int | None]:
    """_split_patch_name(stem) -> (base,idx|None): Parse ..._<int> suffix."""
    m = re.match(r"^(.*)_([0-9]+)$", stem)
    if not m:
        return stem, None
    return m.group(1), int(m.group(2))


def _next_index_for_base(split_dir: Path, subdir: Path, base: str, ext: str) -> int:
    """_next_index_for_base(split_dir,subdir,base,ext) -> int: Next free _k for base within split/subdir."""
    folder = split_dir / subdir
    if not folder.exists():
        return 1
    used = set()
    for p in folder.glob(f"{base}_*{ext}"):
        b = p.stem
        bb, idx = _split_patch_name(b)
        if bb == base and idx is not None:
            used.add(idx)
    k = 1
    while k in used:
        k += 1
    return k


# =========================
# Public API
# =========================
def save_patch(split: str, patch_bundle: dict) -> dict:
    """save_patch(split,patch_bundle) -> dict: Save patch image + append COCO entry; mutates patch_bundle with patch_name (no path)."""
    split = str(split).lower().strip()
    if split not in {"nadir", "offnadir", "sunglint"}:
        raise ValueError("split must be one of: nadir, offnadir, sunglint")

    split_dir = CREATE_DATASET_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    img_file = patch_bundle.get("img_file", None)
    if not isinstance(img_file, str):
        raise ValueError("patch_bundle['img_file'] must be the original image file_name string (e.g. Pelagos2016/...B2.PNG)")

    subdir = Path(img_file).parent  # keep Pelagos2016/...
    ext = Path(img_file).suffix  # keep .PNG / .png exactly as original

    # patch array must be uint8 RGB
    patch = patch_bundle.get("patch", None)
    if patch is None:
        raise ValueError("patch_bundle['patch'] missing")
    patch_u8 = np.asarray(patch)
    if patch_u8.dtype != np.uint8:
        patch_u8 = np.clip(patch_u8, 0, 255).astype(np.uint8)
    if patch_u8.ndim != 3 or patch_u8.shape[2] < 3:
        raise ValueError(f"Expected patch RGB array (H,W,3), got {patch_u8.shape}")

    # Decide patch_name
    if split == "nadir":
        # create new unique patch_name for nadir
        base = Path(img_file).stem  # base stem of original image
        k = _next_index_for_base(split_dir, subdir, base=base, ext=ext)
        patch_name = f"{base}_{k}"
        patch_bundle["patch_name"] = patch_name
    else:
        # reuse same patch_name for offnadir/sunglint
        patch_name = patch_bundle.get("patch_name", None)
        if not isinstance(patch_name, str) or not patch_name:
            raise ValueError("For offnadir/sunglint, patch_bundle must already contain patch_name (set by save_patch('nadir', ...))")

    # Save image (NO overlays)
    out_rel = subdir / f"{patch_name}{ext}"
    out_abs = split_dir / out_rel
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    img_pil = Image.fromarray(patch_u8[:, :, :3], mode="RGB")
    img_pil.save(out_abs)  # PNG encoding; pixel values preserved as uint8

    # Update COCO
    coco_path, coco = _ensure_coco(split)
    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))

    # remove any existing entry with same file_name to avoid duplicates
    file_name = out_rel.as_posix()
    existing_img_ids = {im["id"] for im in images if im.get("file_name") == file_name}
    if existing_img_ids:
        images = [im for im in images if im.get("file_name") != file_name]
        anns = [a for a in anns if a.get("image_id") not in existing_img_ids]

    new_image_id = int(max([int(x.get("id", 0)) for x in images] + [0]) + 1)
    h, w = int(patch_u8.shape[0]), int(patch_u8.shape[1])

    img_info_src = patch_bundle.get("img_info", {})
    img_rec = dict(img_info_src) if isinstance(img_info_src, dict) else {}
    img_rec["id"] = new_image_id
    img_rec["file_name"] = file_name
    img_rec["width"] = w
    img_rec["height"] = h

    images.append(img_rec)

    # Build annotations for this saved image:
    if split == "nadir":
        top_left = patch_bundle.get("top_left", None)
        patch_wh = patch_bundle.get("patch_wh", None)
        offset_xy = patch_bundle.get("offset_xy", (0, 0))
        anns_in = patch_bundle.get("anns", [])

        if not (isinstance(top_left, (tuple, list)) and len(top_left) == 2):
            raise ValueError("patch_bundle['top_left'] must exist for nadir saving")
        if not (isinstance(patch_wh, (tuple, list)) and len(patch_wh) == 2):
            raise ValueError("patch_bundle['patch_wh'] must exist for nadir saving")
        if not isinstance(anns_in, list):
            raise ValueError("patch_bundle['anns'] must be a list")

        anns_kept = _translate_and_clip_anns_to_patch(
            anns=anns_in,
            top_left_xy=(int(top_left[0]), int(top_left[1])),
            patch_wh=(int(patch_wh[0]), int(patch_wh[1])),
            offset_xy=(int(offset_xy[0]), int(offset_xy[1])),
        )
        patch_bundle["anns_patch"] = anns_kept  # in-memory only, not a path
    else:
        # offnadir/sunglint: annotations already translated by translate_offnadir()
        anns_kept = patch_bundle.get("anns_patch", patch_bundle.get("anns", []))
        if not isinstance(anns_kept, list):
            raise ValueError("Expected translated annotations list in patch_bundle['anns_patch'] or ['anns']")

    next_ann_id = int(max([int(a.get("id", 0)) for a in anns] + [0]) + 1)
    out_anns = []
    for a in anns_kept:
        a2 = dict(a)  # preserve keys
        a2["id"] = next_ann_id
        a2["image_id"] = new_image_id
        out_anns.append(a2)
        next_ann_id += 1

    anns.extend(out_anns)

    coco["images"] = images
    coco["annotations"] = anns
    _save_json(coco_path, coco)

    print(f"Saved patch {out_abs.name} | anns_kept={len(out_anns)} | json={coco_path.name}")
    return patch_bundle
