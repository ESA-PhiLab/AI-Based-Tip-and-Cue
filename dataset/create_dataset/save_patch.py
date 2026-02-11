# save_patch.py
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from create_patch import WHALE_CATEGORY_ID


main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

DATASET_PATH = Path("dataset")
CREATE_DATASET_DIR = DATASET_PATH / "create_dataset"

TEMPLATE_JSON_CANDIDATES = [
    CREATE_DATASET_DIR / "patch_raw_255" / "final_annotations.json",
    CREATE_DATASET_DIR / "final_annotations.json",
]

ALLOWED_SPLITS = {
    "patch_raw_255",
    "patch_raw_rot_255",
    "texture_nadir_255",
    "radiance_nadir_255",
    "radiance_nadir_npy",
    "reflection_nadir_255",
    "reflection_nadir_npy",
    "texture_offnadir_255",
    "radiance_offnadir_255",
    "radiance_offnadir_npy",
    "reflection_offnadir_255",
    "reflection_offnadir_npy",
}


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
    """_ensure_coco(split) -> tuple[Path,dict]: Load or create split COCO JSON."""
    out_path = CREATE_DATASET_DIR / split / "final_annotations.json"
    if out_path.is_file():
        coco = _load_json(out_path)
    else:
        tmpl = _load_template_coco()
        coco = dict(tmpl)
        coco["images"] = []
        coco["annotations"] = []

    coco.pop("licenses", None)
    coco["categories"] = [{"id": WHALE_CATEGORY_ID, "name": "whale", "supercategory": "whale"}]

    return out_path, coco




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
    """_intersect_bbox_with_rect(bbox_xywh,rect_xyxy) -> tuple[float,float,float,float] | None: Intersect bbox with rect."""
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

            poly = np.array([(seg[i] - ox, seg[i + 1] - oy) for i in range(0, len(seg), 2)], dtype=float)
            poly = _clip_polygon_to_rect(poly, rect_xyxy)
            if poly.shape[0] < 3:
                continue

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

        a2 = dict(ann)
        a2["segmentation"] = new_segs if any_seg else []
        if new_bbox is not None:
            a2["bbox"] = new_bbox
        out.append(a2)

    return out


def _split_patch_name(stem: str) -> tuple[str, int | None]:
    """_split_patch_name(stem) -> tuple[str,int|None]: Parse ..._<int> suffix."""
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

def _deg_tag(deg: float | int | None) -> str:
    """_deg_tag(deg) -> str: Format angle rounded to nearest 5 deg as '_30deg'."""
    if deg is None:
        return ""
    x = float(deg)
    if not np.isfinite(x):
        return ""

    # round to nearest multiple of 5
    x5 = int(round(x / 5.0) * 5)

    return f"_{x5}deg"



def _split_needs_offnadir_tag(split: str) -> bool:
    """_split_needs_offnadir_tag(split) -> bool: True for offnadir output splits."""
    return split.startswith(("texture_offnadir_", "radiance_offnadir_", "reflection_offnadir_"))

def rollback_patch_outputs(img_file: str, patch_name: str) -> None:
    """rollback_patch_outputs(img_file,patch_name) -> None: Delete files + COCO records for patch_name across all splits."""
    img_file = str(img_file)
    patch_name = str(patch_name)

    subdir = Path(img_file).parent
    ext = Path(img_file).suffix

    # Remove both nadir filenames (no _deg tag) and any offnadir-tagged variants just in case
    # e.g. <subdir>/<patch_name>.PNG  and  <subdir>/<patch_name>_55deg.PNG
    def matches_patch(fn: str) -> bool:
        p = Path(fn)
        if p.parent.as_posix() != subdir.as_posix():
            return False
        s = p.stem  # without extension
        return s == patch_name or s.startswith(patch_name + "_") and s.endswith("deg")

    for split in sorted(ALLOWED_SPLITS):
        split_dir = CREATE_DATASET_DIR / split
        if not split_dir.exists():
            continue

        # delete files
        for p in (split_dir / subdir).glob(f"{patch_name}*{ext}"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in (split_dir / subdir).glob(f"{patch_name}*.npy"):
            try:
                p.unlink()
            except Exception:
                pass

        # scrub COCO json
        coco_path = CREATE_DATASET_DIR / split / "final_annotations.json"
        if not coco_path.is_file():
            continue

        coco = _load_json(coco_path)
        images = list(coco.get("images", []))
        anns = list(coco.get("annotations", []))

        bad_img_ids = {im.get("id") for im in images if isinstance(im, dict) and matches_patch(str(im.get("file_name", "")))}
        if bad_img_ids:
            images = [im for im in images if im.get("id") not in bad_img_ids]
            anns = [a for a in anns if a.get("image_id") not in bad_img_ids]
            coco["images"] = images
            coco["annotations"] = anns
            _save_json(coco_path, coco)

def _label_tag(label_simple: str | None) -> str:
    """_label_tag(label_simple) -> str: Map 'whale','whale_half','ocean' to '_F','_H','_O'."""
    if not isinstance(label_simple, str):
        return ""
    label_simple = label_simple.lower().strip()
    if label_simple == "whale":
        return "_F"
    if label_simple == "whale_half":
        return "_H"
    if label_simple == "ocean":
        return "_O"
    return ""


def save_patch(split: str, patch_bundle: dict) -> dict:
    """save_patch(split,patch_bundle) -> dict: Save patch image + append COCO entry; mutates patch_bundle with patch_name."""
    split = str(split).lower().strip()

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of: {sorted(ALLOWED_SPLITS)}")

    split_dir = CREATE_DATASET_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    img_file = patch_bundle.get("img_file", None)
    if not isinstance(img_file, str):
        raise ValueError("patch_bundle['img_file'] must be the original image file_name string")

    subdir = Path(img_file).parent
    ext = Path(img_file).suffix

    patch = patch_bundle.get("patch", None)
    if patch is None:
        raise ValueError("patch_bundle['patch'] missing")

    arr = np.asarray(patch)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected array (H,W,3), got {arr.shape}")

    is_float_payload = (arr.dtype != np.uint8) or (split.startswith("radiance_") or split.startswith("reflection_"))

    # Naming: patch_raw_255 creates a new patch_name; others reuse it
    # Naming: only patch_raw_255 creates a new patch_name; all others reuse it
    if split == "patch_raw_255":
        base = Path(img_file).stem
        k = _next_index_for_base(split_dir, subdir, base=base, ext=ext)
        patch_name = f"{base}_{k}"
        patch_bundle["patch_name"] = patch_name
    else:
        patch_name = patch_bundle.get("patch_name", None)
        if not isinstance(patch_name, str) or not patch_name:
            raise ValueError(f"For {split}, patch_bundle must contain patch_name (set by save_patch('patch_raw_255', ...))")


    is_npy_split = split.endswith("_npy")

    # Add label tag for all splits except raw rotation intermediate if desired
    label_tag = _label_tag(patch_bundle.get("label_simple", None))

    if _split_needs_offnadir_tag(split):
        tag = label_tag + _deg_tag(patch_bundle.get("offnadir_deg", None))
    else:
        tag = label_tag + "_nadir"

    out_rel = (subdir / f"{patch_name}{tag}.npy") if is_npy_split else (subdir / f"{patch_name}{tag}{ext}")

    out_abs = split_dir / out_rel
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    if is_npy_split:
        np.save(out_abs, arr[:, :, :3].astype(np.float32))
    else:
        patch_u8 = arr
        if patch_u8.dtype != np.uint8:
            patch_u8 = np.clip(patch_u8, 0, 255).astype(np.uint8)
        Image.fromarray(patch_u8[:, :, :3], mode="RGB").save(out_abs)

    coco_path, coco = _ensure_coco(split)
    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))

    file_name = out_rel.as_posix()
    existing_img_ids = {im["id"] for im in images if im.get("file_name") == file_name}
    if existing_img_ids:
        images = [im for im in images if im.get("file_name") != file_name]
        anns = [a for a in anns if a.get("image_id") not in existing_img_ids]

    new_image_id = int(max([int(x.get("id", 0)) for x in images] + [0]) + 1)
    h, w = int(arr.shape[0]), int(arr.shape[1])

    img_info_src = patch_bundle.get("img_info", {})
    img_rec = dict(img_info_src) if isinstance(img_info_src, dict) else {}
    img_rec.pop("license", None)
    img_rec["id"] = new_image_id
    img_rec["file_name"] = file_name
    img_rec["width"] = w
    img_rec["height"] = h

    if "label_simple" in patch_bundle:
        img_rec["label_simple"] = patch_bundle["label_simple"]

    images.append(img_rec)


    # Annotations:
    if split == "patch_raw_255":
        top_left = patch_bundle.get("top_left", None)
        patch_wh = patch_bundle.get("patch_wh", None)
        offset_xy = patch_bundle.get("offset_xy", (0, 0))
        anns_in = patch_bundle.get("anns", [])

        if not (isinstance(top_left, (tuple, list)) and len(top_left) == 2):
            raise ValueError("patch_bundle['top_left'] must exist for raw saving")
        if not (isinstance(patch_wh, (tuple, list)) and len(patch_wh) == 2):
            raise ValueError("patch_bundle['patch_wh'] must exist for raw saving")
        if not isinstance(anns_in, list):
            raise ValueError("patch_bundle['anns'] must be a list")

        anns_kept = _translate_and_clip_anns_to_patch(
            anns=anns_in,
            top_left_xy=(int(top_left[0]), int(top_left[1])),
            patch_wh=(int(patch_wh[0]), int(patch_wh[1])),
            offset_xy=(int(offset_xy[0]), int(offset_xy[1])),
        )
        patch_bundle["anns_patch"] = anns_kept

    else:
        anns_kept = patch_bundle.get("anns_patch", None)
        if not isinstance(anns_kept, list):
            raise ValueError(f"Expected patch-local annotations list in patch_bundle['anns_patch'] for split={split}")

    next_ann_id = int(max([int(a.get("id", 0)) for a in anns] + [0]) + 1)
    cats = coco.get("categories", [])
    dataset_cat_id = int(cats[0]["id"]) if isinstance(cats, list) and cats and isinstance(cats[0], dict) and "id" in cats[0] else WHALE_CATEGORY_ID

    out_anns = []
    for a in anns_kept:
        a2 = dict(a)
        a2["id"] = next_ann_id
        a2["image_id"] = new_image_id
        a2["category_id"] = dataset_cat_id
        out_anns.append(a2)
        next_ann_id += 1

    anns.extend(out_anns)

    coco["images"] = images
    coco["annotations"] = anns
    _save_json(coco_path, coco)

    print(f"Saved patch {out_abs.name} | anns_kept={len(out_anns)} | json={coco_path.name}")
    return patch_bundle
