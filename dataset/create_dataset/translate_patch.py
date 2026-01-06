import os
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import mitsuba as mi
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import tempfile
import uuid

import json

from offnadir_imaging.rendering import generate_image
from offnadir_imaging.create_DEM.create_dummy_DEM import get_DEM
from offnadir_imaging.create_DEM.convert_DEM import convert_DEM
from offnadir_imaging.functions.get_satellite_data import get_satellite, get_spatial_res, get_band_data
from offnadir_imaging.functions.convert_reference_frames import get_ecef_from_lat_lon
from offnadir_imaging.functions.intermediate_functions import get_scene_characteristics, is_dark_from_sun_dir


# =========================
# Path handling (same style)
# =========================
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

DATASET_PATH = Path("dataset")
CREATE_DATASET_DIR = DATASET_PATH / "create_dataset"
PATCH_DIR = CREATE_DATASET_DIR / "patch_raw_255"
CSV_PATH = DATASET_PATH / "whales_from_space" / "WhaleFromSpaceDB_Whales.csv"

DEFAULT_DATETIME_UTC = datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)

BOOLS_BASE = {
    "plot_3d": False,
    "plot_result": False,
    "max_glint": False,
    "print_values": True,
    "crop_black_border": False,  # ignore cropping
    "generate_radiation": True,
    "generate_nadir": True
}

WAVE_PROPERTIES = {"wind_speed": 10.0, "num_waves": 50, "wave_min": 0.05, "wave_max": 0.5}
SAMPLE_COUNT = 512

# Segmentation translation:
MASK_SUPERSAMPLE = 2
MASK_SPP = 1
MASK_FILTER = "bilinear"
MASK_CLOSE_RADIUS = 2

# BBox translation:
ID_SPP = 1
ID_POINT_RADIUS_PX = 4
ID_FILTER = "nearest"


# =========================
# Drawing helpers (baseline)
# =========================
def draw_overlay(img: Image.Image, anns: list) -> Image.Image:
    """draw_overlay(img,anns) -> Image: Draw polygons and bboxes."""
    draw = ImageDraw.Draw(img)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=1)
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
    return img


def draw_overlay_from_polys(img: Image.Image, polys_xy: list, boxes_xywh: list) -> Image.Image:
    """draw_overlay_from_polys(img,polys_xy,boxes_xywh) -> Image: Draw projected polys/boxes."""
    draw = ImageDraw.Draw(img)
    for pts in polys_xy:
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=1)
    for (x, y, w, h) in boxes_xywh:
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
    return img


def to_uint8_rgb(arr) -> np.ndarray:
    """to_uint8_rgb(arr) -> np.ndarray: Clamp/cast to uint8 RGB array."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


# =========================
# Mitsuba emission render
# =========================
def render_emission_texture(dem_obj_path: str,
                            tex_rgb01: np.ndarray,
                            to_world_scene,
                            to_world_sensor,
                            fov_deg: float,
                            res: int,
                            spp: int,
                            filter_type: str) -> np.ndarray:
    """render_emission_texture(dem_obj_path,tex_rgb01,to_world_scene,to_world_sensor,fov_deg,res,spp,filter_type) -> np.ndarray."""
    tex = {
        "type": "bitmap",
        "data": mi.TensorXf(np.asarray(tex_rgb01, dtype=np.float32)),
        "raw": True,
        "wrap_mode": "clamp",
        "filter_type": str(filter_type),
    }

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 1},
        "earth_surface": {
            "type": "obj",
            "filename": dem_obj_path,
            "to_world": to_world_scene,
            "bsdf": {"type": "diffuse", "reflectance": 0.0},
            "emitter": {"type": "area", "radiance": tex},
        },
        "sensor": {
            "type": "perspective",
            "to_world": to_world_sensor,
            "fov": float(fov_deg),
            "far_clip": 1e8,
            "film": {
                "type": "hdrfilm",
                "width": int(res),
                "height": int(res),
                "rfilter": {"type": "box"},
                "sample_border": True,
                "compensate": False,
            },
            "sampler": {"type": "independent", "sample_count": int(max(1, spp))},
        },
    })

    img = np.array(mi.render(scene), copy=False)
    _ = mi.util.convert_to_bitmap(img)

    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =========================
# Segmentation: per-annotation mask -> render -> polygons
# =========================
def build_binary_mask_texture_single(ann: dict, tex_w: int, tex_h: int) -> np.ndarray:
    """build_binary_mask_texture_single(ann,tex_w,tex_h) -> np.ndarray: RGB(0..1) emission texture for one instance."""
    m = Image.new("L", (tex_w, tex_h), 0)
    d = ImageDraw.Draw(m)
    for seg in ann.get("segmentation", []):
        if not seg:
            continue
        pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        if len(pts) >= 3:
            d.polygon(pts, fill=255, outline=255)
    mask_u8 = np.asarray(m, dtype=np.uint8)
    return np.stack([mask_u8, mask_u8, mask_u8], axis=-1).astype(np.float32) / 255.0


def threshold_mask(off_mask_u8: np.ndarray, thr: int) -> np.ndarray:
    """threshold_mask(off_mask_u8,thr) -> np.ndarray: bool mask."""
    g = off_mask_u8[..., 0].astype(np.uint8)
    return g > np.uint8(thr)


def downsample_binary_mask(mask_bool: np.ndarray, out_res: int) -> np.ndarray:
    """downsample_binary_mask(mask_bool,out_res) -> np.ndarray: Downsample (H,W)->(out_res,out_res) by block mean."""
    h, w = mask_bool.shape
    if h == out_res and w == out_res:
        return mask_bool
    if h % out_res != 0 or w % out_res != 0:
        img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
        img = img.resize((out_res, out_res), resample=Image.NEAREST)
        return (np.asarray(img, dtype=np.uint8) > 0)
    sy = h // out_res
    sx = w // out_res
    m = mask_bool.reshape(out_res, sy, out_res, sx).mean(axis=(1, 3))
    return m > 0.5


def postprocess_binary_mask(binary_mask: np.ndarray, close_radius: int) -> np.ndarray:
    """postprocess_binary_mask(binary_mask,close_radius) -> np.ndarray: Close gaps and fill holes."""
    try:
        import cv2  # type: ignore
        m = (binary_mask.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        h, w = m2.shape
        ff = m2.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        filled = cv2.bitwise_or(m2, holes)
        return filled > 127
    except Exception:
        pass

    try:
        from scipy import ndimage  # type: ignore
        m = binary_mask.astype(bool)
        structure = ndimage.generate_binary_structure(2, 1)
        m2 = ndimage.binary_closing(m, structure=structure, iterations=int(max(1, close_radius)))
        m3 = ndimage.binary_fill_holes(m2)
        return m3.astype(bool)
    except Exception:
        raise RuntimeError("Need either opencv-python (cv2) or scipy for mask hole filling/closing.")


def mask_to_polygons(binary_mask: np.ndarray, simplify_eps: float) -> list:
    """mask_to_polygons(binary_mask,simplify_eps) -> list: list of polygons [(x,y),...]."""
    try:
        import cv2  # type: ignore
        m = (binary_mask.astype(np.uint8) * 255)
        contours, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        polys = []
        for c in contours:
            if c.shape[0] < 10:
                continue
            approx = cv2.approxPolyDP(c, epsilon=float(simplify_eps), closed=True)
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(pts) >= 3:
                polys.append(pts)
        return polys
    except Exception:
        pass

    try:
        from skimage import measure  # type: ignore
        contours = measure.find_contours(binary_mask.astype(np.float32), 0.5)
        polys = []
        for c in contours:
            if c.shape[0] < 20:
                continue
            pts = [(int(round(p[1])), int(round(p[0]))) for p in c]
            if len(pts) >= 3:
                polys.append(pts)
        return polys
    except Exception:
        raise RuntimeError("Need opencv-python (cv2) or scikit-image for contour extraction.")


def polygons_to_coco_segmentation(polys_xy: list) -> list:
    """polygons_to_coco_segmentation(polys_xy) -> list: COCO polygon list (flattened)."""
    seg = []
    for pts in polys_xy:
        flat = []
        for x, y in pts:
            flat.extend([float(x), float(y)])
        if len(flat) >= 6:
            seg.append(flat)
    return seg


# =========================
# BBox: ID pass
# =========================
def id_to_rgb_u8(i: int) -> tuple[int, int, int]:
    """id_to_rgb_u8(i) -> (r,g,b): Encode 1..16,777,215 into RGB."""
    i = int(i) & 0xFFFFFF
    return (int((i >> 16) & 255), int((i >> 8) & 255), int(i & 255))


def rgb_u8_to_id(rgb: np.ndarray) -> np.ndarray:
    """rgb_u8_to_id(rgb) -> np.ndarray: Decode RGB uint8 image to int32 IDs."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r << 16) | (g << 8) | b


def build_bbox_id_texture(anns: list, tex_w: int, tex_h: int, radius_px: int) -> tuple[np.ndarray, dict, np.ndarray]:
    """build_bbox_id_texture(anns,tex_w,tex_h,radius_px) -> (tex_rgb01,meta,debug_u8)."""
    id_img = Image.new("RGB", (tex_w, tex_h), (0, 0, 0))
    dbg_img = Image.new("RGB", (tex_w, tex_h), (0, 0, 0))
    id_draw = ImageDraw.Draw(id_img)
    dbg_draw = ImageDraw.Draw(dbg_img)

    next_id = 1
    meta = {}  # pid -> (ann_id, corner_idx)

    def stamp(cx: float, cy: float, ann_id: int, corner_idx: int) -> int:
        nonlocal next_id
        pid = next_id
        next_id += 1
        r, g, b = id_to_rgb_u8(pid)
        rc, gc, bc = (pid * 53) % 255, (pid * 97) % 255, (pid * 193) % 255

        x = int(round(cx))
        y = int(round(cy))
        x0, y0 = x - radius_px, y - radius_px
        x1, y1 = x + radius_px, y + radius_px
        id_draw.ellipse([x0, y0, x1, y1], fill=(r, g, b), outline=None)
        dbg_draw.ellipse([x0, y0, x1, y1], fill=(rc, gc, bc), outline=None)

        meta[pid] = (int(ann_id), int(corner_idx))
        return pid

    for a in anns:
        aid = int(a.get("id", -1))
        if "bbox" in a and isinstance(a["bbox"], (list, tuple)) and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for ci, (cx, cy) in enumerate(corners):
                stamp(cx, cy, aid, ci)

    id_rgb01 = np.asarray(id_img, dtype=np.uint8).astype(np.float32) / 255.0
    dbg_u8 = np.asarray(dbg_img, dtype=np.uint8)
    return id_rgb01, meta, dbg_u8


def decode_id_centroids(id_img_u8: np.ndarray) -> dict:
    """decode_id_centroids(id_img_u8) -> dict: pid -> (x,y) centroid."""
    ids = rgb_u8_to_id(id_img_u8)
    unique = np.unique(ids)
    pos = {}
    for pid in unique:
        if pid == 0:
            continue
        ys, xs = np.where(ids == pid)
        if xs.size == 0:
            continue
        pos[int(pid)] = (float(xs.mean()), float(ys.mean()))
    return pos


def rebuild_bboxes_from_id_by_ann(meta: dict, pos: dict) -> dict:
    """rebuild_bboxes_from_id_by_ann(meta,pos) -> dict: ann_id -> (x,y,w,h)."""
    by_ann = {}
    for pid, (aid, ci) in meta.items():
        if pid in pos:
            by_ann.setdefault(aid, {})[ci] = pos[pid]

    out = {}
    for aid, corners in by_ann.items():
        if len(corners) != 4:
            continue
        pts = [corners[i] for i in range(4)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out[int(aid)] = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    return out


# =========================
# Patch resolver (bundle stores no paths)
# =========================
def _resolve_raw_patch_path(patch_bundle: dict) -> Path:
    """_resolve_raw_patch_path(patch_bundle) -> Path: dataset/create_dataset/patch_raw/<subdir>/<patch_name><ext>."""
    img_file = patch_bundle.get("img_file", None)
    patch_name = patch_bundle.get("patch_name", None)
    if not isinstance(img_file, str):
        raise ValueError("patch_bundle['img_file'] must be the original image file_name string")
    if not isinstance(patch_name, str) or not patch_name:
        raise ValueError("patch_bundle['patch_name'] must exist (set by save_patch('patch_raw', ...))")

    subdir = Path(img_file).parent
    ext = Path(img_file).suffix
    return PATCH_DIR / subdir / f"{patch_name}{ext}"

ANNS_JSON_NAME = "final_annotations.json"

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



import numpy as np


def rotate_raw_patch_bundle(patch_bundle: dict, rotation_angle_deg: float) -> dict:
    """rotate_raw_patch_bundle_for_saving(patch_bundle,rotation_angle_deg) -> dict: Rotate patch + raw anns for save_patch('patch_raw_rot_255')."""
    if "patch" not in patch_bundle:
        raise KeyError("patch_bundle['patch'] missing")
    if "anns" not in patch_bundle or not isinstance(patch_bundle["anns"], list):
        raise KeyError("patch_bundle['anns'] missing (expected raw image-space COCO anns)")

    patch = np.asarray(patch_bundle["patch"])
    if patch.ndim != 3 or patch.shape[2] < 3:
        raise ValueError(f"patch_bundle['patch'] must be HxWx3, got shape={patch.shape}")

    rot_patch, rot_anns = rotate_image_and_annotations(
        orig_img_u8=patch.astype(np.uint8),
        anns=patch_bundle["anns"],
        rotation_angle_deg=float(rotation_angle_deg),
    )

    out = dict(patch_bundle)
    out["patch"] = rot_patch
    out["anns"] = rot_anns
    out["rotation_angle_deg"] = float(rotation_angle_deg)
    out.pop("patch_name", None)     # ensure save_patch creates a new name for the rot split
    out.pop("anns_patch", None)     # ensure save_patch re-derives patch-local anns_patch
    return out



# =========================
# Public API
# =========================
def translate_image(patch_bundle: dict,
                       render_resolution: int,
                       sat_lat: float, sat_lon: float, sat_alt: float,
                       tgt_lat: float, tgt_lon: float, tgt_alt: float,
                       dem_seed: int,
                       show_plot: bool = False,
                       datetime_utc: datetime | None = None,
                       generate_nadir: bool = False,
                       rotation_angle_deg: float = 0.0) -> dict:

    """translate_image(patch_bundle,render_resolution,sat_lat,sat_lon,sat_alt,tgt_lat,tgt_lon,tgt_alt,show_plot,datetime_utc) -> dict: Render offnadir + translate anns."""
    if not isinstance(patch_bundle, dict):
        raise TypeError("translate_image expects patch_bundle as dict")

    img_path = _resolve_raw_patch_path(patch_bundle)
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing patch file: {img_path}")

    anns_path = _resolve_patch_anns_path(img_path)

    anns = patch_bundle.get("anns_patch", None)
    if not isinstance(anns, list):
        raise ValueError("patch_bundle['anns_patch'] missing. Call save_patch('nadir', patch_bundle) before translate_image.")

    orig_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    rot_rgb, rot_anns = rotate_image_and_annotations(orig_rgb, anns, rotation_angle_deg=float(rotation_angle_deg))
    tex_h, tex_w = rot_rgb.shape[:2]
    orig_overlay = draw_overlay(Image.fromarray(rot_rgb, mode="RGB").copy(), rot_anns)

    # write rotated inputs for generate_image (and for functions that read the image file)
    rot_img_path, rot_anns_path = _write_temp_rotated_inputs(rot_rgb, rot_anns, Path(anns_path))

    satellite = get_satellite(str(img_path), str(CSV_PATH), fixed_sat="WV3")
    gsd = get_spatial_res(str(img_path), str(CSV_PATH))

    # from here on, use rotated anns in-memory too
    anns = rot_anns

    dem_folder = main_path / "offnadir_imaging" / "create_DEM"
    dem_tiff_path = str(dem_folder / "input_dem_WV.tiff")
    dem_obj_path = str(dem_folder / "dem_mesh_WV.obj")

    get_DEM(str(rot_img_path), dem_tiff_path, gsd, WAVE_PROPERTIES, random_seed=int(dem_seed), waves=True, curvature=True, plot_DEM=False)
    convert_DEM(str(rot_img_path), dem_tiff_path, dem_obj_path, gsd, scale_km=False, print_output=False, plot_DEM=False)

    _ = get_band_data(satellite, str(main_path / "offnadir_imaging" / "spd_files"))

    dt = datetime_utc if datetime_utc is not None else DEFAULT_DATETIME_UTC

    if generate_nadir:
        sat_lat = tgt_lat
        sat_lon = tgt_lon

    satellite_ecef, target_ecef, sun_ecef = get_ecef_from_lat_lon(
        float(sat_lat), float(sat_lon), float(sat_alt),
        float(tgt_lat), float(tgt_lon), float(tgt_alt),
        dt,
    )

    is_dark, *_ = is_dark_from_sun_dir(
        target_ecef=target_ecef,
        sun_ecef=sun_ecef,
        threshold_deg=-18.0,
        model="wgs84",
        dir_type="target_to_sun",
    )
    if is_dark:
        raise RuntimeError("Dark hours, no image possible")

    satellite_local, target_local, _sun_dir, fov_deg, _off_nadir_rad, azimuth_rad = get_scene_characteristics(
        satellite_ecef, target_ecef, sun_ecef, tex_h, tex_w, gsd
    )

    scene_rotation = mi.ScalarTransform4f().rotate(
        axis=mi.ScalarVector3f(0, 0, 1),
        angle=math.degrees(-azimuth_rad),
    )
    scene_mirror = mi.ScalarTransform4f().scale([-1, 1, 1])
    to_world_scene = scene_rotation @ scene_mirror

    to_world_sensor_off = mi.ScalarTransform4f().look_at(
        origin=satellite_local,
        target=target_local,
        up=[0, 0, 1],
    )

    # ==========================================================
    # (A) SEGMENTATION: per-annotation mask render (NO global mask)
    # ==========================================================
    mask_res_hi = int(int(render_resolution) * MASK_SUPERSAMPLE)
    polys_off_by_ann: dict[int, list] = {}

    for ann in anns:
        ann_id = int(ann.get("id", -1))
        if ann_id < 0:
            continue

        mask_tex_rgb01 = build_binary_mask_texture_single(ann, tex_w, tex_h)

        mask_off_u8_hi = render_emission_texture(
            dem_obj_path=dem_obj_path,
            tex_rgb01=mask_tex_rgb01,
            to_world_scene=to_world_scene,
            to_world_sensor=to_world_sensor_off,
            fov_deg=fov_deg,
            res=mask_res_hi,
            spp=MASK_SPP,
            filter_type=MASK_FILTER,
        )

        mask_off_bin_hi = threshold_mask(mask_off_u8_hi, thr=127)
        mask_off_bin_hi = postprocess_binary_mask(mask_off_bin_hi, close_radius=MASK_CLOSE_RADIUS)
        mask_off_bin = downsample_binary_mask(mask_off_bin_hi, out_res=int(render_resolution))

        polys_off_by_ann[ann_id] = mask_to_polygons(mask_off_bin, simplify_eps=1.5)

    # ==========================================================
    # (B) BBOX: ID corners at off-nadir res
    # ==========================================================
    bbox_id_tex_rgb01, bbox_meta, bbox_dbg_u8 = build_bbox_id_texture(anns, tex_w, tex_h, radius_px=ID_POINT_RADIUS_PX)

    bbox_id_off_u8 = render_emission_texture(
        dem_obj_path=dem_obj_path,
        tex_rgb01=bbox_id_tex_rgb01,
        to_world_scene=to_world_scene,
        to_world_sensor=to_world_sensor_off,
        fov_deg=fov_deg,
        res=int(render_resolution),
        spp=ID_SPP,
        filter_type=ID_FILTER,
    )

    bbox_pos = decode_id_centroids(bbox_id_off_u8)
    boxes_off_by_ann = rebuild_bboxes_from_id_by_ann(bbox_meta, bbox_pos)
    boxes_off = list(boxes_off_by_ann.values())

    all_polys = [p for plist in polys_off_by_ann.values() for p in plist]
    contour_dbg = Image.new("RGB", (int(render_resolution), int(render_resolution)), (0, 0, 0))
    contour_dbg = draw_overlay_from_polys(contour_dbg, all_polys, boxes_off)

    if show_plot:
        fig = plt.figure(figsize=(22, 10))
        fig.add_subplot(2, 3, 1).imshow(orig_overlay); plt.axis("off"); plt.title("Original + annotation")
        fig.add_subplot(2, 3, 2).imshow(np.asarray(contour_dbg)); plt.axis("off"); plt.title("Per-instance contours + bboxes (final res)")
        fig.add_subplot(2, 3, 3).imshow(bbox_dbg_u8); plt.axis("off"); plt.title("BBox ID stamps (texture space)")
        fig.add_subplot(2, 3, 4).imshow(bbox_id_off_u8); plt.axis("off"); plt.title("Off-nadir bbox ID pass")
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # (C) Now run generate_image and overlay (debug only)
    # ==========================================================
    sensor_characteristics = {"resolution": int(render_resolution), "sample_count": int(SAMPLE_COUNT), "GSD": gsd}
    bools = dict(BOOLS_BASE)
    bools["plot_result"] = False

    if generate_nadir:
        bools['generate_nadir'] = True
    else:
        bools['generate_nadir'] = False

    DN255_texture, DN255_no_glint, DN255_glint, radiance_glint, rho_glint, rho_disp, black_mask_full, scale, offnadir_deg = generate_image(
        str(rot_img_path), str(rot_anns_path),
        satellite,
        float(sat_lat), float(sat_lon), float(sat_alt),
        float(tgt_lat), float(tgt_lon), float(tgt_alt),
        dt,
        sensor_characteristics,
        WAVE_PROPERTIES,
        bools,
        dem_seed,
    )

    if DN255_texture is None:
        raise RuntimeError("Renderer returned None (dark hours).")

    translated_u8 = to_uint8_rgb(DN255_texture)
    no_glint_u8 = to_uint8_rgb(DN255_no_glint) if DN255_no_glint is not None else np.zeros_like(translated_u8)
    glint_u8 = to_uint8_rgb(DN255_glint) if DN255_glint is not None else np.zeros_like(translated_u8)

    # reflectance display image is already float in [0,1]
    rho_u8 = (np.clip(rho_disp, 0.0, 1.0) * 255.0).astype(np.uint8) if rho_disp is not None else np.zeros_like(translated_u8)

    off_overlay = draw_overlay_from_polys(Image.fromarray(translated_u8, mode="RGB").copy(), all_polys, boxes_off)
    glint_overlay = draw_overlay_from_polys(Image.fromarray(glint_u8, mode="RGB").copy(), all_polys, boxes_off) if DN255_glint is not None else None
    rho_overlay = draw_overlay_from_polys(Image.fromarray(rho_u8, mode="RGB").copy(), all_polys, boxes_off) if rho_disp is not None else None

    if show_plot:
        fig2 = plt.figure(figsize=(18, 10))
        fig2.add_subplot(1, 4, 1).imshow(orig_overlay);
        plt.axis("off");
        plt.title("original + annotation")
        fig2.add_subplot(1, 4, 2).imshow(off_overlay);
        plt.axis("off");
        plt.title("translated + annotation (per-instance)")
        fig2.add_subplot(1, 4, 3).imshow(glint_overlay if glint_overlay is not None else glint_u8);
        plt.axis("off");
        plt.title("sun glint")
        fig2.add_subplot(1, 4, 4).imshow(rho_overlay if rho_overlay is not None else rho_u8);
        plt.axis("off");
        plt.title("TOA reflectance (scaled)")
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # (D) Translate annotations, preserving keys exactly
    # ==========================================================
    translated_anns: list[dict] = []
    for ann in anns:
        aid = int(ann.get("id", -1))
        polys = polys_off_by_ann.get(aid, [])
        bbox = boxes_off_by_ann.get(aid, None)

        a2 = dict(ann)  # keep everything
        a2["segmentation"] = polygons_to_coco_segmentation(polys)
        if bbox is not None:
            a2["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        translated_anns.append(a2)

    out = dict(patch_bundle)
    out["anns_patch"] = translated_anns
    out["render_resolution"] = int(render_resolution)

    # texture (uint8 RGB)
    out["texture_u8"] = translated_u8

    # radiance (float32 HxWx3) + preview (uint8 RGB)
    out["radiance"] = radiance_glint.astype(np.float32) if radiance_glint is not None else None
    out["radiance_u8"] = glint_u8  # preview (already DN255)

    # reflectance (float32 HxWx3) + preview (uint8 RGB)
    out["reflectance"] = rho_glint.astype(np.float32) if rho_glint is not None else None
    out["reflectance_u8"] = rho_u8  # preview made from rho_disp

    out["black_mask_full"] = black_mask_full
    out["scale"] = scale
    out["offnadir_deg"] = offnadir_deg
    out["rotation_angle_deg"] = rotation_angle_deg

    try:
        Path(rot_img_path).unlink(missing_ok=True)
        Path(rot_anns_path).unlink(missing_ok=True)
    except Exception:
        pass

    return out

